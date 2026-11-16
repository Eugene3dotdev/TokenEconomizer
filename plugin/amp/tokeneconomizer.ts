/**
 * TokenEconomizer for Amp (ampcode CLI).
 *
 * Amp has no hook protocol -- plugins are functions on its event bus, run by Bun inside Amp --
 * so this plugin speaks the one the Claude Code, Codex and Pi integrations use, on Amp's
 * behalf: every event below becomes one call of the shared dispatcher (`../scripts/te-hook
 * --host amp`, JSON in, JSON out), and the answer is applied with Amp's own levers. All of the
 * mechanism logic lives in the dispatcher; this file only translates.
 *
 *   session.start   -> SessionStart       the learned manual, held for the thread's next prompt
 *                                         (a thread this plugin has a copy of is a resumption:
 *                                         its first message already carries the manual)
 *   agent.start     -> UserPromptSubmit   opens a prompt-turn; the manual (once per thread) and
 *                                         the prompt's own context are appended to the person's
 *                                         message, the one place Amp lets a plugin add context
 *   tool.result     -> PostToolUse        the guard; a redirect is appended to the result the
 *                                         model reads next, a still-stuck notice is shown
 *   agent.end       -> Stop               closes the turn; the turn's messages are appended to
 *                                         the plugin's copy of the thread, whose path is sent
 *   onDispose, or the process exiting
 *                   -> SessionEnd         hands each thread to a detached fold. Amp has no
 *                                         session end, and gives `onDispose` about 3 seconds; a
 *                                         thread whose end is missed is folded at a later start,
 *                                         once its copy has been quiet for 10 minutes
 *
 * Amp keeps no thread on the machine, so the copy (`<state root>/amp-threads/<thread id>.jsonl`,
 * private to its owner) is what the fold and the history import read. The plugin process runs
 * in the plugin's own directory: the repository is Amp's workspace root. In a workspace the
 * dispatcher answers is excluded (`privacy.py`: the Messages and Photos libraries, a family
 * folder, the PARA reference and archive trees), no copy is kept and no further event is sent.
 *
 * Like the hooks on the other agents it fails open: a dispatcher that is missing, slow or
 * broken answers "no change". It does nothing under TE_SIDECALL=1 or TE_DISABLE=1, and it
 * never subscribes to `tool.call`, the event that can refuse a call: the plugin guards nothing.
 * The command palette's `TokenEconomizer: ...` entries run the `te` command line for the
 * workspace.
 */

import { spawn, spawnSync } from "node:child_process";
import {
	appendFileSync,
	existsSync,
	mkdirSync,
	readFileSync,
	realpathSync,
	rmSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { PluginAPI } from "@ampcode/plugin";

export const description =
	"TokenEconomizer: repository facts learned from finished sessions, given at a thread's first prompt, and a loop guard on tool results.";

const HERE = (() => {
	try {
		return dirname(realpathSync(fileURLToPath(import.meta.url)));
	} catch {
		return __dirname;
	}
})();
const SCRIPTS = join(HERE, "..", "scripts");
const HOOK_TIMEOUT_MS = 60_000;
// Amp gives `onDispose` about 3 s; the handover must finish inside it.
const DISPOSE_TIMEOUT_MS = 2_500;
const TAG = "tokeneconomizer";

const truthy = (v: string | undefined) =>
	["1", "true", "yes", "on"].includes((v ?? "").trim().toLowerCase());
const disabled = () =>
	truthy(process.env.TE_SIDECALL) || truthy(process.env.TE_DISABLE);

/** The arguments of the Amp process that started this plugin: its runtime is a child of it. */
function parentArgs(): string[] {
	try {
		return readFileSync(`/proc/${process.ppid}/cmdline`, "utf8")
			.split("\0")
			.filter(Boolean);
	} catch {
		try {
			const out = spawnSync("ps", ["-o", "args=", "-p", String(process.ppid)], {
				encoding: "utf8",
				timeout: 2_000,
			});
			return String(out.stdout ?? "")
				.trim()
				.split(/\s+/)
				.filter(Boolean);
		} catch {
			return [];
		}
	}
}

/** `amp -x` (`--execute`) is headless: one prompt, then the process ends. */
export function ampMode(
	args: string[] = parentArgs(),
	env = process.env,
): string {
	if (env.TE_AMP_MODE) return env.TE_AMP_MODE;
	return args.some(
		(a) => a === "-x" || a === "--execute" || a.startsWith("--execute="),
	)
		? "execute"
		: "interactive";
}
const MODE = ampMode();

/** The program and arguments that run a launcher. The launchers are POSIX `sh`, which Windows
 * cannot start; there the bootstrap they run is started with Python itself (`TE_PYTHON`, else
 * `python`). */
export function launch(
	script: string,
	args: string[],
	platform: string = process.platform,
): [string, string[]] {
	if (platform !== "win32") return [join(SCRIPTS, script), args];
	const kind = script === "te-hook" ? "hook" : "cli";
	return [
		process.env.TE_PYTHON || "python",
		[join(SCRIPTS, "te_bootstrap.py"), kind, ...args],
	];
}

/** The plugin runs inside the Amp binary, which is also what `amp threads export` should run. */
const childEnv = (extra: Record<string, string> = {}) => ({
	...process.env,
	PYTHONUTF8: "1",
	TE_AMP_MODE: MODE,
	TE_AMP_CLI: process.env.TE_AMP_CLI || process.execPath,
	...extra,
});

/** Run one of the plugin's launchers with `input` on stdin; resolve to its stdout. */
function run(
	script: string,
	args: string[],
	input: string,
	extra: Record<string, string> = {},
): Promise<string> {
	return new Promise((done) => {
		let out = "";
		let child: any;
		try {
			const [program, argv] = launch(script, args);
			child = spawn(program, argv, {
				env: childEnv(extra),
				stdio: ["pipe", "pipe", "ignore"],
				windowsHide: true,
			});
		} catch {
			done("");
			return;
		}
		const timer = setTimeout(() => {
			child.kill("SIGKILL");
			done("");
		}, HOOK_TIMEOUT_MS);
		child.stdout.setEncoding("utf8");
		child.stdout.on("data", (d: string) => (out += d));
		child.on("error", () => {
			clearTimeout(timer);
			done("");
		});
		child.on("close", () => {
			clearTimeout(timer);
			done(out);
		});
		child.stdin.on("error", () => {});
		child.stdin.end(input);
	});
}

/** One dispatcher call; `{}` (no change) on any failure. */
async function hook(payload: Record<string, unknown>): Promise<any> {
	if (disabled()) return {};
	const out = await run("te-hook", ["--host", "amp"], JSON.stringify(payload));
	try {
		return JSON.parse(out.trim().split("\n").pop() || "{}") ?? {};
	} catch {
		return {};
	}
}

/** The same call, blocking: for a process that is being disposed of or is exiting. */
function hookSync(
	payload: Record<string, unknown>,
	timeout = DISPOSE_TIMEOUT_MS,
) {
	if (disabled()) return;
	try {
		const [program, argv] = launch("te-hook", ["--host", "amp"]);
		spawnSync(program, argv, {
			env: childEnv(),
			input: JSON.stringify(payload),
			timeout,
			stdio: ["pipe", "ignore", "ignore"],
			windowsHide: true,
		});
	} catch {
		/* fail open */
	}
}

const stateRoot = () =>
	resolve(
		process.env.TOKENECONOMIZER_HOME || join(homedir(), ".tokeneconomizer"),
	);

/** The dispatcher's answer for a workspace whose sessions are never kept (`privacy.EXCLUDED_KEY`). */
export const EXCLUDED_KEY = "tokeneconomizer_excluded";

export const copyPath = (threadID: string) =>
	join(stateRoot(), "amp-threads", `${threadID}.jsonl`);

/** The thread's copy, started with its header line where there is none yet. */
function ensureCopy(threadID: string, cwd: string): string {
	const file = copyPath(threadID);
	try {
		if (!existsSync(file)) {
			mkdirSync(dirname(file), { recursive: true, mode: 0o700 });
			appendFileSync(
				file,
				JSON.stringify({
					type: "thread",
					id: threadID,
					cwd,
					created: Date.now(),
					mode: MODE,
				}) + "\n",
				{ mode: 0o600 },
			);
		}
		return file;
	} catch {
		return "";
	}
}

/** Append a turn's messages to the copy; a message seen again (a retried turn) is read once,
 * the later line winning (`trace/amp.py`). */
function appendMessages(
	threadID: string,
	cwd: string,
	messages: unknown[],
): string {
	const file = ensureCopy(threadID, cwd);
	if (!file) return "";
	try {
		const lines = (messages ?? [])
			.filter((m: any) => m && (m.role === "user" || m.role === "assistant"))
			.map((m: any) => JSON.stringify({ type: "message", ...m }) + "\n")
			.join("");
		if (lines) appendFileSync(file, lines, { mode: 0o600 });
	} catch {
		/* fail open: the fold falls back to Amp's own export */
	}
	return file;
}

/** A tool result's text and exit code. The shell tool's result is `{output, exitCode}`. */
export function toolText(output: unknown): { text: string; exit?: number } {
	if (output == null) return { text: "" };
	if (typeof output === "string") return { text: output };
	if (Array.isArray(output)) {
		return {
			text: output
				.map((b: any) =>
					typeof b === "string" ? b : typeof b?.text === "string" ? b.text : "",
				)
				.filter(Boolean)
				.join("\n"),
		};
	}
	if (typeof output === "object") {
		const o = output as Record<string, unknown>;
		const exit = typeof o.exitCode === "number" ? o.exitCode : undefined;
		if (typeof o.output === "string") return { text: o.output, exit };
		try {
			return { text: JSON.stringify(output), exit };
		} catch {
			return { text: String(output), exit };
		}
	}
	return { text: String(output) };
}

/** The result with `nudge` appended where the model reads it; undefined for a shape it cannot
 * be added to. */
export function withNudge(output: unknown, nudge: string): unknown {
	if (output == null) return nudge;
	if (typeof output === "string") return `${output}\n\n${nudge}`;
	if (Array.isArray(output)) return [...output, { type: "text", text: nudge }];
	if (
		typeof output === "object" &&
		typeof (output as any).output === "string"
	) {
		return {
			...(output as object),
			output: `${(output as any).output}\n\n${nudge}`,
		};
	}
	return undefined;
}

export default function tokeneconomizer(amp: PluginAPI) {
	const cwd = (() => {
		try {
			const root = amp.system.workspaceRoot;
			if (root) return amp.helpers.filePathFromURI(root);
		} catch {
			/* no workspace */
		}
		return process.env.PWD || process.cwd();
	})();
	const starts = new Map<string, Promise<void>>();
	const manuals = new Map<string, string>(); // threadID -> manual not yet delivered
	const prompts = new Map<string, number>();
	const turns = new Map<string, string>();
	const open = new Set<string>();
	let excluded = false; // the workspace is an excluded directory: nothing is copied or sent

	const base = (threadID: string) => ({
		session_id: threadID,
		transcript_path: copyPath(threadID),
		cwd,
	});

	const notify = async (ctx: any, text: unknown) => {
		if (typeof text !== "string" || !text) return;
		try {
			await (ctx?.ui ?? amp.ui).notify(text);
		} catch {
			console.error(text);
		}
	};

	const start = (threadID: string, ctx: any): Promise<void> => {
		let p = starts.get(threadID);
		if (!p) {
			p = (async () => {
				open.add(threadID);
				prompts.set(threadID, 0);
				const resumed = existsSync(copyPath(threadID));
				const out = await hook({
					...base(threadID),
					hook_event_name: "SessionStart",
					source: resumed ? "resume" : "startup",
				});
				if (out?.[EXCLUDED_KEY]) {
					excluded = true;
					try {
						rmSync(copyPath(threadID), { force: true });
					} catch {
						/* fail open */
					}
					return;
				}
				ensureCopy(threadID, cwd);
				manuals.set(threadID, out?.hookSpecificOutput?.additionalContext ?? "");
				await notify(ctx, out?.systemMessage);
			})();
			starts.set(threadID, p);
		}
		return p;
	};

	const end = (threadID: string, reason: string, sync = false) => {
		if (!open.delete(threadID)) return;
		starts.delete(threadID);
		if (excluded) return;
		const payload = {
			...base(threadID),
			hook_event_name: "SessionEnd",
			reason,
		};
		if (sync) hookSync(payload);
		else return hook(payload);
	};

	amp.on("session.start", async (event, ctx) => {
		const id = event?.thread?.id;
		if (id) await start(id, ctx);
	});

	amp.on("agent.start", async (event, ctx) => {
		const id = event?.thread?.id;
		if (!id) return;
		await start(id, ctx);
		if (excluded) return;
		const n = (prompts.get(id) ?? 0) + 1;
		prompts.set(id, n);
		const turn = `${id}:${n}`;
		turns.set(id, turn);
		const out = await hook({
			...base(id),
			hook_event_name: "UserPromptSubmit",
			prompt_id: turn,
			prompt: event.message ?? "",
		});
		await notify(ctx, out?.systemMessage);
		const parts = [
			manuals.get(id),
			out?.hookSpecificOutput?.additionalContext,
		].filter((t): t is string => typeof t === "string" && t.length > 0);
		manuals.set(id, "");
		if (!parts.length) return;
		return {
			message: {
				content: `<${TAG}>\n${parts.join("\n\n")}\n</${TAG}>`,
				display: false,
			},
		};
	});

	amp.on("tool.result", async (event, ctx) => {
		const id = event?.thread?.id;
		if (!id || excluded) return;
		const { text, exit } = toolText(event.output);
		const shown =
			text ||
			(event.status !== "done" ? String(event.error ?? event.status) : "");
		const out = await hook({
			...base(id),
			hook_event_name: "PostToolUse",
			prompt_id: turns.get(id) ?? "",
			tool_name: event.tool,
			tool_use_id: event.toolUseID,
			tool_input: event.input ?? {},
			tool_response: {
				content: shown,
				isError: event.status !== "done" || (exit !== undefined && exit !== 0),
				exit_code: exit,
			},
		});
		await notify(ctx, out?.systemMessage);
		const nudge = out?.hookSpecificOutput?.additionalContext;
		if (typeof nudge !== "string" || !nudge) return;
		const output = withNudge(event.output, nudge);
		if (output === undefined) {
			try {
				await ctx.thread.appendUserMessage(
					{ type: "user-message", content: nudge },
					{ steer: true },
				);
			} catch {
				/* fail open */
			}
			return;
		}
		if (event.status === "done") return { status: "done", output };
		return { status: event.status, error: event.error, output };
	});

	amp.on("agent.end", async (event, ctx) => {
		const id = event?.thread?.id;
		if (!id) return;
		if (!open.has(id)) await start(id, ctx);
		if (excluded) return;
		appendMessages(id, cwd, event.messages ?? []);
		// An `error` end can come in the middle of a turn that Amp then retries; the turn is
		// closed by its real end, or by the next prompt.
		if (event.status === "error") return;
		await hook({
			...base(id),
			hook_event_name: "Stop",
			prompt_id: turns.get(id) ?? "",
			stop_hook_active: false,
		});
		// A headless run ends right after its one turn, so the thread is handed over now.
		if (MODE === "execute") await end(id, "execute");
	});

	const handOver = (reason: string) => {
		for (const id of [...open]) end(id, reason, true);
	};
	amp.onDispose(() => handOver("dispose"));
	process.once("exit", () => handOver("exit"));

	const command = (title: string, argv: string[]) =>
		amp.registerCommand(
			`te-${argv.join("-")}`,
			{
				title,
				category: "TokenEconomizer",
				description: `te ${argv.join(" ")} for this workspace`,
			},
			async (ctx: any) => {
				const out = await run(
					"te",
					["--host", "amp", "--cwd", cwd, ...argv],
					"",
					{
						TE_SESSION_ID: String(ctx?.thread?.id ?? ""),
					},
				);
				await notify(
					ctx,
					out.trim() || "te: no output (is Python 3.10+ on PATH?)",
				);
			},
		);
	command("status", ["status"]);
	command("micro-skills", ["microskills"]);
	command("history", ["history", "status"]);
}
