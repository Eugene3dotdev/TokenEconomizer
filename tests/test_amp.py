"""Amp host: the plugin's copy of a thread and Amp's own export, read as a trace; detection,
payload translation, the history source, the export fallback, no side call, the hooks end to
end, and the TypeScript plugin, both its structure and a run under Node with the real launcher.

Threads are synthetic, in the shapes Amp 0.0.1794283240 writes: the plugin API's
`ThreadMessage` (the plugin's copy) and `amp threads export` (structure taken from a real
export; no text from one), with fake ids and paths."""
import io
import json
import os
import re
import shutil
import subprocess
import time

import pytest

from tokeneconomizer import trace
from tokeneconomizer.microskills import observe
from tokeneconomizer.plugin import hook, hosts, sidecall
from tokeneconomizer.plugin.hosts import Amp
from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
from tokeneconomizer.plugin.store import Store
from tokeneconomizer.trace import sources

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(os.path.dirname(HERE), "plugin")
TS = os.path.join(PLUGIN, "amp", "tokeneconomizer.ts")
TID = "T-01a1e493-0000-7000-8000-000000000001"
FACTS = "Facts learned about this repository:\n- run tests with python -m pytest"


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "te"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "TE_MICROSKILLS": "envfacts",
            "HOME": str(tmp_path / "home"), "GIT_CEILING_DIRECTORIES": str(tmp_path),
            "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    (r / "pyproject.toml").write_text("[project]\nname='w'\n")
    return str(r)


def call(env, p):
    out = io.StringIO()
    hook.main(["--host", "amp"], io.StringIO(json.dumps(p)), out, env)
    return json.loads(out.getvalue().splitlines()[-1])


def store(env, repo):
    return Store(repo_state_dir(repo_identity(repo, env), "amp", env))


USAGE = {"model": "openai/gpt-x", "inputTokens": 0, "outputTokens": 40,
         "cacheCreationInputTokens": 900, "cacheReadInputTokens": 100}
TRAP = [("pytest -q tests/test_a.py", "bash: pytest: command not found", 127),
        ("python -m pytest -q tests/test_a.py", "1 passed in 0.1s", 0)]


def plugin_messages(prompt, calls=TRAP, workdir=None):
    """A turn's messages as `agent.end` hands them over (the plugin API's ThreadMessage)."""
    msgs = [{"role": "user", "id": "M-u1", "content": [{"type": "text", "text": prompt}]}]
    for n, (cmd, out, code) in enumerate(calls):
        msgs.append({"role": "assistant", "id": f"M-a{n}", "stopReason": "tool_use",
                     "usage": USAGE, "content": [
                         {"type": "thinking", "thinking": "..."},
                         {"type": "tool_use", "id": f"call_{n}", "name": "shell_command",
                          "input": dict({"command": cmd}, **({"workdir": workdir}
                                                             if workdir else {}))}]})
        msgs.append({"role": "user", "id": f"M-r{n}", "content": [
            {"type": "tool_result", "toolUseID": f"call_{n}", "status": "done",
             "output": {"output": out, "exitCode": code}}]})
    msgs.append({"role": "assistant", "id": "M-end", "stopReason": "end_turn", "usage": USAGE,
                 "content": [{"type": "text", "text": "done"}]})
    return msgs


def write_copy(path, cwd, messages, mode="interactive", created=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tid = os.path.basename(path).split(".")[0]
    lines = [{"type": "thread", "id": tid, "cwd": cwd,
              "created": created or int(time.time() * 1000), "mode": mode}]
    lines += [dict({"type": "message"}, **m) for m in messages]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def write_export(path, cwd, prompt, client="CLI", calls=TRAP):
    """`amp threads export` output: numeric message ids, results under `run`."""
    msgs, n = [{"role": "user", "messageId": 0, "content": [{"type": "text", "text": prompt}],
                "meta": {"sentAt": 1, "fromExecutorClient": True}}], 0
    for n, (cmd, out, code) in enumerate(calls):
        msgs.append({"role": "assistant", "messageId": 2 * n + 1, "usage": dict(
            USAGE, totalInputTokens=1000), "state": {"type": "complete", "stopReason": "tool_use"},
            "content": [{"type": "tool_use", "id": f"call_{n}", "name": "shell_command",
                         "complete": True, "input": {"command": cmd, "workdir": cwd}}]})
        msgs.append({"role": "user", "messageId": 2 * n + 2, "content": [
            {"type": "tool_result", "toolUseID": f"call_{n}",
             "run": {"status": "done", "result": {"output": out, "exitCode": code}}}]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"v": 17, "id": os.path.basename(path)[:-len(".json")], "created": 1763300000000,
                   "env": {"initial": {"workingDirectory": "file://" + cwd.replace(" ", "%20"),
                                       "platform": {"client": client}}},
                   "messages": msgs}, fh)
    return path


def _shell(path, tmp_path):
    out = str(tmp_path / "trace.jsonl")
    assert trace.amp_thread_to_claude(path, out) is not None
    return out, [c for c in trace.load_trajectory(out) if c[1].get("command")]


# ---- the trace -----------------------------------------------------------------------------
def test_the_plugins_copy_reads_as_a_trace_without_what_the_plugin_appended(tmp_path, repo):
    prompt = f"fix the tests\n\n<tokeneconomizer>\n{FACTS}\n</tokeneconomizer>"
    msgs = plugin_messages(prompt)
    path = write_copy(str(tmp_path / "amp-threads" / f"{TID}.jsonl"), repo,
                      msgs + msgs[3:5])              # a retried turn writes two messages again
    out, shell = _shell(path, tmp_path)
    assert [observe.unwrap(a["command"]) for _, a, _, _ in shell] == [c for c, _, _ in TRAP]
    assert [ok for *_, ok in shell] == [False, True]
    assert "command not found" in shell[0][2]
    lines = [json.loads(x) for x in open(out)]
    users = [x["message"]["content"] for x in lines if x["type"] == "user"
             and isinstance(x["message"]["content"], str)]
    assert users == ["fix the tests"]                     # the person's words only
    assert all(x.get("cwd") == repo for x in lines if x["type"] == "assistant")
    assert trace.amp_prompt_count(path) == 1
    u = trace.amp_thread_usage(path)
    assert u == {"input_tokens": 0, "cache_read_input_tokens": 300,
                 "cache_creation_input_tokens": 2700, "output_tokens": 120, "requests": 3,
                 "model": "openai/gpt-x"}


def test_an_export_reads_the_same_and_names_its_directory_and_mode(tmp_path):
    cwd = str(tmp_path / "my repo")
    path = write_export(str(tmp_path / f"{TID}.json"), cwd, "fix the tests",
                        client="CLI Execute Mode")
    head = trace.amp_thread_head(path)
    assert head == {"id": TID, "cwd": cwd, "created": 1763300000.0, "headless": True}
    _, shell = _shell(path, tmp_path)
    assert [ok for *_, ok in shell] == [False, True]
    assert trace.amp_prompt_count(path) == 1
    assert trace.amp_thread_usage(path)["requests"] == 2


def test_a_refused_call_an_error_and_a_tool_result_only_message_are_read_as_amp_means(tmp_path):
    assert trace.amp_result({"status": "done", "output": "Tool rejected by plugin: no"})[2]
    assert trace.amp_result({"status": "error", "error": "boom"}) == ("boom", None, True)
    assert trace.amp_result({"run": {"status": "done", "result": "plain"}}) == (
        "plain", None, False)
    msgs = [{"role": "user", "id": "u", "content": [
        {"type": "tool_result", "toolUseID": "x", "status": "done", "output": "o"}]}]
    path = write_copy(str(tmp_path / f"{TID}.jsonl"), "/r", msgs)
    assert trace.amp_prompt_count(path) == 0


def test_amps_tools_take_claude_codes_names():
    assert trace.amp_tool("shell_command", {"command": "rtk pytest -q", "workdir": "/r"}) == (
        "Bash", {"command": "pytest -q"})
    assert trace.amp_tool("edit_file", {"path": "/r/a.py", "old_str": "x"}) == (
        "Edit", {"file_path": "/r/a.py"})
    assert trace.amp_tool("create_file", {"path": "/r/b.py", "content": "x"}) == (
        "Write", {"file_path": "/r/b.py"})
    assert trace.amp_tool("mcp__x__y", {"a": 1}) == ("mcp__x__y", {"a": 1})


# ---- the history source ----------------------------------------------------------------------
def test_the_source_lists_copies_and_exports_and_reads_the_newer_of_a_pair(env, tmp_path, repo):
    src = sources.SOURCES["amp"]
    root = os.path.join(env["TOKENECONOMIZER_HOME"], "amp-threads")
    a = write_copy(os.path.join(root, f"{TID}.jsonl"), repo, plugin_messages("go"))
    b = write_export(os.path.join(root, f"{TID}.json"), repo, "go")
    other = write_copy(os.path.join(root, "T-2.jsonl"), repo, plugin_messages("go"),
                       mode="execute")
    open(os.path.join(root, "notes.txt"), "w").close()
    os.utime(a, (time.time() - 60, time.time() - 60))
    logs = {x["id"]: x["path"] for x in src.logs(env)}
    assert logs == {TID: b, "T-2": other}
    assert src.head(other)["headless"] and src.head(other)["cwd"] == repo
    os.utime(b, (time.time() - 120, time.time() - 120))
    assert {x["id"]: x["path"] for x in src.logs(env)}[TID] == a


# ---- the host --------------------------------------------------------------------------------
def test_amp_is_named_by_its_flag_or_its_copys_path():
    assert isinstance(hosts.detect({}, hint="amp"), Amp) and "amp" in hosts.HOSTS
    assert isinstance(hosts.detect({"transcript_path": f"/h/.tokeneconomizer/amp-threads/"
                                                       f"{TID}.jsonl"}), Amp)
    assert Amp().headless({"TE_AMP_MODE": "execute"})
    assert not Amp().headless({"TE_AMP_MODE": "interactive"}) and not Amp().headless({})


def test_a_shell_result_is_read_with_its_exit_code():
    name, args, text, code = Amp().tool_call({
        "tool_name": "shell_command", "tool_input": {"command": "pytest -q", "workdir": "/r"},
        "tool_response": {"content": "bash: pytest: command not found", "isError": True,
                          "exit_code": 127}})
    assert (name, args, code) == ("bash", {"command": "pytest -q"}, 127)
    assert "command not found" in text
    assert Amp().tool_call({"tool_name": "edit_file", "tool_input": {"path": "/r/a.py"},
                            "tool_response": {"content": "ok", "isError": False}})[:2] == (
        "Edit", {"file_path": "/r/a.py"})


def test_amp_makes_no_side_call(monkeypatch):
    started = []
    monkeypatch.setattr(sidecall.subprocess, "run", lambda *a, **k: started.append(a))
    assert "amp" in sidecall.NO_SIDE_CALLS
    assert sidecall.runner_for("amp", type("C", (), {"paid_model": "m"})()) is None
    assert started == []


def test_a_copy_with_no_prompt_is_completed_from_amps_export(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path / "te"))
    copy = write_copy(str(tmp_path / "te" / "amp-threads" / f"{TID}.jsonl"), "/r", [])
    body = open(write_export(str(tmp_path / "x" / f"{TID}.json"), "/r", "fix it")).read()
    seen = []

    def fake(argv, **kw):
        seen.append((argv, kw["env"].get("TE_SIDECALL")))
        return subprocess.CompletedProcess(argv, 0, body, "")
    monkeypatch.setattr(subprocess, "run", fake)
    got = Amp().complete(copy, TID)
    assert got == str(tmp_path / "te" / "amp-threads" / f"{TID}.json")
    assert seen == [(["amp", "threads", "export", TID], "1")]
    assert trace.amp_prompt_count(got) == 1
    # Not signed in, or offline: the copy stays what the fold reads.
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 1, "", "not signed in"))
    os.remove(got)
    assert Amp().complete(copy, TID) == copy
    # A copy that has the thread is read as it is, with no call to Amp.
    full = write_copy(str(tmp_path / "te" / "amp-threads" / "T-3.jsonl"), "/r",
                      plugin_messages("go"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("called Amp"))
    assert Amp().complete(full, "T-3") == full


# ---- the hooks, end to end -------------------------------------------------------------------
def test_amp_learns_from_its_thread_copies_and_hands_the_manual_to_the_next_thread(
        env, repo, tmp_path):
    root = os.path.join(env["TOKENECONOMIZER_HOME"], "amp-threads")
    starts = []
    for i in range(5):
        tid = f"T-{i}"
        copy = write_copy(os.path.join(root, f"{tid}.jsonl"), repo,
                          plugin_messages(f"fix bug {i}"))
        base = {"session_id": tid, "transcript_path": copy, "cwd": repo}
        starts.append(call(env, dict(base, hook_event_name="SessionStart", source="startup")))
        call(env, dict(base, hook_event_name="SessionEnd", reason="dispose"))
    assert all(s == {} for s in starts[:4])
    manual = starts[4]["hookSpecificOutput"]["additionalContext"]
    assert "`python -m pytest …` works." in manual
    rec = store(env, repo).read_lines("ledger.jsonl")[0]
    assert rec["host"] == "amp" and rec["requests"] == 3
    # A thread the plugin already has a copy of carries its manual in its first message.
    again = call(env, {"session_id": "T-4", "transcript_path": "", "cwd": repo,
                       "hook_event_name": "SessionStart", "source": "resume"})
    assert "hookSpecificOutput" not in again


def test_the_guards_backstop_redirects_an_amp_turn_and_never_stops_it(env, repo):
    def turn(key, calls, mode):
        base = {"session_id": TID, "transcript_path": "", "cwd": repo, "prompt_id": key}
        # The guard's backstop: Amp has no judge to ask (`NO_SIDE_CALLS`).
        e = dict(env, TE_AMP_MODE=mode, TE_LOOPGUARD="on")
        call(e, dict(base, hook_event_name="UserPromptSubmit", prompt="go"))
        outs = [call(e, dict(base, hook_event_name="PostToolUse", tool_name=n, tool_input=a,
                             tool_response=r)) for n, a, r in calls]
        call(e, dict(base, hook_event_name="Stop", stop_hook_active=False))
        return outs
    green = [("edit_file", {"path": "/r/a.py"}, {"content": "ok", "isError": False})] * 8
    for k in range(5):
        turn(f"g{k}", green, "execute")
    fail = "FAILED tests/test_a.py::test_x - AssertionError: assert 1 == 2"
    stuck = [("edit_file", {"path": "/r/pkg/a.py"}, {"content": "ok", "isError": False})
             if k % 2 == 0 else ("shell_command", {"command": "pytest -q", "workdir": "/r"},
                                 {"content": fail, "isError": True, "exit_code": 1})
             for k in range(40)]
    outs = turn("stuck", stuck, "execute")
    redirect = [o["hookSpecificOutput"]["additionalContext"] for o in outs
                if "hookSpecificOutput" in o]
    assert len(redirect) == 1 and "Process check (automated)" in redirect[0]
    assert [o.get("systemMessage") for o in outs if o] == [
        "TokenEconomizer nudged this turn (cum-backstop): the same failure kept recurring."]
    assert not any("continue" in o or "stopReason" in o for o in outs)     # never a stop
    assert (Amp().stop_hint(True), Amp().stop_hint(False)) == (
        "Press Ctrl+C to end the run.", "Press Esc to cancel the turn.")


# ---- the TypeScript plugin -------------------------------------------------------------------
def test_the_amp_plugin_drives_the_shared_launchers_on_the_dispatchers_events():
    src = open(TS, encoding="utf-8").read()
    assert 'join(HERE, "..", "scripts")' in src and '"te-hook", ["--host", "amp"]' in src
    # Not `tool.call`, the one event that can refuse a call: the plugin guards nothing.
    assert set(re.findall(r'amp\.on\("([\w.]+)"', src)) == {
        "session.start", "agent.start", "tool.result", "agent.end"}
    assert "amp.onDispose(" in src
    # Amp shows a static `description` literal of at most 300 characters, else none.
    m = re.search(r'export const description =\s*"([^"\\]*)";', src)
    assert m and len(m.group(1)) <= 300
    assert "@ampcode/plugin" in src and "import type" in src


NODE_DRIVER = r"""
const { pathToFileURL, fileURLToPath } = await import("node:url");
const [plugin, repo, tid] = process.argv.slice(1);
const mod = await import(pathToFileURL(plugin).href);
const handlers = {}, notices = [], commands = [], disposers = [];
const ui = { notify: async (t) => { notices.push(t); } };
const amp = {
  on: (name, fn) => { handlers[name] = fn; },
  onDispose: (fn) => { disposers.push(fn); },
  registerCommand: (id, opts) => { commands.push(`${opts.category}: ${opts.title}`); },
  system: { workspaceRoot: pathToFileURL(repo).href },
  helpers: { filePathFromURI: (u) => fileURLToPath(u) },
  ui,
};
mod.default(amp);
const thread = { id: tid };
const ctx = { ui, thread: { id: tid, appendUserMessage: async () => {} } };
await handlers["session.start"]({ thread }, ctx);
const start = await handlers["agent.start"]({ thread, message: "fix the tests", id: "M-u1" }, ctx);
const tool = await handlers["tool.result"]({ thread, toolUseID: "call_0", tool: "shell_command",
  input: { command: "pytest -q", workdir: repo }, status: "done",
  output: { output: "bash: pytest: command not found", exitCode: 127 } }, ctx);
await handlers["agent.end"]({ thread, message: "fix the tests", id: "M-u1", status: "done",
  messages: JSON.parse(process.env.TE_TEST_MESSAGES) }, ctx);
for (const fn of disposers) await fn();
console.log(JSON.stringify({ events: Object.keys(handlers).sort(), start, tool: tool ?? null,
  commands, modes: [mod.ampMode(["amp", "-x", "hi"], {}), mod.ampMode(["amp"], {}),
                    mod.ampMode(["amp", "-x"], { TE_AMP_MODE: "interactive" })],
  nudged: mod.withNudge({ output: "o", exitCode: 1 }, "n"), text: mod.toolText(["a", { type: "text", text: "b" }]) }));
"""


def test_the_amp_plugin_runs_under_node_with_the_real_launcher(env, repo, tmp_path):
    """The plugin file itself, with a stand-in for Amp's plugin API and the real `te-hook`: the
    manual is appended to the first prompt inside the plugin's tags, the turn's messages land in
    the copy the fold reads, and the thread is handed over when Amp disposes of the plugin."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    store(env, repo).write("injection.json", {"rendered": FACTS, "rendered_ids": ["a"]})
    run_env = dict(os.environ, **env, TE_AMP_MODE="interactive",
                   TE_TEST_MESSAGES=json.dumps(plugin_messages("fix the tests")))
    run_env.pop("TE_FOLD_SYNC")
    run_env.pop("TE_SIDECALL", None)
    run_env.pop("TE_DISABLE", None)
    run = subprocess.run([node, "--experimental-strip-types", "--no-warnings",
                          "--input-type=module", "-e", NODE_DRIVER, TS, repo, TID],
                         capture_output=True, text=True, timeout=120, env=run_env)
    if run.returncode and "strip-types" in run.stderr:
        pytest.skip("this node cannot load TypeScript")
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["events"] == ["agent.end", "agent.start", "session.start", "tool.result"]
    msg = out["start"]["message"]
    assert msg["display"] is False
    assert msg["content"].startswith("<tokeneconomizer>\n") and FACTS in msg["content"]
    assert msg["content"].endswith("\n</tokeneconomizer>")
    assert out["tool"] is None                            # a cold guard changes nothing
    assert out["commands"] == ["TokenEconomizer: status", "TokenEconomizer: micro-skills",
                               "TokenEconomizer: history"]
    assert out["modes"] == ["execute", "interactive", "interactive"]
    assert out["nudged"] == {"output": "o\n\nn", "exitCode": 1}
    assert out["text"] == {"text": "a\nb"}
    copy = os.path.join(env["TOKENECONOMIZER_HOME"], "amp-threads", f"{TID}.jsonl")
    head = trace.amp_thread_head(copy)
    assert head["id"] == TID and head["cwd"] == repo and not head["headless"]
    assert trace.amp_prompt_count(copy) == 1
    sess = store(env, repo).read(f"sessions/{TID}.json")
    assert sess["host"] == "amp" and len(sess["turns"]) == 1
    assert sess["ended"] and sess["end_reason"] == "dispose"
