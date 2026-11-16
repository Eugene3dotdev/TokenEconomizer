"""TokenEconomizer for ZeroClaw (zeroclaw-labs/zeroclaw): a relay in front of `zeroclaw acp`.

    python3 /path/to/TokenEconomizer/plugin/zeroclaw/te_zeroclaw_acp.py acp [--config-dir DIR]

An ACP client (an editor, a harness) runs this in place of `zeroclaw acp`, with the same
arguments. It starts `zeroclaw` with them and passes every line between the client and ZeroClaw
unchanged, except one: a prompt the plugin has context for, which gets that context in front of
it. ZeroClaw has no hook or plugin surface that sees a session (`tokeneconomizer/plugin/hosts.py`,
`ZeroClaw`, says what was checked), so the session's events are read off this pipe, and each
becomes one call of the shared dispatcher (`../scripts/te_bootstrap.py hook --host zeroclaw`,
JSON in, JSON out), in a process of its own. All of the mechanism logic lives in the dispatcher;
this file only translates:

    session/new answered            -> SessionStart       its context waits for the first prompt
    session/load, session/resume    -> SessionStart       (source `resume`)
    session/prompt                  -> UserPromptSubmit   the start context, and the prompt's
                                                          own, go in front of the prompt
    tool_call / tool_call_update    -> PostToolUse        the guard's input; ACP has no way into
                                                          a running turn, so no redirect
    session/prompt answered         -> Stop               with its `stopReason`
    session/close, session/stop     -> SessionEnd         a detached fold, which reads the
    answered, or the pipe closing                         session from ZeroClaw's store

Hook calls run one at a time, in order, on a thread of their own; only the two whose answer goes
into the conversation (a session's start, a prompt) hold the line they belong to until they
answer, or for `HOOK_TIMEOUT_S` at most. Notices go to stderr, which the client logs. Every
failure means the line passes unchanged. `TE_DISABLE=1` (or `TE_SIDECALL=1`) makes this a plain
pipe. `TE_ZEROCLAW_BIN` names the `zeroclaw` to run (default: the one on PATH, else
`~/.local/bin/zeroclaw`); `TE_PYTHON` the Python the dispatcher runs in.
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.realpath(__file__))
BOOTSTRAP = os.path.join(HERE, "..", "scripts", "te_bootstrap.py")
HOOK_TIMEOUT_S = 20
END_WAIT_S = 60
HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _truthy(v):
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def disabled(env=None):
    env = os.environ if env is None else env
    return _truthy(env.get("TE_SIDECALL")) or _truthy(env.get("TE_DISABLE"))


def zeroclaw_bin(env=None):
    env = os.environ if env is None else env
    if env.get("TE_ZEROCLAW_BIN"):
        return env["TE_ZEROCLAW_BIN"]
    found = shutil.which("zeroclaw")
    if found:
        return found
    exe = os.path.join(os.path.expanduser("~"), ".local", "bin", "zeroclaw")
    return exe if os.path.isfile(exe) else "zeroclaw"


def config_dir_arg(argv):
    """The `--config-dir` ZeroClaw is given, as an absolute path; None without one."""
    for i, a in enumerate(argv):
        if a == "--config-dir" and i + 1 < len(argv):
            return os.path.abspath(os.path.expanduser(argv[i + 1]))
        if a.startswith("--config-dir="):
            return os.path.abspath(os.path.expanduser(a.split("=", 1)[1]))
    return None


def state_root(env=None):
    env = os.environ if env is None else env
    return os.path.abspath(env.get("TOKENECONOMIZER_HOME")
                           or os.path.join(os.path.expanduser("~"), ".tokeneconomizer"))


def dispatch(payload, cwd, env):
    """One dispatcher call; {} (no change) on any failure. It runs in the session's directory."""
    run_env = dict(env, PYTHONUTF8="1")
    try:
        out = subprocess.run([env.get("TE_PYTHON") or sys.executable, BOOTSTRAP, "hook",
                              "--host", "zeroclaw"],
                             input=json.dumps(payload), capture_output=True, encoding="utf-8",
                             errors="replace", timeout=HOOK_TIMEOUT_S,
                             cwd=cwd if cwd and os.path.isdir(cwd) else None, env=run_env,
                             **HIDDEN)
        answer = json.loads((out.stdout or "").strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}
    return answer if isinstance(answer, dict) else {}


def _context(out):
    return ((out or {}).get("hookSpecificOutput") or {}).get("additionalContext") or ""


def _prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n\n".join(p.get("text") for p in prompt
                           if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _key(mid):
    return json.dumps(mid, sort_keys=True)


def _parse(line):
    try:
        msg = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
    except (ValueError, UnicodeDecodeError):
        return None
    return msg if isinstance(msg, dict) else None


def _encode(msg):
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


class Relay:
    """What the plugin keeps about the ACP sessions on one pipe, keyed by ACP `sessionId`."""

    def __init__(self, config_dir=None, call=dispatch, env=None, note=None):
        self.env = dict(os.environ if env is None else env)
        if config_dir:
            self.env["ZEROCLAW_CONFIG_DIR"] = config_dir
        self.call = call
        self.note = note or (lambda text: (sys.stderr.write(text + "\n"), sys.stderr.flush()))
        self.sessions = {}
        self.pending = {}                  # request id -> what its answer means
        self.guard = threading.Lock()
        self.jobs = queue.Queue()
        self.worker = threading.Thread(target=self._work, name="te-zeroclaw-hooks", daemon=True)
        self.worker.start()

    # -- the hook thread -------------------------------------------------------------------
    def _work(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            fn, done, box = job
            try:
                box.append(fn())
            except Exception:
                box.append({})
            finally:
                done.set()

    def _send(self, rec, event, wait=False, **fields):
        payload = dict(fields, session_id=rec["id"], cwd=rec["cwd"], hook_event_name=event,
                       transcript_path=os.path.join(state_root(self.env), "zeroclaw-sessions",
                                                    f"{rec['id']}.json"))

        def run():
            out = self.call(payload, rec["cwd"], self.env) or {}
            notice = out.get("systemMessage")
            if isinstance(notice, str) and notice:
                self.note(notice)
            return out
        done, box = threading.Event(), []
        self.jobs.put((run, done, box))
        if not wait:
            return {}
        done.wait(HOOK_TIMEOUT_S + 5)
        return box[0] if box else {}

    def _record(self, sid, cwd):
        rec = {"id": sid, "cwd": cwd or os.getcwd(), "n": 0, "prompt_id": "", "in_turn": False,
               "calls": {}, "start_context": ""}
        self.sessions[sid] = rec
        return rec

    # -- client -> ZeroClaw ----------------------------------------------------------------
    def from_client(self, line):
        """The line to pass on to ZeroClaw for one line from the client."""
        msg = _parse(line)
        if msg is None or "method" not in msg or "id" not in msg:
            return line
        method, mid = msg.get("method"), msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        sid = params.get("sessionId") or params.get("session_id")
        cwd = params.get("cwd") or params.get("workspaceDir") or params.get("workspace_dir")
        with self.guard:
            if method == "session/new":
                self.pending[_key(mid)] = ("new", None, cwd)
            elif method in ("session/load", "session/resume") and isinstance(sid, str):
                self.pending[_key(mid)] = ("load", sid, cwd)
            elif method in ("session/close", "session/stop") and sid in self.sessions:
                self.pending[_key(mid)] = ("end", sid, None)
            elif method == "session/prompt" and sid in self.sessions:
                rec = self.sessions[sid]
            else:
                return line
            if method != "session/prompt":
                return line
            rec["n"] += 1
            rec["prompt_id"] = f"{sid}:{rec['n']}"
            rec["in_turn"], rec["calls"] = True, {}
            self.pending[_key(mid)] = ("prompt", sid, None)
            start, rec["start_context"] = rec["start_context"], ""
        prompt = params.get("prompt")
        out = self._send(rec, "UserPromptSubmit", wait=True, prompt_id=rec["prompt_id"],
                         prompt=_prompt_text(prompt))
        extra = "\n\n".join(x for x in (start, _context(out)) if x)
        if not extra:
            return line
        parts = (prompt if isinstance(prompt, list) else
                 [{"type": "text", "text": prompt}] if isinstance(prompt, str) else [])
        return _encode(dict(msg, params=dict(params, prompt=[{"type": "text", "text": extra}]
                                             + parts)))

    # -- ZeroClaw -> client ----------------------------------------------------------------
    def from_agent(self, line):
        """The line to pass on to the client for one line from ZeroClaw (always the same line)."""
        msg = _parse(line)
        if msg is None:
            return line
        if "id" in msg and "method" not in msg:
            self._answer(msg)
        elif msg.get("method") == "session/update":
            self._update(msg.get("params") if isinstance(msg.get("params"), dict) else {})
        return line

    def _answer(self, msg):
        with self.guard:
            what = self.pending.pop(_key(msg.get("id")), None)
        if what is None:
            return
        kind, sid, cwd = what
        result = msg.get("result") if isinstance(msg.get("result"), dict) else None
        if kind == "new":
            sid = (result or {}).get("sessionId")
            if not isinstance(sid, str) or not sid:
                return
            with self.guard:
                rec = self._record(sid, cwd)
            out = self._send(rec, "SessionStart", wait=True, source="startup")
            rec["start_context"] = _context(out)
        elif kind == "load":
            if result is None and "error" in msg:
                return
            with self.guard:
                rec = self.sessions.get(sid) or self._record(sid, cwd)
            out = self._send(rec, "SessionStart", wait=True, source="resume")
            rec["start_context"] = _context(out)
        elif kind == "prompt":
            rec = self.sessions.get(sid)
            if rec is None:
                return
            rec["in_turn"] = False
            reason = (result or {}).get("stopReason") or ("error" if "error" in msg else "")
            self._send(rec, "Stop", prompt_id=rec["prompt_id"], stop_reason=str(reason),
                       stop_hook_active=False)
        elif kind == "end":
            with self.guard:
                rec = self.sessions.pop(sid, None)
            if rec is not None:
                self._send(rec, "SessionEnd", reason="close")

    def _update(self, params):
        update = params.get("update") if isinstance(params.get("update"), dict) else {}
        rec = self.sessions.get(params.get("sessionId"))
        # Outside a prompt, updates are a loaded session's history being replayed.
        if rec is None or not rec["in_turn"]:
            return
        kind, cid = update.get("sessionUpdate"), str(update.get("toolCallId") or "")
        if kind == "tool_call":
            rec["calls"][cid] = {"name": update.get("name") or update.get("title") or "",
                                 "args": update.get("rawInput")}
        elif kind == "tool_call_update" and update.get("status") in ("completed", "failed"):
            call = rec["calls"].pop(cid, {})
            self._send(rec, "PostToolUse", prompt_id=rec["prompt_id"],
                       tool_name=update.get("name") or call.get("name") or "",
                       tool_use_id=cid, tool_input=call.get("args") if isinstance(
                           call.get("args"), dict) else {},
                       tool_response={"content": update.get("rawOutput"),
                                      "status": update.get("status")})

    def close(self, reason="exit"):
        """End every session still open (the pipe closed under them), and wait for the hooks."""
        with self.guard:
            recs, self.sessions = list(self.sessions.values()), {}
        for rec in recs:
            self._send(rec, "SessionEnd", reason=reason)
        self.jobs.put(None)
        self.worker.join(END_WAIT_S)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    argv = argv or ["acp"]
    try:
        child = subprocess.Popen([zeroclaw_bin(), *argv], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, **HIDDEN)
    except OSError as e:
        sys.stderr.write(f"TokenEconomizer: could not start zeroclaw: {e}\n")
        return 127
    relay = None
    if not disabled():
        try:
            relay = Relay(config_dir_arg(argv))
        except Exception:
            relay = None
    out = sys.stdout.buffer

    def safe(fn, line):
        try:
            return fn(line)
        except Exception:
            return line

    def pump():
        for line in iter(child.stdout.readline, b""):
            out.write(safe(relay.from_agent, line) if relay else line)
            out.flush()
    reader = threading.Thread(target=pump, name="te-zeroclaw-out", daemon=True)
    reader.start()
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for line in iter(sys.stdin.buffer.readline, b""):
            try:
                child.stdin.write(safe(relay.from_client, line) if relay else line)
                child.stdin.flush()
            except (BrokenPipeError, OSError):
                break
    finally:
        try:
            child.stdin.close()
        except OSError:
            pass
        code = child.wait()
        reader.join(5)
        if relay is not None:
            relay.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
