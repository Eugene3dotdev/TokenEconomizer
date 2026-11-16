"""ZeroClaw: its ACP session store as a history source, the host, and the ACP relay.

ZeroClaw keeps the sessions `zeroclaw acp` runs in `<config dir>/data/sessions/acp-sessions.db`.
The store here is synthetic, built with the schema ZeroClaw 0.9.0 creates (copied from a store a
stub-model smoke run wrote) and rows in the shapes `acp_session_store.rs` writes. The relay is run
against a stand-in dispatcher, then against the real one with an inline fold, and once as a
process in front of a stand-in `zeroclaw` that speaks the ACP lines the real one does. None of
that runs ZeroClaw or a model. The tests at the end drive a real `zeroclaw acp`
(`TE_ZEROCLAW_E2E=1`, `TE_ZEROCLAW_BIN`): a config directory of their own, a local stand-in for
the model.
"""
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tokeneconomizer import trace
from tokeneconomizer.plugin import hosts, sidecall
from tokeneconomizer.trace import sources

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELAY_FILE = os.path.join(REPO, "plugin", "zeroclaw", "te_zeroclaw_acp.py")

_spec = importlib.util.spec_from_file_location("te_zeroclaw_acp", RELAY_FILE)
relay_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay_mod)

SCHEMA = """
CREATE TABLE acp_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_uuid TEXT NOT NULL UNIQUE,
    agent_alias TEXT NOT NULL, workspace_dir TEXT NOT NULL, interaction_surface TEXT,
    token_count INTEGER NOT NULL DEFAULT 0, killed_at TEXT, retained_context_json TEXT,
    retained_context_frontier INTEGER, created_at TEXT NOT NULL, last_activity TEXT NOT NULL,
    plan_json TEXT, projected_message_count INTEGER NOT NULL DEFAULT 0,
    projected_count_through INTEGER, trim_breadcrumb INTEGER, principal_id TEXT);
CREATE TABLE acp_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES acp_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, reasoning_content TEXT, created_at TEXT NOT NULL);
CREATE TABLE acp_tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES acp_messages(id) ON DELETE CASCADE,
    tool_call_id TEXT NOT NULL, tool_name TEXT NOT NULL, event_kind TEXT NOT NULL,
    payload TEXT NOT NULL, outcome TEXT, created_at TEXT NOT NULL);
CREATE TABLE acp_session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES acp_sessions(id) ON DELETE CASCADE,
    action TEXT NOT NULL, outcome TEXT NOT NULL, payload TEXT, created_at TEXT NOT NULL);
"""

# The calls of a session that meets the missing `pytest` and works around it, as ZeroClaw
# stores them: a failed shell call's result starts with `Error: ` (tool_execution.rs).
TRAP = [("shell", {"command": "pytest -q tests/test_a.py"},
         "Error: bash: pytest: command not found"),
        ("shell", {"command": "python -m pytest -q tests/test_a.py"}, "1 passed in 0.1s")]
SECRET_PROMPT = "fix the tests; the deploy key is hunter2"
SECRET_REPLY = "I will look at the failing test first."


def rfc3339(t):
    """A time as ZeroClaw writes it: UTC, nine digits of fraction."""
    whole = int(t)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(whole)) + \
        f".{int((t - whole) * 1e9):09d}+00:00"


def db_path(config_dir):
    return os.path.join(str(config_dir), "data", "sessions", "acp-sessions.db")


def zeroclaw_store(config_dir, sessions):
    """An ACP store under `config_dir`; `sessions` is [(uuid, cwd, age_s)] or (uuid, cwd, age_s,
    calls). Each session is one turn: a prompt (with ZeroClaw's date line), one assistant message
    per call with its `in` and `out` rows, and a closing reply."""
    db = db_path(config_dir)
    os.makedirs(os.path.dirname(db), exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    for s in sessions:
        sid, cwd, age_s = s[:3]
        calls = s[3] if len(s) > 3 else TRAP
        end = time.time() - age_s
        at = rfc3339(end)
        cur = con.execute(
            "INSERT INTO acp_sessions (session_uuid, agent_alias, workspace_dir, created_at, "
            "last_activity, projected_message_count) VALUES (?,?,?,?,?,?)",
            (sid, "default", cwd, rfc3339(end - 300), at, 2 + len(calls)))
        row = cur.lastrowid
        con.execute("INSERT INTO acp_messages (session_id, role, content, created_at) "
                    "VALUES (?,?,?,?)", (row, "user", "[CURRENT DATE & TIME: 2026-11-16 14:38:08 "
                                         f"+01:00]\n\n{SECRET_PROMPT}", at))
        for n, (name, args, out) in enumerate(calls):
            msg = con.execute("INSERT INTO acp_messages (session_id, role, content, created_at) "
                              "VALUES (?,?,?,?)", (row, "assistant",
                                                   SECRET_REPLY if n == 0 else "", at)).lastrowid
            con.execute("INSERT INTO acp_tool_calls (message_id, tool_call_id, tool_name, "
                        "event_kind, payload, outcome, created_at) VALUES (?,?,?,?,?,NULL,?)",
                        (msg, f"call_{n}", name, "in", json.dumps(args), at))
            con.execute("INSERT INTO acp_tool_calls (message_id, tool_call_id, tool_name, "
                        "event_kind, payload, outcome, created_at) VALUES (?,?,?,?,?,?,?)",
                        (msg, f"call_{n}", name, "out", out, "unknown", at))
        con.execute("INSERT INTO acp_messages (session_id, role, content, created_at) "
                    "VALUES (?,?,?,?)", (row, "assistant", "Done: the tests pass.", at))
    con.commit()
    con.close()
    return db


def _repo(path):
    os.makedirs(path)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return str(path)


# ---- the store, as a history source --------------------------------------------------------
def test_the_config_dir_is_resolved_as_zeroclaw_resolves_it(tmp_path):
    home = str(tmp_path / "h")
    assert trace.zeroclaw_config_dir({}) is None
    assert trace.zeroclaw_config_dir({"HOME": home}) == os.path.join(home, ".zeroclaw")
    assert trace.zeroclaw_config_dir({"HOME": home, "ZEROCLAW_CONFIG_DIR": "/c",
                                      "ZEROCLAW_DATA_DIR": "/d"}) == "/c"
    install = tmp_path / "install"
    install.mkdir()
    (install / "config.toml").write_text("")
    assert trace.zeroclaw_config_dir({"ZEROCLAW_DATA_DIR": str(install)}) == str(install)
    # A data directory beside a `.zeroclaw` install (the container images' layout).
    assert trace.zeroclaw_config_dir({"ZEROCLAW_DATA_DIR": str(tmp_path / "x" / "data")}) == str(
        tmp_path / "x" / ".zeroclaw")


def test_zeroclaws_nanosecond_times_are_read():
    import calendar
    from tokeneconomizer.trace.zeroclaw import zeroclaw_epoch
    whole = calendar.timegm((2026, 11, 16, 13, 38, 8, 0, 0, 0))
    assert zeroclaw_epoch("2026-11-16T13:38:08.477058310+00:00") == pytest.approx(
        whole + 0.477058, abs=1e-6)
    assert zeroclaw_epoch("2026-11-16T14:38:08+01:00") == whole
    assert zeroclaw_epoch("not a time") is None and zeroclaw_epoch(None) is None


def test_the_source_reads_the_default_install_and_every_named_one_beside_it(tmp_path):
    home = tmp_path / "home"
    zeroclaw_store(home / ".zeroclaw", [("a-1", "/r", 3600)])
    zeroclaw_store(home / ".zeroclaw-cliproxyapi", [("b-1", "/r", 1800)])
    src = sources.SOURCES["zeroclaw"]
    assert [x["id"] for x in src.logs({"HOME": str(home)})] == ["a-1", "b-1"]
    # An install the environment names is the only one read, as ZeroClaw reads only it.
    assert [x["id"] for x in src.logs({"HOME": str(home), "ZEROCLAW_CONFIG_DIR": str(
        home / ".zeroclaw-cliproxyapi")})] == ["b-1"]
    assert src.logs({"HOME": str(home), "ZEROCLAW_CONFIG_DIR": str(tmp_path / "none")}) == []


def test_a_session_in_its_agents_own_workspace_belongs_to_no_repository(tmp_path):
    cfg = tmp_path / ".zeroclaw"
    ws = str(cfg / "agents" / "default" / "workspace")
    zeroclaw_store(cfg, [("ws-1", ws, 3600), ("r-1", "/src/widget", 3600)])
    src = sources.SOURCES["zeroclaw"]
    heads = {x["id"]: src.head(x["path"]) for x in src.logs({"ZEROCLAW_CONFIG_DIR": str(cfg)})}
    assert heads["ws-1"]["cwd"] is None
    assert heads["r-1"]["cwd"] == "/src/widget" and not heads["r-1"]["headless"]
    assert heads["r-1"]["started"] < time.time() - 3600


def test_the_export_keeps_the_tools_and_drops_the_conversations_words(tmp_path):
    cfg = tmp_path / ".zeroclaw"
    db = zeroclaw_store(cfg, [("s-1", "/src/widget", 3600)])
    out = str(tmp_path / "te" / "zeroclaw-sessions" / "s-1.json")
    assert trace.zeroclaw_export(db, "s-1", out)
    text = open(out, encoding="utf-8").read()
    assert SECRET_PROMPT not in text and "hunter2" not in text and SECRET_REPLY not in text
    assert "Done: the tests pass." not in text
    assert "python -m pytest -q tests/test_a.py" in text and "command not found" in text
    assert trace.zeroclaw_prompt_count(out) == 1
    assert trace.zeroclaw_session_info(out)["workspace_dir"] == "/src/widget"
    assert oct(os.stat(out).st_mode & 0o777) == "0o600"
    assert not trace.zeroclaw_export(db, "no-such-session", str(tmp_path / "x.json"))
    assert not trace.zeroclaw_export(str(tmp_path / "missing.db"), "s-1", str(tmp_path / "y.json"))


def test_a_trim_note_and_an_interrupted_turns_marker_are_not_prompts(tmp_path):
    cfg = tmp_path / ".zeroclaw"
    db = zeroclaw_store(cfg, [("s-1", "/src/widget", 3600)])
    con = sqlite3.connect(db)
    at = rfc3339(time.time())
    for role, content in (("user", "[earlier turns omitted to fit the context window]"),
                          ("__zeroclaw_turn_stream_interrupted__", "x"),
                          ("user", "[CURRENT DATE & TIME: 2026-11-16]\n\na second prompt")):
        con.execute("INSERT INTO acp_messages (session_id, role, content, created_at) "
                    "VALUES (1,?,?,?)", (role, content, at))
    con.commit()
    con.close()
    out = str(tmp_path / "s.json")
    assert trace.zeroclaw_export(db, "s-1", out)
    assert trace.zeroclaw_prompt_count(out) == 2
    assert sources.SOURCES["zeroclaw"].prompt_count(f"{db}#s-1") == 2


def test_the_trace_marks_failures_by_the_prefix_zeroclaw_gives_them(tmp_path):
    calls = [("shell", {"command": "make"}, "Error: make: *** No rule to make target"),
             ("file_read", {"path": "a.txt"}, "Error: Failed to resolve file path"),
             ("file_edit", {"path": "a.py", "old_string": "x", "new_string": "y"}, "ok"),
             ("content_search", {"pattern": "def main", "path": "src"}, "src/a.py:1"),
             ("web_fetch", {"url": "https://example.com"}, "Denied by user.")]
    db = zeroclaw_store(tmp_path / ".zeroclaw", [("s-1", "/src/widget", 60, calls)])
    out, tr = str(tmp_path / "s.json"), str(tmp_path / "s.jsonl")
    assert trace.zeroclaw_export(db, "s-1", out)
    assert trace.zeroclaw_session_to_claude(out, tr) == 5
    got = trace.load_trajectory(tr)
    assert [(n, ok) for n, _, _, ok in got] == [("Bash", False), ("Read", False), ("Edit", True),
                                                ("Grep", True), ("web_fetch", False)]
    assert got[1][1] == {"file_path": "a.txt"} and got[3][1]["pattern"] == "def main"
    usage = trace.zeroclaw_session_usage(out)
    assert usage["input_tokens"] is None and usage["model"] is None and usage["requests"] == 6


def test_a_trace_of_nothing_is_none(tmp_path):
    assert trace.zeroclaw_session_to_claude(str(tmp_path / "none.json"), str(tmp_path / "t")) is None
    (tmp_path / "empty.json").write_text('{"info": {}, "messages": []}')
    assert trace.zeroclaw_session_to_claude(str(tmp_path / "empty.json"), str(tmp_path / "t")) is None


# ---- the host --------------------------------------------------------------------------------
def test_the_host_is_named_and_makes_no_model_calls():
    h = hosts.detect({}, hint="zeroclaw")
    assert isinstance(h, hosts.ZeroClaw) and h.name == "zeroclaw"
    assert hosts.HOSTS["zeroclaw"] is hosts.ZeroClaw
    assert "zeroclaw" in sidecall.NO_SIDE_CALLS
    assert sidecall.runner_for("zeroclaw", object()) is None
    assert h.prompt_context and h.keeps_start_context
    # Only a session's start and a prompt can carry context into the conversation.
    assert h.context("SessionStart", "x")["hookSpecificOutput"]["additionalContext"] == "x"
    assert h.context("UserPromptSubmit", "x")
    assert h.context("PostToolUse", "x") == {}


def test_the_host_reads_a_tool_call_as_the_mechanisms_name_it():
    h = hosts.ZeroClaw()
    assert h.tool_call({"tool_name": "shell", "tool_input": {"command": "pytest"},
                        "tool_response": {"content": "Error: boom", "status": "completed"}}) == (
        "Bash", {"command": "pytest"}, "Error: boom", 1)
    assert h.tool_call({"tool_name": "shell", "tool_input": {"command": "ls"},
                        "tool_response": {"content": "a", "status": "completed"}})[3] == 0
    assert h.tool_call({"tool_name": "file_read", "tool_input": {"path": "a"},
                        "tool_response": {"content": "1: x"}})[:2] == ("Read", {"file_path": "a"})
    assert h.tool_call({"tool_name": "deliver_file", "tool_input": {},
                        "tool_response": {"content": "", "status": "failed"}})[3] is None
    assert h.stopped_normally({"stop_reason": "end_turn"})
    assert not h.stopped_normally({"stop_reason": "cancelled"})
    assert h.turn_key({"prompt_id": "s:2"}) == "s:2"


def test_the_fold_reads_the_session_from_the_store_it_ran_against(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    zeroclaw_store(cfg, [("s-1", "/src/widget", 60)])
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path / "te"))
    monkeypatch.setenv("ZEROCLAW_CONFIG_DIR", str(tmp_path / "another-install"))
    h = hosts.ZeroClaw()
    path = str(tmp_path / "te" / "zeroclaw-sessions" / "s-1.json")
    assert h.session_state({"ZEROCLAW_CONFIG_DIR": str(cfg)}) == {"zeroclaw_config_dir": str(cfg)}
    with pytest.raises(hosts.FoldLater):
        h.complete(None, "s-1", session={"updated": time.time()})
    assert h.complete(None, "s-1", session={"zeroclaw_config_dir": str(cfg)}) == path
    assert h.prompt_count(path) == 1
    assert h.complete("/x/y.json", "../s-1") == "/x/y.json"
    assert h.complete(None, "gone", session={"updated": time.time() - h.FOLD_RETRY_S - 1}) is None


# ---- the relay -------------------------------------------------------------------------------
class FakeDispatcher:
    """Stands in for `relay_mod.dispatch`; records calls, and answers from `answers[event]`."""

    def __init__(self):
        self.calls, self.answers, self.lock = [], {}, threading.Lock()

    def __call__(self, payload, cwd, env):
        with self.lock:
            self.calls.append(dict(payload, _cwd=cwd, _config=env.get("ZEROCLAW_CONFIG_DIR")))
        return self.answers.get(payload["hook_event_name"], {})

    def events(self):
        return [c["hook_event_name"] for c in self.calls]


def line(msg):
    return (json.dumps(msg) + "\n").encode("utf-8")


def req(mid, method, **params):
    return line({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})


def res(mid, **result):
    return line({"jsonrpc": "2.0", "id": mid, "result": result})


def update(sid, **u):
    return line({"jsonrpc": "2.0", "method": "session/update",
                 "params": {"sessionId": sid, "update": u}})


def _drain(relay):
    relay.close()


def test_the_relay_turns_one_acp_session_into_the_hook_events(tmp_path):
    fake = FakeDispatcher()
    fake.answers["SessionStart"] = {"hookSpecificOutput": {"additionalContext": "FACTS"},
                                    "systemMessage": "TokenEconomizer: added 1 learned fact"}
    notes = []
    r = relay_mod.Relay(config_dir="/cfg", call=fake, env={"TOKENECONOMIZER_HOME": "/te"},
                        note=notes.append)
    cwd = str(tmp_path)
    assert r.from_client(req(1, "initialize", protocolVersion=1)) == req(1, "initialize",
                                                                         protocolVersion=1)
    assert r.from_client(req(2, "session/new", cwd=cwd, mcpServers=[])) == req(
        2, "session/new", cwd=cwd, mcpServers=[])
    assert r.from_agent(res(2, sessionId="S")) == res(2, sessionId="S")
    sent = json.loads(r.from_client(req(3, "session/prompt", sessionId="S",
                                        prompt=[{"type": "text", "text": "fix it"}])))
    assert sent["params"]["prompt"] == [{"type": "text", "text": "FACTS"},
                                        {"type": "text", "text": "fix it"}]
    assert sent["id"] == 3 and sent["method"] == "session/prompt"
    r.from_agent(update("S", sessionUpdate="tool_call", toolCallId="c1", name="shell",
                        title="pytest", rawInput={"command": "pytest"}, status="pending"))
    r.from_agent(update("S", sessionUpdate="tool_call_update", toolCallId="c1", name="shell",
                        status="completed", rawOutput="Error: pytest: command not found"))
    r.from_agent(update("S", sessionUpdate="agent_message_chunk",
                        content={"type": "text", "text": "hi"}))
    r.from_agent(res(3, sessionId="S", stopReason="end_turn", content="done"))
    # A second prompt gets no start context again.
    assert r.from_client(req(4, "session/prompt", sessionId="S", prompt="again")) == req(
        4, "session/prompt", sessionId="S", prompt="again")
    r.from_agent(res(4, sessionId="S", stopReason="cancelled", content=""))
    r.from_client(req(5, "session/close", sessionId="S"))
    r.from_agent(res(5))
    _drain(r)
    assert fake.events() == ["SessionStart", "UserPromptSubmit", "PostToolUse", "Stop",
                             "UserPromptSubmit", "Stop", "SessionEnd"]
    start, prompt, tool, stop, _, cancelled, end = fake.calls
    assert start["source"] == "startup" and start["cwd"] == cwd and start["_cwd"] == cwd
    assert start["_config"] == "/cfg"
    assert start["transcript_path"] == os.path.join("/te", "zeroclaw-sessions", "S.json")
    assert prompt["prompt"] == "fix it" and prompt["prompt_id"] == "S:1"
    assert tool["tool_name"] == "shell" and tool["tool_input"] == {"command": "pytest"}
    assert tool["tool_response"]["content"].startswith("Error: ") and tool["prompt_id"] == "S:1"
    assert stop["stop_reason"] == "end_turn" and cancelled["stop_reason"] == "cancelled"
    assert end["reason"] == "close"
    assert notes == ["TokenEconomizer: added 1 learned fact"]


def test_a_loaded_sessions_replayed_history_is_not_read_as_new_calls(tmp_path):
    fake = FakeDispatcher()
    r = relay_mod.Relay(call=fake, env={})
    r.from_client(req(1, "session/load", sessionId="S", cwd=str(tmp_path), mcpServers=[]))
    r.from_agent(update("S", sessionUpdate="tool_call", toolCallId="old", name="shell",
                        rawInput={"command": "ls"}, status="completed"))
    r.from_agent(update("S", sessionUpdate="tool_call_update", toolCallId="old",
                        status="completed", rawOutput="a"))
    r.from_agent(res(1))
    _drain(r)
    assert fake.events() == ["SessionStart", "SessionEnd"]
    assert fake.calls[0]["source"] == "resume" and fake.calls[1]["reason"] == "exit"


def test_the_relay_passes_what_it_does_not_follow_unchanged():
    fake = FakeDispatcher()
    r = relay_mod.Relay(call=fake, env={})
    for raw in (b"not json\n", b"[1, 2]\n", req(1, "session/prompt", sessionId="unknown",
                                                prompt="x"),
                line({"jsonrpc": "2.0", "id": 9, "method": "session/request_permission",
                      "params": {"sessionId": "unknown", "options": []}}),
                line({"jsonrpc": "2.0", "id": 77, "error": {"code": -1, "message": "no"}})):
        assert r.from_client(raw) == raw and r.from_agent(raw) == raw
    r.from_client(req(2, "session/new", cwd="/x"))
    r.from_agent(line({"jsonrpc": "2.0", "id": 2, "error": {"code": -32000, "message": "no"}}))
    _drain(r)
    assert fake.events() == []


def test_a_hook_that_fails_leaves_the_prompt_as_it_was(tmp_path):
    def broken(payload, cwd, env):
        raise RuntimeError("boom")
    r = relay_mod.Relay(call=broken, env={})
    r.from_client(req(1, "session/new", cwd=str(tmp_path)))
    r.from_agent(res(1, sessionId="S"))
    p = req(2, "session/prompt", sessionId="S", prompt=[{"type": "text", "text": "hi"}])
    assert r.from_client(p) == p
    _drain(r)


def test_the_relay_finds_the_config_dir_it_was_given():
    assert relay_mod.config_dir_arg(["acp", "--config-dir", "/c"]) == "/c"
    assert relay_mod.config_dir_arg(["--config-dir=/d", "acp"]) == "/d"
    assert relay_mod.config_dir_arg(["acp"]) is None


def test_the_whole_path_folds_a_session_through_the_real_dispatcher(tmp_path, monkeypatch):
    """The relay, a process per event, the session keyed by the directory it was opened in, and
    the fold reading the session from ZeroClaw's store once it ends."""
    root = tmp_path / "te"
    for var in ("TE_DISABLE", "TE_SIDECALL", "ZEROCLAW_CONFIG_DIR", "ZEROCLAW_DATA_DIR"):
        monkeypatch.delenv(var, raising=False)
    env = dict(os.environ, TOKENECONOMIZER_HOME=str(root), TE_FOLD_SYNC="1",
               TE_CORE_PATH=REPO, HOME=str(tmp_path / "home"))
    repo = _repo(tmp_path / "widget")
    cfg = tmp_path / "cfg"
    r = relay_mod.Relay(config_dir=str(cfg), env=env)
    r.from_client(req(1, "session/new", cwd=repo, mcpServers=[]))
    r.from_agent(res(1, sessionId="zc-1"))
    r.from_client(req(2, "session/prompt", sessionId="zc-1", prompt="fix the tests"))
    for n, (name, args, out) in enumerate(TRAP):
        r.from_agent(update("zc-1", sessionUpdate="tool_call", toolCallId=f"c{n}", name=name,
                            rawInput=args, status="pending"))
        r.from_agent(update("zc-1", sessionUpdate="tool_call_update", toolCallId=f"c{n}",
                            name=name, status="completed", rawOutput=out))
    zeroclaw_store(cfg, [("zc-1", repo, 0)])        # what ZeroClaw persists at the turn's end
    r.from_agent(res(2, sessionId="zc-1", stopReason="end_turn", content="done"))
    r.from_client(req(3, "session/close", sessionId="zc-1"))
    r.from_agent(res(3))
    r.close()
    errors = root / "errors.log"
    assert not errors.exists() or not errors.read_text("utf-8").strip(), errors.read_text("utf-8")
    ledgers = list(root.rglob("ledger.jsonl"))
    assert len(ledgers) == 1
    records = [json.loads(ln) for ln in ledgers[0].read_text("utf-8").splitlines()]
    (rec,) = [x for x in records if x.get("kind") == "session"]
    assert rec["session_id"] == "zc-1" and rec["prompts"] == 1 and not rec.get("error")
    state = json.loads((ledgers[0].parent / "sessions" / "zc-1.json").read_text("utf-8"))
    assert state["host"] == "zeroclaw" and state["zeroclaw_config_dir"] == str(cfg)
    export = root / "zeroclaw-sessions" / "zc-1.json"
    assert export.exists() and "hunter2" not in export.read_text("utf-8")


FAKE_ZEROCLAW = textwrap.dedent("""\
    import json, os, sys
    log = open(os.environ["FAKE_LOG"], "a")
    log.write(json.dumps(sys.argv[1:]) + "\\n")
    def send(m):
        sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
    for raw in sys.stdin:
        m = json.loads(raw)
        log.write(raw); log.flush()
        if m.get("method") == "session/new":
            send({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": "P"}})
        elif m.get("method") == "session/prompt":
            send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "P",
                  "update": {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text", "text": "ok"}}}})
            send({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": "P",
                  "stopReason": "end_turn", "content": "ok"}})
        elif "id" in m:
            send({"jsonrpc": "2.0", "id": m["id"], "result": {}})
""")


def _relay_process(tmp_path, extra_env, lines):
    fake = tmp_path / "fake_zeroclaw.py"
    fake.write_text(FAKE_ZEROCLAW)
    exe = tmp_path / "zeroclaw"
    exe.write_text(f"#!{sys.executable}\nexec(open({str(fake)!r}).read())\n")
    exe.chmod(0o755)
    env = dict(os.environ, TE_ZEROCLAW_BIN=str(exe), FAKE_LOG=str(tmp_path / "fake.log"),
               TOKENECONOMIZER_HOME=str(tmp_path / "te"), TE_CORE_PATH=REPO,
               HOME=str(tmp_path / "home"), **extra_env)
    for var in ("TE_DISABLE", "TE_SIDECALL"):
        if var not in extra_env:
            env.pop(var, None)
    out = subprocess.run([sys.executable, RELAY_FILE, "acp", "--config-dir", str(tmp_path / "c")],
                         input=b"".join(lines), capture_output=True, env=env, timeout=120)
    seen = (tmp_path / "fake.log").read_text("utf-8").splitlines()
    return out, json.loads(seen[0]), [json.loads(x) for x in seen[1:]]


@pytest.mark.skipif(os.name == "nt", reason="the stand-in zeroclaw is a POSIX script")
def test_the_relay_process_sits_between_a_client_and_zeroclaw(tmp_path):
    repo = _repo(tmp_path / "widget")
    lines = [req(1, "initialize", protocolVersion=1), req(2, "session/new", cwd=repo),
             req(3, "session/prompt", sessionId="P", prompt=[{"type": "text", "text": "hi"}]),
             req(4, "session/close", sessionId="P")]
    out, argv, got = _relay_process(tmp_path, {"TE_FOLD_SYNC": "1"}, lines)
    assert out.returncode == 0, out.stderr
    assert argv == ["acp", "--config-dir", str(tmp_path / "c")]
    assert [m.get("method") for m in got] == ["initialize", "session/new", "session/prompt",
                                              "session/close"]
    answers = [json.loads(x) for x in out.stdout.decode().splitlines()]
    assert [a.get("id") for a in answers] == [1, 2, None, 3, 4]
    state = tmp_path / "te" / "repos"
    assert list(state.rglob("sessions/P.json")), out.stderr
    # ZeroClaw's store has no row (the stand-in keeps none): the fold waits for one.
    assert not list(state.rglob("ledger.jsonl"))


@pytest.mark.skipif(os.name == "nt", reason="the stand-in zeroclaw is a POSIX script")
def test_with_the_plugin_off_the_relay_is_a_plain_pipe(tmp_path):
    lines = [req(2, "session/new", cwd=str(tmp_path)),
             req(3, "session/prompt", sessionId="P", prompt="hi")]
    out, _, got = _relay_process(tmp_path, {"TE_DISABLE": "1"}, lines)
    assert out.returncode == 0
    assert got[1] == json.loads(lines[1]) and not (tmp_path / "te").exists()


# ---- a real ZeroClaw -------------------------------------------------------------------------
ZEROCLAW = os.environ.get("TE_ZEROCLAW_BIN")
e2e = pytest.mark.skipif(os.environ.get("TE_ZEROCLAW_E2E") != "1" or not ZEROCLAW,
                         reason="drives a real ZeroClaw: set TE_ZEROCLAW_E2E=1 and TE_ZEROCLAW_BIN")


class _Chat(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint. A prompt holding `FAIL n` gets n `shell` calls of a command
    that is not installed; anything else gets text. Every request is kept."""
    requests = []

    def log_message(self, *args):
        pass

    def _send(self, kind, body):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send("application/json", json.dumps({"object": "list", "data": [
            {"id": "te-model", "object": "model"}]}).encode())

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        _Chat.requests.append(req)
        msgs = req.get("messages") or []
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=0)
        prompt = json.dumps(msgs[last].get("content")) if msgs else ""
        done = sum(1 for m in msgs[last:] if m.get("role") == "tool")
        want = re.search(r"FAIL (\d+)", prompt)
        offered = [t.get("function", {}).get("name") for t in req.get("tools") or []]
        call = bool("shell" in offered and want and done < int(want.group(1)))
        delta = {"role": "assistant"}
        if call:
            delta["tool_calls"] = [{"index": 0, "id": f"call_{time.time_ns()}", "type": "function",
                                    "function": {"name": "shell", "arguments": json.dumps(
                                        {"command": "te-not-installed-tool --version"})}}]
        else:
            delta["content"] = "done"
        usage = {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010}
        if not req.get("stream"):
            message = {"role": "assistant", "content": delta.get("content")}
            if call:
                message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"}
                                         for c in delta["tool_calls"]]
            self._send("application/json", json.dumps({
                "id": "c", "object": "chat.completion", "created": 0, "model": "te-model",
                "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop",
                             "message": message}], "usage": usage}).encode())
            return

        def chunk(choices, **extra):
            return "data: " + json.dumps(dict(id="c", object="chat.completion.chunk", created=0,
                                              model="te-model", choices=choices, **extra)) + "\n\n"
        self._send("text/event-stream", (
            chunk([{"index": 0, "delta": delta, "finish_reason": None}])
            + chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls" if call else "stop"}])
            + chunk([], usage=usage) + "data: [DONE]\n\n").encode())


CONFIG = """schema_version = 4
locale = "en"

[providers.models.custom.stub]
uri = "http://127.0.0.1:{port}/v1"
model = "te-model"
wire_api = "chat_completions"
native_tools = true

[memory]
backend = "none"
auto_save = false

[agents.default]
model_provider = "custom.stub"
risk_profile = "te"
runtime_profile = "te"

[agents.default.memory]
backend = "none"

[risk_profiles.te]
level = "full"
workspace_only = true
allowed_tools = ["shell", "file_read"]
allowed_commands = ["te-not-installed-tool"]

[runtime_profiles.te]
"""


class Client:
    """An ACP client on the relay's stdio: requests by id, permission requests allowed."""

    def __init__(self, env, config_dir, log):
        self.proc = subprocess.Popen([sys.executable, RELAY_FILE, "acp", "--config-dir",
                                      str(config_dir)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=open(log, "ab"), env=env)
        self.next, self.answers, self.updates = 1, {}, []
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _write(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if "method" in msg and "id" in msg:
                options = (msg.get("params") or {}).get("options") or []
                allow = next((o for o in options if str(o.get("kind", "")).startswith("allow")),
                             options[0] if options else None)
                self._write({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": {
                    "outcome": "selected", "optionId": allow["optionId"]}} if allow else {}})
                continue
            with self.cv:
                if "id" in msg:
                    self.answers[msg["id"]] = msg
                else:
                    self.updates.append(msg)
                self.cv.notify_all()

    def call(self, method, timeout=120, **params):
        mid, self.next = self.next, self.next + 1
        self._write({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        with self.cv:
            assert self.cv.wait_for(lambda: mid in self.answers, timeout), method
            return self.answers.pop(mid)

    def finish(self):
        self.proc.stdin.close()
        return self.proc.wait(60)


@pytest.fixture
def zc_world(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Chat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Chat.requests = []
    cfg = tmp_path / "zeroclaw-config"
    cfg.mkdir()
    (cfg / "config.toml").write_text(CONFIG.format(port=server.server_address[1]),
                                     encoding="utf-8")
    te, home = tmp_path / "te", tmp_path / "home"
    te.mkdir()
    home.mkdir()
    env = dict(os.environ, HOME=str(home), TOKENECONOMIZER_HOME=str(te), TE_CORE_PATH=REPO,
               TE_ZEROCLAW_BIN=ZEROCLAW, TE_MICROSKILLS="envfacts", TE_TOOLPRUNE="off",
               NO_COLOR="1")
    for var in ("TE_DISABLE", "TE_SIDECALL", "ZEROCLAW_CONFIG_DIR", "ZEROCLAW_DATA_DIR",
                "ZEROCLAW_WORKSPACE", "TE_FOLD_SYNC"):
        env.pop(var, None)
    yield type("World", (), {"env": env, "cfg": cfg, "te": te, "tmp": tmp_path,
                             "log": str(tmp_path / "relay.stderr")})
    server.shutdown()
    server.server_close()


def _wait(check, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        got = check()
        if got:
            return got
        time.sleep(0.5)
    return check()


def _sessions(te):
    records = []
    for p in te.rglob("ledger.jsonl"):
        records += [json.loads(ln) for ln in p.read_text("utf-8").splitlines()]
    return [r for r in records if r.get("kind") == "session"]


def _session(world, repo, prompt):
    c = Client(world.env, world.cfg, world.log)
    assert "result" in c.call("initialize", protocolVersion=1, clientCapabilities={})
    sid = c.call("session/new", cwd=repo, mcpServers=[])["result"]["sessionId"]
    answer = c.call("session/prompt", timeout=170, sessionId=sid,
                    prompt=[{"type": "text", "text": prompt}])
    closed = c.call("session/close", sessionId=sid)
    assert c.finish() == 0
    return sid, answer, closed, c.updates


@e2e
def test_a_real_session_is_folded_from_zeroclaws_store_and_keeps_none_of_its_words(zc_world):
    repo = _repo(zc_world.tmp / "repo")
    sid, answer, closed, updates = _session(zc_world, repo, "FAIL 2 SECRET-WORDS")
    assert answer["result"]["stopReason"] == "end_turn" and "result" in closed, answer
    calls = [u["params"]["update"] for u in updates
             if u["params"]["update"].get("sessionUpdate") == "tool_call_update"]
    assert len(calls) >= 2, updates[-3:]
    (done,) = _wait(lambda: _sessions(zc_world.te))
    assert done["session_id"] == sid and done["host"] == "zeroclaw"
    assert done["prompts"] == 1 and done["requests"] and not done.get("error")
    assert os.path.basename(done["repo"]) == "repo"
    (export,) = zc_world.te.rglob("zeroclaw-sessions/*.json")
    tr = str(zc_world.tmp / "trace.jsonl")
    assert trace.zeroclaw_session_to_claude(str(export), tr) >= 2
    shells = [c for c in trace.load_trajectory(tr) if c[0] == "Bash"]
    assert shells and not any(ok for *_, ok in shells)        # the failure is read as one
    assert all("not found" in out for _, _, out, _ in shells), shells
    held = [p for p in zc_world.te.rglob("*") if p.is_file() and b"SECRET-WORDS" in p.read_bytes()]
    assert not held


@e2e
def test_learned_context_reaches_the_model_in_front_of_a_real_prompt(zc_world):
    repo = _repo(zc_world.tmp / "repo")
    _session(zc_world, repo, "warm up")
    (state,) = _wait(lambda: [p.parent for p in zc_world.te.rglob("ledger.jsonl")])
    (state / "injection.json").write_text(json.dumps({
        "rendered": "<tokeneconomizer>\n- FACT-MARKER\n</tokeneconomizer>"}), encoding="utf-8")
    _Chat.requests = []
    _session(zc_world, repo, "second")
    users = [json.dumps([m for m in r.get("messages") or [] if m.get("role") == "user"])
             for r in _Chat.requests if r.get("tools")]
    assert users and "FACT-MARKER" in users[0] and "second" in users[0]
