"""Every agent's history as a source, and what crosses between agents.

Each source gets a synthetic store in the shape its agent writes (verified on the real logs
named in `trace/sources.py`), under a home the test owns: a source finds a store only through
the environment it is given, never this machine's. The cross-agent cases run the hooks inline
(`TE_FOLD_SYNC=1`) and check the stores, not the hooks' replies.
"""
import io
import json
import os
import sqlite3
import time
from urllib.parse import quote

import pytest

from test_history_import import (CLAIM, ctx_for, detached, history_log, ledger_ids,
                                 live_session, start_hook, trap)
from test_live_hooks import call, store_for, trap_session  # noqa: F401
from tokeneconomizer import trace
from tokeneconomizer.microskills import observe
from tokeneconomizer.plugin import cli, fold, history
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.paths import repo_identity
from tokeneconomizer.trace import sources

HOME_KEYS = ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "COPILOT_HOME", "PI_CODING_AGENT_DIR",
             "GROK_HOME", "HERMES_HOME", "XDG_DATA_HOME", "FACTORY_HOME")


@pytest.fixture
def home(tmp_path):
    """A home with every agent's store directory, empty."""
    h = tmp_path / "home"
    for d in (".claude/projects", ".codex/sessions/2026/11/08", ".copilot/session-state",
              ".pi/agent/sessions", ".grok/sessions", ".hermes", ".local/share/opencode",
              ".factory/sessions", ".openclaw/agents/main/agent/codex-home/sessions"):
        (h / d).mkdir(parents=True)
    return h


@pytest.fixture
def env(tmp_path, home):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "te"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "TE_MICROSKILLS": "envfacts",
            "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    (r / "pyproject.toml").write_text("[project]\nname='w'\n")
    return str(r)


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000Z"


def _age(path, age_s):
    end = time.time() - age_s
    os.utime(path, (end, end))
    return path


def trap_calls(repo, i):
    """The calls of a session that meets the missing `pytest` and works around it."""
    return [("Bash", {"command": "pytest -q tests/test_a.py"},
             "Exit code 127\nbash: pytest: command not found", True),
            ("Bash", {"command": "python -m pytest -q tests/test_a.py"}, "1 passed in 0.1s",
             False)]


# ---- one store per agent, in its own shape ---------------------------------------------------
def codex_log(home, cwd, sid, i, age_s=3600, source="vscode"):
    d = home / ".codex" / "sessions" / "2026" / "11" / "08"
    path = d / f"rollout-2026-11-08T10-00-0{i}-{sid}.jsonl"
    t = time.time() - age_s - 300
    lines = [{"timestamp": _iso(t), "type": "session_meta",
              "payload": {"id": sid, "timestamp": _iso(t), "cwd": cwd, "originator": "codex-tui",
                          "cli_version": "0.163.0", "source": source}},
             {"timestamp": _iso(t), "type": "event_msg", "payload": {"type": "task_started"}},
             {"timestamp": _iso(t), "type": "response_item",
              "payload": {"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": f"fix bug {i}"}]}}]
    for n, (name, args, out, failed) in enumerate(trap_calls(cwd, i)):
        lines.append({"timestamp": _iso(t), "type": "event_msg", "payload": {
            "type": "item_completed", "item": {
                "id": f"item_{n}", "type": "CommandExecution", "command": ["bash", "-lc",
                                                                           args["command"]],
                "aggregated_output": out, "exit_code": 127 if failed else 0,
                "status": "failed" if failed else "completed"}}})
    lines.append({"timestamp": _iso(t), "type": "event_msg",
                  "payload": {"type": "task_complete", "last_agent_message": "done"}})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(str(path), age_s)


def copilot_log(home, cwd, sid, i, age_s=3600):
    d = home / ".copilot" / "session-state" / sid
    d.mkdir(parents=True)
    t = time.time() - age_s - 300
    lines = [{"type": "session.start", "timestamp": _iso(t), "id": "e0",
              "data": {"sessionId": sid, "copilotVersion": "1.0.106", "startTime": _iso(t),
                       "context": {"cwd": cwd}}},
             {"type": "user.message", "timestamp": _iso(t), "id": "e1",
              "data": {"content": f"fix bug {i}"}}]
    for n, (name, args, out, failed) in enumerate(trap_calls(cwd, i)):
        lines.append({"type": "tool.execution_start", "timestamp": _iso(t), "id": f"s{n}",
                      "data": {"toolCallId": f"c{n}", "toolName": "bash", "arguments": args}})
        lines.append({"type": "tool.execution_complete", "timestamp": _iso(t), "id": f"d{n}",
                      "data": {"toolCallId": f"c{n}", "success": not failed,
                               "result": {"content": out}}})
    lines.append({"type": "session.shutdown", "timestamp": _iso(t), "id": "e9",
                  "data": {"modelMetrics": {}}})
    path = d / "events.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(str(path), age_s)


def pi_log(home, cwd, sid, i, age_s=3600):
    d = home / ".pi" / "agent" / "sessions" / trace.claude_project_folder(cwd)
    d.mkdir(parents=True, exist_ok=True)
    t = time.time() - age_s - 300
    lines = [{"type": "session", "version": 3, "id": sid, "timestamp": _iso(t), "cwd": cwd},
             {"type": "message", "id": "m1", "parentId": None, "timestamp": _iso(t),
              "message": {"role": "user", "content": [{"type": "text", "text": f"fix {i}"}]}}]
    prev = "m1"
    for n, (name, args, out, failed) in enumerate(trap_calls(cwd, i)):
        lines.append({"type": "message", "id": f"a{n}", "parentId": prev, "timestamp": _iso(t),
                      "message": {"role": "assistant", "content": [
                          {"type": "toolCall", "id": f"tc{n}", "name": "bash",
                           "arguments": args}], "model": "x"}})
        lines.append({"type": "message", "id": f"r{n}", "parentId": f"a{n}",
                      "timestamp": _iso(t), "message": {
                          "role": "toolResult", "toolCallId": f"tc{n}", "isError": failed,
                          "content": [{"type": "text", "text": out}]}})
        prev = f"r{n}"
    path = d / f"2026-11-08T10-00-0{i}-000Z_{sid}.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(str(path), age_s)


def grok_log(home, cwd, sid, i, age_s=3600):
    d = home / ".grok" / "sessions" / quote(cwd, safe="") / sid
    d.mkdir(parents=True)
    t = time.time() - age_s - 300
    lines = [{"timestamp": int(t), "method": "session/update", "params": {
        "sessionId": sid, "update": {"sessionUpdate": "user_message_chunk",
                                     "content": {"type": "text", "text": f"fix bug {i}"},
                                     "_meta": {"promptIndex": 0}}}}]
    for n, (name, args, out, failed) in enumerate(trap_calls(cwd, i)):
        lines.append({"timestamp": int(t), "method": "session/update", "params": {
            "sessionId": sid, "update": {"sessionUpdate": "tool_call", "toolCallId": f"tc{n}",
                                         "title": "run_terminal_command", "rawInput": args}}})
        lines.append({"timestamp": int(t), "method": "session/update", "params": {
            "sessionId": sid, "update": {
                "sessionUpdate": "tool_call_update", "toolCallId": f"tc{n}",
                "status": "failed" if failed else "completed",
                "rawOutput": {"type": "Bash", "output": list(out.encode()),
                              "exit_code": 127 if failed else 0,
                              "output_for_prompt": out, "current_dir": cwd}}}})
    lines.append({"timestamp": int(t), "method": "session/update", "params": {
        "sessionId": sid, "update": {"sessionUpdate": "turn_completed"}}})
    path = d / "updates.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(str(path), age_s)


def droid_log(home, cwd, sid, i, age_s=3600):
    d = home / ".factory" / "sessions" / trace.claude_project_folder(cwd)
    d.mkdir(parents=True, exist_ok=True)
    t = time.time() - age_s - 300
    lines = [{"type": "session_start", "id": sid, "title": "s", "owner": "u", "version": 2,
              "cwd": cwd},
             {"type": "message", "id": "p1", "timestamp": _iso(t),
              "message": {"role": "user", "content": [{"type": "text", "text": f"fix {i}"}]}}]
    for n, (name, args, out, failed) in enumerate(trap_calls(cwd, i)):
        lines.append({"type": "message", "id": f"a{n}", "timestamp": _iso(t), "message": {
            "role": "assistant", "modelId": "m", "content": [
                {"type": "tool_use", "id": f"tu{n}", "name": "Execute", "input": args}]}})
        lines.append({"type": "message", "id": f"r{n}", "timestamp": _iso(t), "message": {
            "role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu{n}",
                                         "content": out, "is_error": failed}]}})
    path = d / f"{sid}.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(str(path), age_s)


def hermes_db(home, sessions):
    """`state.db` with the tables the reader uses; `sessions` is [(sid, cwd, age_s, source)]."""
    db = home / ".hermes" / "state.db"
    con = sqlite3.connect(str(db))
    con.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, created_source TEXT, cwd TEXT,
            model TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, end_reason TEXT,
            message_count INTEGER);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
            content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL,
            message_uid TEXT, _compressed_summary INTEGER);
        CREATE TABLE session_model_usage (session_id TEXT, model TEXT, prompt_tokens INTEGER,
            completion_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
            reasoning_tokens INTEGER, requests INTEGER);
    """)
    for sid, cwd, age_s, source in sessions:
        end = time.time() - age_s
        con.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (sid, source, source, cwd, "m", None, end - 300, end, "complete", 5))
        con.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES "
                    "(?,?,?,?)", (sid, "user", "fix the tests", end - 300))
        for n, (name, args, out, failed) in enumerate(trap_calls(cwd, 0)):
            con.execute("INSERT INTO messages (session_id, role, content, tool_calls, timestamp) "
                        "VALUES (?,?,?,?,?)", (sid, "assistant", "", json.dumps([{
                            "id": f"c{n}", "type": "function", "function": {
                                "name": "terminal", "arguments": json.dumps(args)}}]),
                                               end - 200 + n))
            con.execute("INSERT INTO messages (session_id, role, content, tool_call_id, "
                        "tool_name, timestamp) VALUES (?,?,?,?,?,?)",
                        (sid, "tool", json.dumps({"output": out,
                                                  "exit_code": 127 if failed else 0}),
                         f"c{n}", "terminal", end - 199 + n))
    con.commit()
    con.close()
    return str(db)


def opencode_db(home, sessions):
    """`opencode.db` with the tables the reader uses; `sessions` is [(sid, cwd, age_s)]."""
    db = home / ".local" / "share" / "opencode" / "opencode.db"
    con = sqlite3.connect(str(db))
    con.executescript("""
        CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, slug TEXT,
            directory TEXT, title TEXT, version TEXT, time_created INTEGER, time_updated INTEGER);
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,
            time_updated INTEGER, data TEXT);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
            time_created INTEGER, time_updated INTEGER, data TEXT);
    """)
    for sid, cwd, age_s in sessions:
        end = int((time.time() - age_s) * 1000)
        con.execute("INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?)",
                    (sid, "p", None, "s", cwd, "t", "1", end - 300000, end))
        con.execute("INSERT INTO message VALUES (?,?,?,?,?)",
                    (f"{sid}_u", sid, end - 300000, end - 300000, json.dumps(
                        {"role": "user", "time": {"created": end - 300000}})))
        con.execute("INSERT INTO part VALUES (?,?,?,?,?,?)",
                    (f"{sid}_u_p", f"{sid}_u", sid, end - 300000, end - 300000,
                     json.dumps({"type": "text", "text": "fix the tests"})))
        con.execute("INSERT INTO message VALUES (?,?,?,?,?)",
                    (f"{sid}_a", sid, end - 200000, end, json.dumps(
                        {"role": "assistant", "modelID": "m", "providerID": "p",
                         "path": {"cwd": cwd, "root": cwd}, "cost": 0,
                         "tokens": {"input": 1, "output": 1, "reasoning": 0,
                                    "cache": {"read": 0, "write": 0}},
                         "time": {"created": end - 200000, "completed": end}})))
        for n, (name, args, out, failed) in enumerate(trap_calls(cwd, 0)):
            con.execute("INSERT INTO part VALUES (?,?,?,?,?,?)",
                        (f"{sid}_a_p{n}", f"{sid}_a", sid, end - 100000 + n, end,
                         json.dumps({"type": "tool", "tool": "bash", "callID": f"c{n}",
                                     "state": dict({"status": "error" if failed else "completed",
                                                    "input": args},
                                                   **({"error": out} if failed
                                                      else {"output": out}))})))
    con.commit()
    con.close()
    return str(db)


# ---- the contract ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(sources.SOURCES))
def test_a_source_finds_nothing_without_a_home_and_nothing_in_an_empty_one(name, env, home):
    src = sources.SOURCES[name]
    assert src.logs({}) == []
    assert src.logs({"PATH": "/usr/bin"}) == []
    assert src.logs(env) == []


def _write(name, home, cwd, sid, i=0, age_s=3600):
    if name == "claude":
        return history_log(home / ".claude", cwd, sid, trap(cwd, i), age_s=age_s)
    if name == "codex":
        return codex_log(home, cwd, sid, i, age_s)
    if name == "copilot":
        return copilot_log(home, cwd, sid, i, age_s)
    if name == "pi":
        return pi_log(home, cwd, sid, i, age_s)
    if name == "grok":
        return grok_log(home, cwd, sid, i, age_s)
    if name == "droid":
        return droid_log(home, cwd, sid, i, age_s)
    if name == "hermes":
        return hermes_db(home, [(sid, cwd, age_s, "cli")])
    if name == "opencode":
        return opencode_db(home, [(sid, cwd, age_s)])
    raise AssertionError(name)


SID = {"codex": "0b6f3c2e-6a4f-4c39-9b55-1f2d9d1c3b7a", "pi": "0b6f3c2e-6a4f-4c39-9b55-1f2d9d1c3b7b"}


@pytest.mark.parametrize("name", sorted(sources.SOURCES))
def test_every_source_lists_its_log_reads_its_head_and_writes_a_trace_the_mechanisms_read(
        name, env, home, repo, tmp_path):
    sid = SID.get(name, f"{name}-s1")
    _write(name, home, repo, sid, age_s=3600)
    src = sources.SOURCES[name]
    logs = src.logs(env)
    assert [x["id"] for x in logs] == [sid]
    log = logs[0]
    assert time.time() - 4000 < log["mtime"] < time.time() - 3000
    head = src.head(log["path"])
    assert head["session_id"] == sid and head["cwd"] == repo and not head["headless"]
    assert head["started"] is None or head["started"] <= log["mtime"] + 1
    assert src.prompt_count(log["path"]) == 1
    work = tmp_path / "work"
    work.mkdir()
    got = src.materialize(log["path"], str(work))
    assert got and os.path.isfile(got)
    tr = src.to_trace(got, str(work))
    assert tr and os.path.isfile(tr)
    shell = [c for c in trace.load_trajectory(tr) if isinstance(c[1], dict) and c[1].get("command")]
    cmds = [observe.unwrap(a["command"]) for n, a, _, _ in shell]
    assert cmds == ["pytest -q tests/test_a.py", "python -m pytest -q tests/test_a.py"]
    assert [ok for _, _, _, ok in shell] == [False, True]
    assert "command not found" in shell[0][2]
    assert isinstance(src.usage(got), dict)


def test_a_scripted_codex_run_and_a_one_shot_hermes_session_are_headless(env, home, repo):
    codex_log(home, repo, SID["codex"], 0, source="exec")
    hermes_db(home, [("h1", repo, 3600, "oneshot")])
    assert sources.SOURCES["codex"].head(sources.SOURCES["codex"].logs(env)[0]["path"])["headless"]
    assert sources.SOURCES["hermes"].head(sources.SOURCES["hermes"].logs(env)[0]["path"])[
        "headless"]


def test_openclaws_embedded_codex_homes_are_codex_roots(env, home, repo):
    d = home / ".openclaw" / "agents" / "main" / "agent" / "codex-home" / "sessions" / "2026"
    d.mkdir(parents=True)
    (d / f"rollout-2026-01-01T00-00-00-{SID['codex']}.jsonl").write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
        "payload": {"id": SID["codex"], "cwd": repo}}) + "\n")
    assert [x["id"] for x in sources.SOURCES["codex"].logs(env)] == [SID["codex"]]
    assert sources.codex_session_roots({"HOME": str(home)})[0] == str(
        home / ".codex" / "sessions")


# ---- what crosses between agents -------------------------------------------------------------
def test_the_first_start_in_a_repository_reads_every_agents_sessions_oldest_first(
        env, home, repo, tmp_path):
    """Claude Code's store learns the repository's fact from four agents' sessions, in the
    order they ended; tool pruning counts only Claude Code's own."""
    codex_log(home, repo, SID["codex"], 0, age_s=4 * 3600)
    history_log(home / ".claude", repo, "c-old", trap(repo, 1), age_s=3 * 3600)
    pi_log(home, repo, SID["pi"], 2, age_s=2 * 3600)
    droid_log(home, repo, "droid-1", 3, age_s=1 * 3600)
    start_hook(env, repo, "new1", tmp_path)
    st = store_for(env, repo)
    assert ledger_ids(st) == [f"codex:{SID['codex']}", "c-old", f"pi:{SID['pi']}",
                              "droid:droid-1"]
    recs = {r["session_id"]: r for r in st.read_lines("ledger.jsonl") if r["kind"] == "session"}
    assert recs["c-old"]["origin"] == "claude" and recs["c-old"]["untool"] is not None
    assert all(recs[k]["untool"] is None for k in recs if k != "c-old")
    doc = st.read("history.json")
    assert doc["imported"] == 4 and doc["origins"] == {"claude": 1, "codex": 1, "droid": 1,
                                                         "pi": 1}
    assert st.read("untool.json")["sessions"] == 1          # the Claude session only
    second = start_hook(env, repo, "new2", tmp_path)
    assert CLAIM in second["hookSpecificOutput"]["additionalContext"]
    assert ("read 4 earlier session(s) of acme/widget from Claude Code, Codex, Factory Droid "
            "and Pi history.") in second["systemMessage"]


def test_a_live_fold_first_folds_the_other_agents_sessions_that_ended_before_it(
        env, home, repo, tmp_path):
    """Codex's session ends in a repository Claude Code has already learned in; Claude Code's
    next fold, of its own later session, folds Codex's first, so it is not stranded behind."""
    ctx = ctx_for(env, repo)
    ctx.store.write("history.json", {"state": "done", "through": time.time() - 7200,
                                     "caught_up": time.time(), "imported": 0, "noticed": True})
    for i in range(3):
        codex_log(home, repo, f"0b6f3c2e-6a4f-4c39-9b55-1f2d9d1c3b{i:02d}", i, age_s=3600 + i)
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5),
                 started=time.time() - 600)
    rec = fold.fold_session(ctx, "live0")
    assert rec is not None
    ids = ledger_ids(ctx.store)
    assert ids[-1] == "live0" and len(ids) == 4 and all(i.startswith("codex:") for i in ids[:3])
    doc = ctx.store.read("history.json")
    assert doc["imported"] == 3 and doc["origins"] == {"codex": 3} and doc["noticed"] is False
    assert doc["through"] >= time.time() - 60
    assert history.notice(ctx) == ("TokenEconomizer read 3 earlier session(s) of acme/widget "
                                   "from Codex's history.")
    # A Codex session that ends after this fold waits for the next start, which catches up.
    codex_log(home, repo, "0b6f3c2e-6a4f-4c39-9b55-1f2d9d1c3b99", 9, age_s=30)
    assert fold.fold_session(ctx, "live0") is None          # folded already; nothing else


def test_a_later_catch_up_skips_what_is_behind_the_watermark(env, home, repo, tmp_path):
    history_log(home / ".claude", repo, "c-old", trap(repo, 1), age_s=3 * 3600)
    start_hook(env, repo, "new1", tmp_path)
    st = store_for(env, repo)
    assert ledger_ids(st) == ["c-old"]
    # Another agent's session that ended before Claude's, found only now: behind.
    codex_log(home, repo, SID["codex"], 0, age_s=4 * 3600)
    st.write("history.json", dict(st.read("history.json"), caught_up=0))
    start_hook(env, repo, "new2", tmp_path)
    doc = st.read("history.json")
    assert ledger_ids(st) == ["c-old"] and doc["skipped"].get("behind") == 1


def test_history_sources_self_reads_only_this_agents_logs(env, home, repo, tmp_path):
    codex_log(home, repo, SID["codex"], 0, age_s=7200)
    history_log(home / ".claude", repo, "c-old", trap(repo, 1), age_s=3600)
    start_hook(dict(env, TE_HISTORY_SOURCES="self"), repo, "new1", tmp_path)
    assert ledger_ids(store_for(env, repo)) == ["c-old"]
    other = dict(env, TOKENECONOMIZER_HOME=env["TOKENECONOMIZER_HOME"] + "2",
                 TE_HISTORY_SOURCES="codex,droid")
    start_hook(other, repo, "new1", tmp_path)
    assert ledger_ids(store_for(other, repo)) == [f"codex:{SID['codex']}"]


def test_import_all_reads_every_repository_into_each_agent_that_ran_there(
        env, home, repo, tmp_path, monkeypatch, capsys):
    other = tmp_path / "other"
    (other / ".git" / "info").mkdir(parents=True)
    (other / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (other / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/gadget.git\n')
    history_log(home / ".claude", repo, "c1", trap(repo, 1), age_s=7200)
    codex_log(home, repo, SID["codex"], 0, age_s=3600)
    pi_log(home, str(other), SID["pi"], 2, age_s=3600)
    droid_log(home, str(other), "droid-1", 3, age_s=1800)      # no store of its own
    gone = tmp_path / "gone"
    gone.mkdir()
    history_log(home / ".claude", str(gone), "g1", trap(str(gone), 1), age_s=3600)
    gone.rmdir()
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 repositories" in out and "acme/widget [claude]: 2 session(s)" in out
    assert "acme/gadget [pi]: 2 session(s)" in out and "[droid]" not in out
    assert store_for(env, repo).read("history.json") is None
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all"]) == 0
    assert "3 store(s), 6 session(s) read" in capsys.readouterr().out
    assert ledger_ids(store_for(env, repo)) == ["c1", f"codex:{SID['codex']}"]
    assert ledger_ids(store_for(env, repo, host="codex")) == ["claude:c1", SID["codex"]]
    assert ledger_ids(store_for(env, str(other), host="pi")) == [SID["pi"], "droid:droid-1"]
    assert not os.path.isdir(store_for(env, str(other), host="claude").base)
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all"]) == 0
    assert "0 session(s) read" in capsys.readouterr().out


def test_the_head_cache_is_derived_state(env, home, repo, tmp_path):
    codex_log(home, repo, SID["codex"], 0, age_s=3600)
    ctx = ctx_for(env, repo)
    p = history.plan(ctx)
    cache = ctx.store.__class__(ctx.root).read(history.CACHE)
    path = p["sessions"][0]["path"]
    assert cache[path]["head"]["cwd"] == repo and cache[path]["prompts"] == 1
    os.remove(os.path.join(ctx.root, history.CACHE))
    assert history.plan(ctx) == p


def test_every_agents_shell_wrapper_is_stripped_before_a_command_is_read():
    for wrapped in ("/bin/bash -lc 'pytest -q'", "bash -lc 'pytest -q'", '/bin/sh -c "pytest -q"',
                    "/etc/profiles/per-user/u/bin/zsh -c 'pytest -q'", "zsh -lc 'pytest -q'"):
        assert observe.unwrap(wrapped) == "pytest -q", wrapped
    assert observe.unwrap("pytest -q") == "pytest -q"
    assert observe.unwrap("nu -c 'pytest -q'") == "nu -c 'pytest -q'"      # not a wrapper
