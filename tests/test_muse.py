"""Muse Code as a host: detection, payload translation, its session log, the start context, the
live fold of a Muse session, no side calls, its native plugin manifest, and its install copy
finding the core.

Payloads and log records are synthetic, in the shape Muse Code 1.4.20 sent its plugin hooks and
wrote its `session.jsonl` in a disposable profile (2026-11-16), with fake ids and paths."""
import io
import json
import os
import shutil
import sys
import time

import pytest

from tests.conftest import git_sh
from tokeneconomizer import trace
from tokeneconomizer.plugin import hook, hosts, sidecall
from tokeneconomizer.plugin.hosts import Codex, Muse
from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
from tokeneconomizer.plugin.store import Store

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(REPO, "plugin")
SID = "01a1e628-a8a2-7f71-ac6d-f3f6958de68b"          # a UUIDv7 made on 2026-11-16 (UTC)
CHILD = "01a1e628-ab57-7231-b1d7-8518cd9eeb70"
FACTS = "Facts learned about this repository:\n- run tests with python -m pytest\n- no bare pytest"
EVENTS = {"SessionStart", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "Stop",
          "SessionEnd"}


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_MICROSKILLS": "envfacts", "GIT_CEILING_DIRECTORIES": str(tmp_path),
            "PATH": os.environ.get("PATH", ""), "XDG_DATA_HOME": str(tmp_path / "data"),
            "MUSE_SESSION_ID": SID}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    return str(r)


def payload(event, repo, sid=SID, **kw):
    return dict({"session_id": sid, "transcript_path": None, "cwd": repo,
                 "hook_event_name": event, "model": "muse-wonder-full-1.0",
                 "model_provider": "meta", "permission_mode": "default"}, **kw)


def shell(code, output):
    return json.dumps({"chunk_id": "exec-1-1", "command": "x", "exit_code": code,
                       "terminal_status": "completed" if code == 0 else "failed",
                       "output": output})


def call(env, p, host="muse"):
    out = io.StringIO()
    hook.main(["--host", host] if host else [], io.StringIO(json.dumps(p)), out, env)
    return json.loads(out.getvalue().splitlines()[-1])


def store(env, repo):
    return Store(repo_state_dir(repo_identity(repo, env), "muse", env))


def session_log(data, repo, sid=SID, calls=()):
    """A session.jsonl as 1.4.20 writes it: a retained frame first, then records."""
    d = os.path.join(data, "muse", "sessions", "2026", "11", "16", sid)
    os.makedirs(d, exist_ok=True)
    t = 1794862786732931
    seq = iter(range(2, 1000))

    def rec(payload_type, payload):
        return {"schema_version": 1, "id": "r", "stream": {"kind": "session", "id": sid},
                "sequence": next(seq), "recorded_at": t, "record_type": "event",
                "payload_type": payload_type, "payload": payload}

    def ev(event):
        return rec("runtime.session", {"kind": event["kind"], "run_id": "run1", "event": event})
    first = rec("runtime.session.permission_format_declared", {"format": "profile_v1"})
    lines = [{"retained_frame": "session_permission_transaction", "children": [
                 {"child_index": 0, "record_json": json.dumps(first)}]},
             rec("runtime.session.metadata", {"kind": "metadata", "record": {
                 "model_id": "muse-wonder-full-1.0", "provider_id": "meta",
                 "workspace_root": repo}}),
             ev({"kind": "started", "task_id": "t0"}),
             ev({"kind": "started", "prompt": "fix the tests"}),
             ev({"kind": "context_block_updated", "source": "runtime_hook", "text": FACTS})]
    for n, (cmd, code, out) in enumerate(calls):
        lines.append(ev({"kind": "assistant_message_committed", "phase": "commentary",
                         "message_id": f"c{n}", "text": "Running it."}))
        lines.append(ev({"kind": "assistant_tool_calls_committed", "message_id": f"m{n}",
                         "tool_calls": [{"name": "bash", "call_id": f"call_{n}", "id": f"fc_{n}",
                                         "args": json.dumps({"command": cmd,
                                                             "description": "d"})}]}))
        lines.append(ev({"kind": "model_completed", "model": "muse-wonder-full-1.0",
                         "usage": {"input_tokens": 1000, "cached_tokens": 600,
                                   "cache_read_tokens": 600, "cache_write_tokens": 10,
                                   "output_tokens": 20, "reasoning_tokens": 5}}))
        lines.append({"retained_marker": "omitted_live_only", "schema_version": 1})
        lines.append(ev({"kind": "tool_result_batch_committed", "results": [
            {"tool_call_id": f"call_{n}", "text": shell(code, out)}]}))
    lines.append(ev({"kind": "assistant_message_committed", "message_id": "z", "text": "done"}))
    lines.append(ev({"kind": "goal_usage_attribution", "record": {
        "owner": {"owner_type": "reminder_child"}, "quantity": {"input_tokens": 31046}}}))
    path = os.path.join(d, "session.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


# ---- detection ----------------------------------------------------------------------------
def test_muse_is_recognised_by_its_provider_field_before_codex_by_its_turn_id():
    p = payload("PostToolUse", "/w", turn_id="t1")
    assert isinstance(hosts.detect(p), Muse)
    assert isinstance(hosts.detect(p, hint="muse"), Muse)
    codex = {"session_id": "s", "turn_id": "t1", "transcript_path": "/x/.codex/rollout-1.jsonl",
             "model_provider": "openai"}
    assert isinstance(hosts.detect(codex), Codex)
    assert isinstance(hosts.detect({"session_id": "s", "turn_id": "t1"}), Codex)
    assert hosts.HOSTS["muse"] is Muse and Muse.name == "muse"


def test_a_te_command_under_muse_is_about_muse():
    assert hosts.default_host({"MUSE_SESSION_ID": "m"}) == "muse"
    assert hosts.default_host({"MUSE_SESSION_ID": "m", "CLAUDECODE": "1"}) == "claude"
    assert hosts.default_host({"MUSE_SESSION_ID": "m", "TE_HOST": "codex"}) == "codex"
    assert Muse().env_session_id({"MUSE_SESSION_ID": "m"}) == "m"
    assert Muse().env_session_id({"MUSE_SESSION_ID": "m", "TE_SESSION_ID": "t"}) == "t"


# ---- payload translation ------------------------------------------------------------------
def test_the_turn_is_turn_id_and_the_log_is_found_from_the_session_id(tmp_path, repo):
    data = str(tmp_path / "data")
    path = session_log(data, repo)
    env = {"XDG_DATA_HOME": data, "MUSE_SESSION_ID": SID}
    p = Muse().prepare(payload("UserPromptSubmit", repo, prompt="go", turn_id="t1"), env)
    assert Muse().turn_key(p) == "t1"
    assert p["transcript_path"] == path and p["hook_event_name"] == "UserPromptSubmit"
    assert trace.muse_find_session(os.path.join(data, "muse", "sessions"), "nope") is None


def test_a_reminder_agents_tool_calls_are_not_the_sessions(tmp_path, repo):
    env = {"XDG_DATA_HOME": str(tmp_path / "data"), "MUSE_SESSION_ID": SID}
    p = Muse().prepare(payload("PostToolUse", repo, sid=CHILD, tool_name="submit_reminder_decision",
                               tool_input={}, tool_response="{}", turn_id="t9"), env)
    assert p["hook_event_name"] == ""
    p = Muse().prepare(payload("PostToolUse", repo, tool_name="bash"), env)
    assert p["hook_event_name"] == "PostToolUse"


def test_a_shell_result_is_read_from_its_json(tmp_path):
    host = Muse()
    name, args, text, code = host.tool_call(payload(
        "PostToolUse", "/w", tool_name="bash",
        tool_input={"command": "cat README.md", "description": "d", "workdir": "/w"},
        tool_response=shell(0, "hi\n")))
    assert (name, args, text, code) == ("Bash", {"command": "cat README.md"}, "hi\n", 0)
    name, _, text, code = host.tool_call(payload(
        "PostToolUseFailure", "/w", tool_name="bash", tool_input={"command": "ls nosuchfile"},
        error=shell(2, "ls: cannot access 'nosuchfile'"), is_interrupt=False))
    assert (name, code) == ("Bash", 2) and "cannot access" in text
    name, args, text, code = host.tool_call(payload(
        "PostToolUse", "/w", tool_name="read_file", tool_input={"path": "/w/a.py"},
        tool_response="x = 1"))
    assert (name, args, text, code) == ("Read", {"file_path": "/w/a.py"}, "x = 1", None)


# ---- the session log ----------------------------------------------------------------------
def test_the_log_reads_as_calls_prompts_and_usage(tmp_path, repo):
    path = session_log(str(tmp_path / "data"), repo, calls=[
        ("pytest -q", 127, "bash: pytest: command not found"), ("python -m pytest -q", 0, "ok")])
    head = trace.muse_session_head(path)
    assert head == {"session_id": SID, "cwd": repo, "model": "muse-wonder-full-1.0",
                    "started": 1794862786.732931}
    assert trace.muse_prompt_count(path) == 1 and Muse().prompt_count(path) == 1
    out = Muse().mechanism_trace(path, str(tmp_path))
    calls = trace.load_trajectory(out)
    assert [(c[0], c[1], c[3]) for c in calls] == [
        ("Bash", {"command": "pytest -q"}, False), ("Bash", {"command": "python -m pytest -q"},
                                                     True)]
    assert trace.claude_prompt_count(out) == 1
    usage = Muse().session_usage(path)
    assert usage == {"input_tokens": 800, "cache_read_input_tokens": 1200,
                     "cache_creation_input_tokens": 20, "output_tokens": 40, "requests": 2,
                     "model": "muse-wonder-full-1.0"}
    assert Muse().mechanism_trace(str(tmp_path / "none.jsonl"), str(tmp_path)) is None


def test_muse_tools_are_named_as_the_mechanisms_know_them():
    assert trace.muse_tool("write_file", {"path": "/w/a"}) == ("Write", {"file_path": "/w/a"})
    assert trace.muse_tool("edit_file", {"path": "/w/a"}) == ("Edit", {"file_path": "/w/a"})
    assert trace.muse_tool("search", {"pattern": "x", "path": "/w"}) == \
        ("Grep", {"pattern": "x", "path": "/w"})
    assert trace.muse_tool("workflow", {"a": 1}) == ("workflow", {"a": 1})


# ---- injection ----------------------------------------------------------------------------
def test_the_start_context_goes_back_at_session_start(env, repo):
    store(env, repo).write("injection.json", {"rendered": FACTS, "rendered_ids": ["a", "b"]})
    out = call(env, payload("SessionStart", repo, source="startup"))
    assert out["hookSpecificOutput"] == {"hookEventName": "SessionStart",
                                         "additionalContext": FACTS}
    assert store(env, repo).read(f"sessions/{SID}.json")["injected_chars"] == len(FACTS)
    assert call(env, payload("UserPromptSubmit", repo, prompt="go", turn_id="t1")) == {}


# ---- the live fold ------------------------------------------------------------------------
def test_a_finished_muse_session_is_folded_once_at_session_end(env, repo, tmp_path):
    call(env, payload("SessionStart", repo, source="startup"))
    call(env, payload("UserPromptSubmit", repo, prompt="fix the tests", turn_id="t1"))
    session_log(env["XDG_DATA_HOME"], repo, calls=[
        ("pytest -q", 127, "bash: pytest: command not found"), ("python -m pytest -q", 0, "ok")])
    call(env, payload("PostToolUseFailure", repo, tool_name="bash", turn_id="t1",
                      tool_input={"command": "pytest -q"},
                      error=shell(127, "bash: pytest: command not found")))
    call(env, payload("PostToolUse", repo, tool_name="bash", turn_id="t1",
                      tool_input={"command": "python -m pytest -q"}, tool_response=shell(0, "ok")))
    call(env, payload("PostToolUse", repo, sid=CHILD, tool_name="submit_reminder_decision",
                      tool_input={}, tool_response="{}", turn_id="t9"))
    call(env, payload("Stop", repo, turn_id="t1", stop_hook_active=False,
                      last_assistant_message="done"))
    call(env, payload("SessionEnd", repo, reason="other"))
    st = store(env, repo)
    assert SID in (st.read("folded.json") or {})
    assert st.read(f"sessions/{CHILD}.json") is None
    ledger = st.read_lines("ledger.jsonl")
    assert len(ledger) == 1 and ledger[0]["session_id"] == SID
    call(env, payload("SessionEnd", repo, reason="other"))
    assert len(st.read_lines("ledger.jsonl")) == 1


def test_a_log_not_recorded_at_the_start_is_found_at_the_fold(tmp_path, repo):
    data = str(tmp_path / "data")
    path = session_log(data, repo)
    host = Muse()
    assert host.complete("", SID, repo, {"muse_data_dir": os.path.join(data, "muse")}) == path
    assert host.session_state({"XDG_DATA_HOME": data}) == {
        "muse_data_dir": os.path.join(data, "muse")}


def test_in_a_hook_the_data_root_is_the_one_around_the_plugins_data_dir(tmp_path):
    """1.4.20 drops `XDG_DATA_HOME` from a hook's environment and sets
    `MUSE_PLUGIN_DATA_DIR` to `<data root>/plugins/data/<plugin>`."""
    root = str(tmp_path / "d" / "muse")
    env = {"MUSE_PLUGIN_DATA_DIR": os.path.join(root, "plugins", "data", "tokeneconomizer"),
           "XDG_DATA_HOME": "/elsewhere", "HOME": "/home/u"}
    assert trace.muse_data_dir(env) == root
    assert trace.muse_data_dir({"MUSE_PLUGIN_DATA_DIR": "/x/y", "HOME": "/home/u"}) == \
        os.path.join("/home/u", ".local", "share", "muse")
    assert trace.muse_data_dir({"HOME": "/home/u"}) == "/home/u/.local/share/muse"


# ---- side calls ---------------------------------------------------------------------------
def test_muse_gets_no_side_call_and_never_falls_back_to_claude(monkeypatch):
    monkeypatch.setattr(sidecall, "claude_call", lambda *a, **k: pytest.fail("no Claude call"))
    config = type("C", (), {"paid_model": "haiku"})()
    assert "muse" in sidecall.NO_SIDE_CALLS
    assert sidecall.runner_for("muse", config) is None
    assert sidecall.call_name("muse") == "muse exec"


# ---- packaging ----------------------------------------------------------------------------
def _json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def test_the_native_manifest_runs_one_launcher_per_event():
    doc = _json(PLUGIN, ".muse-plugin", "plugin.json")
    assert doc["name"] == _json(PLUGIN, ".claude-plugin", "plugin.json")["name"]
    assert doc["version"] == _json(PLUGIN, ".claude-plugin", "plugin.json")["version"]
    assert doc["compat"] == {"source": "native", "manifestDir": ".muse-plugin"}
    hooks = doc["capabilities"]["hooks"]
    assert {h["event"] for h in hooks} == EVENTS
    # Muse refuses a plugin whose hooks share a source file (1.4.20).
    sources = [h["command"][1] for h in hooks]
    assert len(set(sources)) == len(sources) == len(hooks)
    assert len({h["id"] for h in hooks}) == len(hooks)
    for h in hooks:
        assert h["command"][0] == "sh" and h["timeoutMs"] <= 60000
        with open(os.path.join(PLUGIN, h["command"][1]), encoding="utf-8") as fh:
            body = fh.read()
        assert '"${0%/*}/../scripts/te-hook" --host muse "$@"' in body


def test_a_launcher_answers_through_the_hook_dispatcher(tmp_path):
    import subprocess
    sh = git_sh()
    if not sh:
        pytest.skip("no sh")
    env = dict(os.environ, TOKENECONOMIZER_HOME=str(tmp_path / "te"), TE_DISABLE="1",
               TE_PYTHON=sys.executable)
    out = subprocess.run([sh, os.path.join(PLUGIN, "muse", "session-start")],
                         input=json.dumps(payload("SessionStart", str(tmp_path))),
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0 and json.loads(out.stdout.strip().splitlines()[-1]) == {}


def _bootstrap():
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    try:
        import te_bootstrap
    finally:
        sys.path.pop(0)
    return te_bootstrap


def test_a_muse_install_copy_finds_the_core_in_the_checkout_it_came_from(tmp_path):
    """Muse's install copies the plugin directory without the `lib` symlink, into
    `plugins/cache/local/<plugin>/<sha>/package`, and records where it came from in
    `plugins/installed.json` (1.4.20)."""
    te_bootstrap = _bootstrap()
    plugins = tmp_path / "data" / "muse" / "plugins"
    copy = plugins / "cache" / "local" / "tokeneconomizer" / ("ab" * 32) / "package"
    copy.mkdir(parents=True)
    (plugins / "installed.json").write_text(json.dumps({"version": 1, "plugins": {
        "other": {"id": "other", "source": {"provenance": "native-local", "path": "/nowhere"}},
        "tokeneconomizer": {"id": "tokeneconomizer", "manifest_family": "muse",
                            "source": {"provenance": "native-local", "path": PLUGIN}}}}))
    found = te_bootstrap._muse_sources(str(copy))
    assert REPO in found and "/nowhere" not in found
    assert te_bootstrap._muse_sources(str(tmp_path / "x" / "package")) == []
    os.remove(str(plugins / "installed.json"))
    assert te_bootstrap._muse_sources(str(copy)) == []
