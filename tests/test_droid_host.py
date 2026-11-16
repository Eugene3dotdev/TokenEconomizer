"""Factory Droid as a host: detection, payload translation, the start context, the live fold of
a Droid session, no side calls, its plugin manifest, and its install copy finding the core.

Payloads are synthetic, in the shape Droid 1.12.1 sent its plugin hooks in a disposable profile
(2026-11-16): Claude Code's snake_case fields plus `permission_mode` values of Droid's own, with
fake ids and paths. Droid runs the plugin's Claude-format `hooks/hooks.json`, with no `--host`."""
import io
import json
import os
import shutil
import sys

import pytest

from tests.test_droid import ENTRIES, assistant, user, write_droid_log
from tokeneconomizer import trace
from tokeneconomizer.plugin import hook, hosts, sidecall
from tokeneconomizer.plugin.hosts import ClaudeCode, Droid
from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
from tokeneconomizer.plugin.store import Store

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(REPO, "plugin")
SID = "60238b32-0000-4000-8000-000000000001"
# Droid 1.12.1's own list, from the warning it gives for any other event name.
DROID_EVENTS = {"PreToolUse", "PostToolUse", "Notification", "UserPromptSubmit", "Stop",
                "SubagentStop", "PreCompact", "SessionStart", "SessionEnd"}
FACTS = "Facts learned about this repository:\n- run tests with python -m pytest\n- no bare pytest"


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_MICROSKILLS": "envfacts", "GIT_CEILING_DIRECTORIES": str(tmp_path),
            "PATH": os.environ.get("PATH", ""), "FACTORY_PROJECT_DIR": "/elsewhere"}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    return str(r)


def sessions(tmp_path):
    return str(tmp_path / "fhome" / ".factory" / "sessions")


def transcript(tmp_path, repo, sid=SID):
    return os.path.join(sessions(tmp_path), trace.claude_project_folder(repo), f"{sid}.jsonl")


def payload(event, repo, tpath, sid=SID, **kw):
    return dict({"session_id": sid, "transcript_path": tpath, "cwd": repo,
                 "permission_mode": "auto-low", "hook_event_name": event}, **kw)


def call(env, p, host=None):
    out = io.StringIO()
    hook.main(["--host", host] if host else [], io.StringIO(json.dumps(p)), out, env)
    return json.loads(out.getvalue().splitlines()[-1])


def store(env, repo):
    return Store(repo_state_dir(repo_identity(repo, env), "droid", env))


# ---- detection ----------------------------------------------------------------------------
def test_droid_is_recognised_from_the_claude_hook_file_it_runs_with_no_host_flag():
    tp = "/home/u/.factory/sessions/-work-repo/s.jsonl"
    assert isinstance(hosts.detect(payload("SessionStart", "/work/repo", tp)), Droid)
    no_mode = {"session_id": "s", "transcript_path": tp, "hook_event_name": "Stop"}
    assert isinstance(hosts.detect(no_mode), Droid)                  # by its session directory
    win = {"session_id": "s", "transcript_path": r"C:\Users\u\.factory\sessions\-w\s.jsonl"}
    assert isinstance(hosts.detect(win), Droid)
    moved = {"session_id": "s", "transcript_path": "/x/s.jsonl", "permission_mode": "auto-high"}
    assert isinstance(hosts.detect(moved), Droid)                    # by its permission mode
    assert isinstance(hosts.detect(payload("Stop", "/w", tp), hint="claude"), ClaudeCode)
    assert hosts.HOSTS["droid"] is Droid and Droid.name == "droid"


def test_claude_code_stays_claude_code_beside_droid():
    claude = {"session_id": "s", "hook_event_name": "PostToolUse", "permission_mode": "default",
              "transcript_path": "/home/u/.claude/projects/-work-repo/s.jsonl"}
    assert isinstance(hosts.detect(claude, {"FACTORY_PROJECT_DIR": "/w",
                                            "FACTORY_SESSION_ID": "d"}), ClaudeCode)
    for mode in ("plan", "acceptEdits", "bypassPermissions"):
        assert isinstance(hosts.detect(dict(claude, permission_mode=mode)), ClaudeCode)


def test_a_te_command_in_a_droid_shell_is_about_droid():
    assert hosts.default_host({"FACTORY_SESSION_ID": "d"}) == "droid"
    assert hosts.default_host({"FACTORY_SESSION_ID": "d", "CLAUDECODE": "1"}) == "claude"
    assert hosts.default_host({"FACTORY_SESSION_ID": "d", "TE_HOST": "codex"}) == "codex"
    assert Droid().env_session_id({"FACTORY_SESSION_ID": "d"}) == "d"
    assert Droid().env_session_id({"FACTORY_SESSION_ID": "d", "TE_SESSION_ID": "t"}) == "t"


# ---- payload translation ------------------------------------------------------------------
def test_the_turn_is_the_prompts_message_and_the_project_dir_is_a_fallback_cwd():
    host = Droid()
    prompt = payload("UserPromptSubmit", "/w", "/t", prompt="go", message_id="m1")
    assert host.turn_key(prompt) == "m1"
    assert host.turn_key(payload("PostToolUse", "/w", "/t", message_id="m2")) == ""
    assert host.turn_key(payload("UserPromptSubmit", "/w", "/t")) == ""
    assert host.cwd({"session_id": "s"}, {"FACTORY_PROJECT_DIR": "/proj"}) == "/proj"
    assert host.cwd({"cwd": "/w"}, {"FACTORY_PROJECT_DIR": "/proj"}) == "/w"
    assert host.session_id(prompt) == SID and host.transcript(prompt) == "/t"


def test_droid_tools_are_named_as_the_mechanisms_know_them():
    host = Droid()
    name, args, text, code = host.tool_call(payload(
        "PostToolUse", "/w", "/t", tool_name="Execute",
        tool_input={"command": "pytest -q", "riskLevel": "low"},
        tool_response="1 failed\n\n[Process exited with code 1]"))
    assert (name, args, code) == ("Bash", {"command": "pytest -q"}, 1) and "1 failed" in text
    assert host.tool_call(payload("PostToolUse", "/w", "/t", tool_name="Execute",
                                  tool_input={"command": "ls"}, tool_response="a\n"))[3] is None
    assert host.tool_call(payload("PostToolUse", "/w", "/t", tool_name="Create",
                                  tool_input={"file_path": "/w/a.py", "content": "x"},
                                  tool_response="ok"))[:2] == ("Write", {"file_path": "/w/a.py"})
    name, args, text, _ = host.tool_call(payload("PostToolUse", "/w", "/t", tool_name="LS",
                                                 tool_input={"directory_path": "/w"},
                                                 tool_response="a.py\nb.py"))
    assert (name, args, text) == ("LS", {"path": "/w"}, "a.py\nb.py")


def test_a_droid_exec_run_is_headless_and_the_terminal_ui_is_not(monkeypatch):
    for args, headless in ((["/usr/bin/droid exec --cwd /w say hi"], True),
                           (["sh -c x", "droid exec -o json hi"], True),
                           (["/home/u/.local/bin/droid"], False),
                           (["droid --resume"], False), (["droidexec run"], False)):
        monkeypatch.setattr(hosts, "_ancestor_args", lambda a=args: a)
        assert Droid().headless({}) is headless, args


def test_hook_rows_and_hook_warnings_in_the_log_are_not_prompts(tmp_path):
    hook_row = {"type": "message", "id": "h1", "timestamp": "2026-11-16T15:27:58.446Z",
                "message": {"role": "user", "content": [], "visibility": "user_only",
                            "hookEventName": "SessionStart", "hookResults": []}}
    warning = {"type": "message", "id": "h2", "timestamp": "2026-11-16T15:27:58.450Z",
               "message": {"role": "user", "systemNotice": True, "visibility": "user_only",
                           "content": [{"type": "text", "text": "[Hook Warning] x"}]}}
    path = write_droid_log(sessions(tmp_path), "/work/w", "s1", [hook_row, warning] + ENTRIES)
    assert trace.droid_prompt_count(path) == 1
    out = str(tmp_path / "t.jsonl")
    trace.droid_session_to_claude(path, out)
    assert trace.claude_prompt_count(out) == 1


def test_sessions_live_under_the_home_droid_was_given():
    assert trace.droid_sessions_dir({"FACTORY_HOME_OVERRIDE": "/o"}) == "/o/.factory/sessions"
    assert trace.droid_sessions_dir({"FACTORY_HOME": "/f", "FACTORY_HOME_OVERRIDE": "/o"}) == \
        "/f/sessions"


# ---- injection ----------------------------------------------------------------------------
def test_the_start_context_goes_back_at_session_start_in_droids_shape(env, repo, tmp_path):
    store(env, repo).write("injection.json", {"rendered": FACTS, "rendered_ids": ["a", "b"]})
    tp = transcript(tmp_path, repo)
    out = call(env, payload("SessionStart", repo, tp, source="startup"))
    assert out["hookSpecificOutput"] == {"hookEventName": "SessionStart",
                                         "additionalContext": FACTS}
    assert store(env, repo).read(f"sessions/{SID}.json")["injected_chars"] == len(FACTS)
    assert call(env, payload("UserPromptSubmit", repo, tp, prompt="go", message_id="m1")) == {}


# ---- the live fold ------------------------------------------------------------------------
def test_a_finished_droid_session_is_folded_once_at_session_end(env, repo, tmp_path):
    tp = transcript(tmp_path, repo)
    entries = [user("m1", [{"type": "text", "text": "fix the tests"}], "2026-11-16T10:00:00.000Z"),
               *ENTRIES[2:]]
    call(env, payload("SessionStart", repo, tp, source="startup"))
    call(env, payload("UserPromptSubmit", repo, tp, prompt="fix the tests", message_id="m1"))
    write_droid_log(sessions(tmp_path), repo, SID, entries)
    for cmd, resp in (("pytest -q", "bash: pytest: command not found\n"
                                    "[Process exited with code 127]"),
                      ("python -m pytest -q", "3 passed")):
        call(env, payload("PostToolUse", repo, tp, tool_name="Execute",
                          tool_input={"command": cmd}, tool_response=resp))
    call(env, payload("Stop", repo, tp, stop_hook_active=False))
    call(env, payload("SessionEnd", repo, tp, reason="other"))
    st = store(env, repo)
    assert SID in (st.read("folded.json") or {})
    ledger = st.read_lines("ledger.jsonl")
    assert len(ledger) == 1 and ledger[0]["session_id"] == SID
    call(env, payload("SessionEnd", repo, tp, reason="other"))
    assert len(st.read_lines("ledger.jsonl")) == 1
    assert not os.path.isdir(repo_state_dir(repo_identity(repo, env), "claude", env)) or \
        SID not in (Store(repo_state_dir(repo_identity(repo, env), "claude", env))
                    .read("folded.json") or {})


def test_the_fold_reads_the_droid_log_as_calls(tmp_path):
    path = write_droid_log(sessions(tmp_path), "/work/w", "s1", ENTRIES + [
        assistant("a4", [{"type": "tool_use", "id": "toolu_3", "name": "Create",
                          "input": {"file_path": "/work/w/x.py", "content": "x"}}],
                  "2026-11-03T11:55:00.000Z")])
    out = Droid().mechanism_trace(path, str(tmp_path))
    assert out and [c[0] for c in trace.load_trajectory(out)] == ["Bash", "Bash", "Write"]
    assert Droid().prompt_count(path) == 1
    assert Droid().session_usage(path)["input_tokens"] is None
    assert Droid().mechanism_trace(str(tmp_path / "none.jsonl"), str(tmp_path)) is None


# ---- side calls ---------------------------------------------------------------------------
def test_droid_gets_no_side_call_and_never_falls_back_to_claude(monkeypatch):
    monkeypatch.setattr(sidecall, "claude_call", lambda *a, **k: pytest.fail("no Claude call"))
    config = type("C", (), {"paid_model": "haiku"})()
    assert "droid" in sidecall.NO_SIDE_CALLS
    assert sidecall.runner_for("droid", config) is None
    assert sidecall.call_name("droid") == "droid exec"


# ---- packaging ----------------------------------------------------------------------------
def _json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def test_the_droid_manifest_and_marketplace_name_the_one_plugin():
    droid = _json(PLUGIN, ".factory-plugin", "plugin.json")
    claude = _json(PLUGIN, ".claude-plugin", "plugin.json")
    assert droid["name"] == claude["name"] == "tokeneconomizer"
    assert droid["version"] == claude["version"]
    # A manifest `hooks` path is read beside hooks/hooks.json, not instead of it (1.12.1): one
    # would run every hook twice.
    assert "hooks" not in droid
    market = _json(REPO, ".factory-plugin", "marketplace.json")
    assert [(p["name"], p["source"]) for p in market["plugins"]] == [("tokeneconomizer",
                                                                       "./plugin")]
    assert market["name"] == _json(REPO, ".claude-plugin", "marketplace.json")["name"]


def test_the_hook_file_droid_runs_uses_events_and_a_root_it_knows():
    doc = _json(PLUGIN, "hooks", "hooks.json")
    # Droid warns about PostToolUseFailure, which it lacks, and runs the rest.
    assert set(doc["hooks"]) - {"PostToolUseFailure"} <= DROID_EVENTS
    assert {"SessionStart", "UserPromptSubmit", "PostToolUse", "SessionEnd"} <= set(doc["hooks"])
    for groups in doc["hooks"].values():
        for g in groups:
            for h in g["hooks"]:
                assert h["command"].startswith('"${CLAUDE_PLUGIN_ROOT}/scripts/te-hook"')
                assert "--host" not in h["command"] and h["timeout"] <= 60


def _bootstrap():
    sys.path.insert(0, os.path.join(PLUGIN, "scripts"))
    try:
        import te_bootstrap
    finally:
        sys.path.pop(0)
    return te_bootstrap


def test_a_droid_install_copy_finds_the_core_in_the_marketplace_it_came_from(tmp_path):
    """Droid's install copies the plugin directory without its symlinks, into
    `plugins/cache/<marketplace>-<hash>/<plugin>-<hash>/<version>`, and records the marketplace
    in `plugins/known_marketplaces/<marketplace, lower case>-<hash>.json` (1.12.1)."""
    te_bootstrap = _bootstrap()
    home = tmp_path / "fhome" / ".factory"
    copy = (home / "plugins" / "cache" / "TokenEconomizer-b5c3b9490b82" /
            "tokeneconomizer-7ab58e7646fc" / "6e83dcea10eb")
    copy.mkdir(parents=True)
    known = home / "plugins" / "known_marketplaces"
    known.mkdir()
    (known / "other-0123456789ab.json").write_text(json.dumps(
        {"name": "other", "entry": {"source": {"source": "local", "path": "/nowhere"}}}))
    (known / "tokeneconomizer-b5c3b9490b82.json").write_text(json.dumps(
        {"name": "TokenEconomizer", "entry": {"source": {"source": "local", "path": REPO},
                                              "installLocation": REPO}}))
    found = te_bootstrap._marketplace_sources(str(copy))
    assert REPO in found and "/nowhere" not in found
    assert te_bootstrap._droid_marketplace_sources(str(home), "TokenEconomizer-b5c3b9490b82") \
        == [REPO]
    shutil.rmtree(str(known))
    assert te_bootstrap._droid_marketplace_sources(str(home), "TokenEconomizer-b5c3b9490b82") \
        == []
