"""A known answer for the environment facts other agents' history teaches.

The corpus is synthetic and written in each agent's own on-disk format (Claude Code's project
log, Codex's rollout, Factory Droid's session file), then read the way the plugin reads it: the
history sources, the real fold, the session-start hook. One repository meets the same missing
test runner in four sessions across three agents and repairs it the same way; two scripted
one-prompt runs there say the same and must not count; another repository's sessions run their
tests fine. Later, two sessions run the bare runner successfully, and the fact must go.

The corpus comes in two spellings of the same commands: as plain shell commands, and as this
fleet's agents type them, every command wrapped in Nushell (`bash -lc "nu -l -c '<cmd>'"` in
Codex's argv, Nu's own error text for a missing command).

What a fact needs (`microskills.lifecycle`, `microskills.envfacts`): FormRepair is found when a
bare form fails with an environment error and a later form of the same intent works as typed in
the same session; it is promoted after K_PROMOTE (3) confirming sessions and injected only once
FIRE_MIN_N (4) sessions exercised it with a firing rate of at least FIRE_GATE (40%). A
contradiction -- the bare form runs the tests -- counts only on CONTRADICT_STREAK (2) consecutive
sessions, and then outweighs the confirmations (LAMBDA 5) and retires the fact.
"""
import json
import os
import time
import uuid

import pytest

from test_history_import import CLAIM, history_log
from test_live_hooks import store_for
from tokeneconomizer import trace
from tokeneconomizer.microskills import lifecycle, observe
from tokeneconomizer.microskills.envfacts import CONTRADICT_STREAK
from tokeneconomizer.plugin import history, hook, hosts
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.trace import sources

HOSTS = {"claude": hosts.ClaudeCode, "codex": hosts.Codex}


def nu_missing(cmd):
    """What Nu 0.116 prints when an external command is not there."""
    return (f"Error: nu::shell::external_command\n\n  x External command failed\n"
            f"   ,-[source:1:1]\n 1 | {cmd} -q\n   : ^^^|^^\n"
            f"   :    `-- Command `{cmd}` not found\n   `----\n"
            f"  help: `{cmd}` is neither a Nushell built-in or a known external command\n")


# (command, output, exit code), per spelling
MISSING = {"posix": [("pytest -q", "bash: pytest: command not found", 127),
                     ("python -m pytest -q", "3 passed in 0.2s", 0)],
           "nu": [("nu -l -c 'pytest -q'", nu_missing("pytest"), 1),
                  ("nu -l -c 'python -m pytest -q'", "3 passed in 0.2s", 0)]}
FIXED = {"posix": [("pytest -q", "5 passed in 0.3s", 0)],
         "nu": [("nu -l -c 'pytest -q'", "5 passed in 0.3s", 0)]}
FINE = {"posix": [("pytest -q", "2 passed in 0.1s", 0)],
        "nu": [("nu -l -c 'pytest -q'", "2 passed in 0.1s", 0)]}


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000Z"


def _age(path, age_s):
    end = time.time() - age_s
    os.utime(path, (end, end))
    return str(path)


# ---- each agent's own format ----------------------------------------------------------------
def claude_log(home, cwd, sid, calls, age_s, entrypoint="cli"):
    """Claude Code: a failed Bash call's result starts `Exit code N` and is an error."""
    return history_log(home / ".claude", cwd, sid, [("fix the tests", [
        ("Bash", {"command": cmd}, f"Exit code {code}\n{out}" if code else out, bool(code))
        for cmd, out, code in calls])], age_s=age_s, entrypoint=entrypoint)


def codex_log(home, cwd, calls, age_s, source="cli"):
    """Codex 0.163: a command is the argv `[bash, -lc, <command>]`, its result an exit code."""
    sid = str(uuid.uuid4())
    d = home / ".codex" / "sessions" / "2026" / "11" / "16"
    d.mkdir(parents=True, exist_ok=True)
    t = time.time() - age_s - 300
    lines = [{"timestamp": _iso(t), "type": "session_meta",
              "payload": {"id": sid, "timestamp": _iso(t), "cwd": cwd, "originator": "codex-tui",
                          "cli_version": "0.163.0", "source": source}},
             {"timestamp": _iso(t), "type": "event_msg", "payload": {"type": "task_started"}},
             {"timestamp": _iso(t), "type": "response_item",
              "payload": {"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": "fix the tests"}]}}]
    for n, (cmd, out, code) in enumerate(calls):
        lines.append({"timestamp": _iso(t), "type": "event_msg", "payload": {
            "type": "item_completed", "item": {
                "id": f"item_{n}", "type": "CommandExecution", "command": ["bash", "-lc", cmd],
                "aggregated_output": out, "exit_code": code,
                "status": "failed" if code else "completed"}}})
    lines.append({"timestamp": _iso(t), "type": "event_msg",
                  "payload": {"type": "task_complete", "last_agent_message": "done"}})
    path = d / f"rollout-2026-11-16T10-00-00-{sid}.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(path, age_s)


def droid_log(home, cwd, sid, calls, age_s):
    """Factory Droid: an `Execute` call; a failed one's result is an error naming its exit code."""
    d = home / ".factory" / "sessions" / trace.claude_project_folder(cwd)
    d.mkdir(parents=True, exist_ok=True)
    t = time.time() - age_s - 300
    lines = [{"type": "session_start", "id": sid, "title": "s", "owner": "u", "version": 2,
              "cwd": cwd},
             {"type": "message", "id": "p1", "timestamp": _iso(t),
              "message": {"role": "user", "content": [{"type": "text", "text": "fix the tests"}]}}]
    for n, (cmd, out, code) in enumerate(calls):
        lines.append({"type": "message", "id": f"a{n}", "timestamp": _iso(t), "message": {
            "role": "assistant", "modelId": "m", "content": [
                {"type": "tool_use", "id": f"tu{n}", "name": "Execute",
                 "input": {"command": cmd, "riskLevel": "low"}}]}})
        result = {"type": "tool_result", "tool_use_id": f"tu{n}",
                  "content": f"{out}\n\nExit code {code}" if code else out}
        if code:
            result["is_error"] = True
        lines.append({"type": "message", "id": f"r{n}", "timestamp": _iso(t),
                      "message": {"role": "user", "content": [result]}})
    path = d / f"{sid}.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return _age(path, age_s)


# ---- the setting ----------------------------------------------------------------------------
@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    for d in (".claude/projects", ".codex/sessions", ".factory/sessions"):
        (h / d).mkdir(parents=True)
    return h


@pytest.fixture
def env(tmp_path, home):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "te"), "TE_FOLD_SYNC": "1",
            "TE_HISTORY_IMPORT": "on", "TE_TOOLPRUNE_QUIET_MINUTES": "0",
            "TE_MICROSKILLS": "envfacts", "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"), "CODEX_HOME": str(home / ".codex"),
            "FACTORY_HOME": str(home / ".factory"),
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


def _repo(tmp_path, name):
    r = tmp_path / name
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = https://github.com/acme/{name}.git\n')
    (r / "pyproject.toml").write_text(f"[project]\nname='{name}'\n")
    return str(r)


def start(env, repo, host, sid, tmp_path):
    t = str(tmp_path / f"{host}-{sid}.jsonl")
    open(t, "w").close()
    out = hook.run({"hook_event_name": "SessionStart", "session_id": sid, "source": "startup",
                    "transcript_path": t, "cwd": repo}, env, host_hint=host)
    return (out.get("hookSpecificOutput") or {}).get("additionalContext") or ""


def form_repair_facts(env, repo, host):
    st = store_for(env, repo, host=host)
    facts = (st.read("injection.json") or {}).get("facts") or {}
    return {k: v for k, v in facts.items() if not k.startswith("_") and
            v.get("family") == "form_repair"}


def envfacts_state(env, repo, host):
    st = store_for(env, repo, host=host)
    d = os.path.join(st.base, "mechanisms", "envfacts_state")
    (name,) = os.listdir(d)
    with open(os.path.join(d, name)) as fh:
        return json.load(fh)


# ---- each layer on its own: reader output, then the observer ------------------------------
@pytest.mark.parametrize("shape", ["posix", "nu"])
@pytest.mark.parametrize("agent", ["claude", "codex", "droid"])
def test_each_agents_log_reaches_the_observer_with_its_failure_and_its_repair(
        agent, shape, home, tmp_path):
    repo = _repo(tmp_path, "widget")
    if agent == "claude":
        claude_log(home, repo, "s1", MISSING[shape], 3600)
    elif agent == "codex":
        codex_log(home, repo, MISSING[shape], 3600)
    else:
        droid_log(home, repo, "s1", MISSING[shape], 3600)
    src = sources.SOURCES[agent]
    (log,) = src.logs({"HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude")})
    work = tmp_path / "work"
    work.mkdir()
    tr = src.to_trace(src.materialize(log["path"], str(work)), str(work))
    shell = [c for c in trace.load_trajectory(tr) if c[0].lower() in ("bash", "shell")]
    assert [ok for *_, ok in shell] == [False, True]           # the reader keeps both results
    assert "not found" in shell[0][2] and "3 passed" in shell[1][2]
    obs = observe.observe_task(tr, observe.GENERIC)["obs"]
    assert [(o.intent, o.form, o.outcome) for o in obs] == [
        ("run_tests", "pytest", "command_missing"), ("run_tests", "python -m pytest", "ran")]
    assert obs[0].not_found and obs[1].tests_ran


# ---- the whole chain --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["posix", "nu"])
@pytest.mark.parametrize("receiver", ["codex", "claude"])
def test_a_fact_three_agents_found_reaches_another_agent_only_where_it_holds_and_goes_when_it_stops(
        receiver, shape, env, home, tmp_path):
    widget, gadget = _repo(tmp_path, "widget"), _repo(tmp_path, "gadget")
    h = 3600
    # The repository's sessions: the same missing runner, met and repaired by three agents.
    claude_log(home, widget, "c1", MISSING[shape], 10 * h)
    codex_log(home, widget, MISSING[shape], 9 * h)
    droid_log(home, widget, "d1", MISSING[shape], 8 * h)
    codex_log(home, widget, MISSING[shape], 7 * h)
    # Excluded: one-prompt scripted runs (`codex exec`, `claude -p`) that say the same.
    codex_log(home, widget, MISSING[shape], 8.5 * h, source="exec")
    claude_log(home, widget, "c-sdk", MISSING[shape], 7.5 * h, entrypoint="sdk-cli")
    # Another repository, whose runner works.
    claude_log(home, gadget, "g1", FINE[shape], 10 * h)
    codex_log(home, gadget, FINE[shape], 9 * h)
    droid_log(home, gadget, "g2", FINE[shape], 8 * h)
    codex_log(home, gadget, FINE[shape], 7 * h)

    assert start(env, widget, receiver, "live-w1", tmp_path) == ""   # imports; says it next time
    doc = store_for(env, widget, host=receiver).read("history.json")
    assert doc["state"] == "done" and doc["imported"] == 4
    assert doc["origins"] == {"claude": 1, "codex": 2, "droid": 1}
    assert doc["skipped"].get("headless") == 2
    assert CLAIM in start(env, widget, receiver, "live-w2", tmp_path)
    (fact,) = form_repair_facts(env, widget, receiver).values()
    assert fact["state"] == "ACTIVE" and fact["fire"] == "4/4" and fact["inject"]
    life = next(iter(envfacts_state(env, widget, receiver)["facts"].values()))["life"]
    assert len(life["confirms"]) == 4 == lifecycle.FIRE_MIN_N and life["contradicts"] == []

    start(env, gadget, receiver, "live-g1", tmp_path)
    assert start(env, gadget, receiver, "live-g2", tmp_path) == ""
    assert form_repair_facts(env, gadget, receiver) == {}
    assert store_for(env, gadget, host=receiver).read("injection.json")["rendered"] == ""
    assert store_for(env, gadget, host=receiver).read("history.json")["imported"] == 4

    # The runner works now. One session saying so is not yet a fixed environment ...
    codex_log(home, widget, FIXED[shape], 2 * h)
    ctx = LiveContext.build(HOSTS[receiver](), widget, env)
    assert history.run(ctx)["read"] == 1
    assert CLAIM in store_for(env, widget, host=receiver).read("injection.json")["rendered"]
    # ... CONTRADICT_STREAK in a row are, and the fact is retired.
    assert CONTRADICT_STREAK == 2
    droid_log(home, widget, "d2", FIXED[shape], 1 * h)
    assert history.run(ctx)["read"] == 1
    (fact,) = form_repair_facts(env, widget, receiver).values()
    assert fact["state"] == "RETIRED" and not fact["inject"]
    assert store_for(env, widget, host=receiver).read("injection.json")["rendered"] == ""
    assert CLAIM not in start(env, widget, receiver, "live-w3", tmp_path)


def test_without_the_excluded_runs_three_sessions_are_not_enough(env, home, tmp_path):
    """The scripted runs are what would make a fourth session: left out, the fact waits."""
    widget = _repo(tmp_path, "widget")
    claude_log(home, widget, "c1", MISSING["nu"], 10 * 3600)
    codex_log(home, widget, MISSING["nu"], 9 * 3600)
    droid_log(home, widget, "d1", MISSING["nu"], 8 * 3600)
    codex_log(home, widget, MISSING["nu"], 7 * 3600, source="exec")
    start(env, widget, "codex", "w1", tmp_path)
    assert CLAIM not in start(env, widget, "codex", "w2", tmp_path)
    (fact,) = form_repair_facts(env, widget, "codex").values()
    assert fact["state"] == "ACTIVE" and fact["fire"] == "3/3" and not fact["inject"]
