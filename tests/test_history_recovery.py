"""Imports that stop before they finish, and imports that run twice or at once.

An import's process can die anywhere between two of its writes (a signal, the machine sleeping,
the agent that started it tearing down its process group). On himalayas, seven stores were left
`running` that way, their heartbeats days old and no process behind them, each in a temporary
directory no session would start in again. These cases kill a real worker process at each
place it saves, then check that what it left is reported as stalled, that the next session start
(or `te history import --all`) resumes it from that place, that no session is folded twice, and
that two starters, two workers or two agents at once still give one import per store.

The sessions are Claude Code logs read through the real history reader (`history_log`).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import pytest

from test_history_import import (claude_home, ctx_for, detached, env, history_log,  # noqa: F401
                                 ledger_ids, live_session, repo, start_hook, trap)
from test_live_hooks import store_for, trap_session
from tokeneconomizer.plugin import cli, fold, history
from tokeneconomizer.plugin.context import LiveContext

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = [f"old{i}" for i in range(4)]

# The seven `history.json` documents himalayas held, as read on 2026-11-16 at 15:29:16 UTC
# (`OBSERVED`): written by the worker's first save, before it planned; no pid, no plan.
OBSERVED = datetime(2026, 11, 16, 15, 29, 16, 457696, tzinfo=timezone.utc).timestamp()
HIMALAYAS = [(1794392226.889861, 1794392226.747744), (1794392235.657778, 1794392235.5638778),
             (1794392255.6014369, 1794392254.5339289), (1794392233.038864, 1794392232.838592),
             (1794392242.041138, 1794392241.8978171), (1794392229.185299, 1794392229.1047418),
             (1794392231.5830011, 1794392231.494582)]

# A worker, `te history import --auto` as a session start spawns it, that kills itself with
# SIGKILL at a checkpoint: nothing it could catch, no traceback, its pid gone.
CHILD = r"""
import os, signal, sys
from tokeneconomizer.plugin import cli, fold, history
at, n = sys.argv[1], int(sys.argv[2])
seen = [0]

def dies(real, before=True):
    def wrapped(*a, **k):
        seen[0] += 1
        if before and seen[0] == n:
            os.kill(os.getpid(), signal.SIGKILL)
        out = real(*a, **k)
        if not before and seen[0] == n:
            os.kill(os.getpid(), signal.SIGKILL)
        return out
    return wrapped

if at == "planning":
    history.plan = dies(history.plan)
elif at == "planned":
    history._import_one = dies(history._import_one)
elif at == "folding":
    fold.fold_locked = dies(fold.fold_locked)
elif at == "folded":
    fold.fold_locked = dies(fold.fold_locked, before=False)
elif at == "finished":
    fold.pending_sessions = dies(fold.pending_sessions)
sys.exit(cli.main(sys.argv[3:]))
"""

needs_kill = pytest.mark.skipif(os.name == "nt", reason="SIGKILL and pid probes are POSIX")


def child_env(env):
    return dict(detached(env), HOME=os.environ["HOME"], PYTHONPATH=_REPO, PYTHONUTF8="1")


def killing_spawn(env, at, n, ran):
    """`LiveContext.spawn` for one worker that dies at `at` (the `n`th call there); any later
    spawn is only recorded."""
    def spawn(self, argv, log):
        if ran:
            ran.append(("later", argv))
            return True
        cmd = [sys.executable, "-c", CHILD, at, str(n), *argv, "--host", self.host.name,
               "--cwd", self.repo.root]
        p = subprocess.run(cmd, env=child_env(env), cwd=self.repo.root, capture_output=True,
                           text=True, timeout=120)
        ran.append(p)
        return True
    return spawn


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", ""])
    p.wait()
    return p.pid


def corpus(claude_home, repo):
    for i, sid in enumerate(OLD):
        history_log(claude_home, repo, sid, trap(repo, i), age_s=3600 * (10 - i))


def folded_keys(store):
    return sorted(store.read("folded.json") or {})


# ---- a worker killed at each checkpoint ------------------------------------------------------
CHECKPOINTS = [
    # (where, n, what it left: next, sessions folded before it died)
    ("planning", 1, None, 0),       # first save written, plan not yet (himalayas)
    ("planned", 1, 0, 0),           # plan saved, nothing folded
    ("folding", 2, 1, 1),           # `attempt` saved for old1, its fold not done
    ("folded", 2, 1, 2),            # old1 folded, its place not saved
    ("planned", 4, 3, 3),           # three places saved
    ("finished", 1, None, 4),       # plan done; the pending live session not yet folded
]


@needs_kill
@pytest.mark.parametrize("at,n,next_,folded_before", CHECKPOINTS,
                         ids=[f"{c[0]}-{c[1]}" for c in CHECKPOINTS])
def test_a_worker_killed_at_each_checkpoint_is_reported_and_resumed_exactly_once(
        env, repo, claude_home, tmp_path, monkeypatch, capsys, at, n, next_, folded_before):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    live_session(ctx, "live0", trap_session(str(tmp_path / "live0.jsonl"), repo, 5))
    ran = []
    monkeypatch.setattr(LiveContext, "spawn", killing_spawn(env, at, n, ran))
    history.maybe_start(ctx)                              # the session start that spawns it
    assert len(ran) == 1 and ran[0].returncode == -signal.SIGKILL, ran[0].stderr

    doc = ctx.store.read("history.json")
    assert len(ledger_ids(ctx.store)) == folded_before
    if at == "finished":
        assert doc["state"] == "done" and not fold.import_blocks(ctx)
        assert fold.pending_sessions(ctx) == ["live0"]
    else:
        assert doc["state"] == "running" and isinstance(doc["pid"], int)
        assert doc["host"] == history._this_host() and doc.get("next") == next_
        assert history._pid_alive(doc["pid"]) is False                # the worker is gone
        why = history.stalled(doc)
        assert why == f"its process, pid {doc['pid']}, is gone"       # at once, not in 15 min
        s = history.summary(ctx)
        assert s["status"] == "stalled" and s["stalled"] == why
        assert history.describe(s).startswith("stalled: the import stopped before it finished")
        assert fold.import_blocks(ctx) and fold.fold_session(ctx, "live0") is None

    # The next session start resumes it from its place (inline here: TE_FOLD_SYNC=1).
    start_hook(env, repo, "next", tmp_path)
    doc = ctx.store.read("history.json")
    assert doc["state"] == "done" and doc["imported"] == 4 and "attempt" not in doc
    assert ledger_ids(ctx.store) == OLD + ["live0"]                  # each exactly once
    assert folded_keys(ctx.store) == sorted(OLD + ["live0"])
    assert len(ctx.store.read("envelope.json")["resolved_turns"]) == 4
    assert history.summary(ctx)["status"] == "done"
    assert all(r[0] == "later" for r in ran[1:])                     # nothing else spawned
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "status"]) == 0
    assert "read 4 earlier session(s) from Claude Code's history" in capsys.readouterr().out


def test_a_claim_whose_worker_never_started_waits_for_its_heartbeat(env, repo, claude_home,
                                                                    tmp_path, monkeypatch):
    """The session start wrote `running` and its spawn never ran: no pid to probe, so it
    counts as running until its heartbeat goes quiet, and the start after that resumes it."""
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    history.maybe_start(ctx)
    doc = ctx.store.read("history.json")
    assert doc["state"] == "running" and "pid" not in doc
    assert history.summary(ctx)["status"] == "running"
    history.maybe_start(ctx)
    assert len(spawned) == 1
    ctx.store.write("history.json", dict(doc, heartbeat=doc["heartbeat"]
                                         - history.HEARTBEAT_STALE_S - 1))
    assert history.summary(ctx)["status"] == "stalled"
    assert history.summary(ctx)["stalled"].startswith("no heartbeat for 15 min")
    start_hook(env, repo, "next", tmp_path)
    assert ctx.store.read("history.json")["state"] == "done"
    assert ledger_ids(ctx.store) == OLD


# ---- the himalayas documents -----------------------------------------------------------------
def test_the_himalayas_orphans_read_as_stalled_and_are_resumed_or_settled(
        env, repo, tmp_path, monkeypatch, capsys):
    """Seven claims left `running` in temporary directories, each with the session that
    started it pending. One directory still exists: a session start there resumes the import.
    The six that are gone: `te history import --all` marks them stopped, since no session
    start can come; the person's real repository is not touched."""
    shift = time.time() - OBSERVED
    dirs = []
    for i, (beat, started) in enumerate(HIMALAYAS):
        d = tmp_path / f"probe{i}"
        d.mkdir()
        ctx = ctx_for(detached(env), str(d))
        assert ctx.repo.how == "dir"
        ctx.store.write("history.json", {"heartbeat": beat + shift, "noticed": False,
                                         "started": started + shift, "state": "running"})
        live_session(ctx, f"probe{i}", trap_session(str(tmp_path / f"p{i}.jsonl"), str(d), i))
        dirs.append((str(d), ctx))
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    d0, c0 = dirs[0]
    s = history.summary(c0)
    assert s["status"] == "stalled" and s["stalled"] == "no heartbeat for 5 days"
    assert "none planned yet" in history.describe(s)
    assert fold.import_blocks(c0)
    cli.main(["--cwd", d0, "--host", "claude", "status"])
    assert "  history      stalled: the import stopped before it finished" in (
        capsys.readouterr().out)
    cli.main(["--cwd", repo, "--host", "claude", "repos"])
    assert capsys.readouterr().out.count("claude: its import of the agents' history stalled") == 7

    start_hook(env, d0, "next", tmp_path)                 # the directory that still exists
    assert c0.store.read("history.json")["state"] == "done"
    assert ledger_ids(c0.store) == ["probe0"]

    for d, _ in dirs[1:]:
        shutil.rmtree(d)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import", "--all",
                     "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("it would be marked stopped") == 6
    assert "6 stalled import(s) of repositories no log names would be marked stopped" in out
    assert all(c.store.read("history.json")["state"] == "running" for _, c in dirs[1:])
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import", "--all"]) == 0
    out = capsys.readouterr().out
    assert "6 stalled import(s) of repositories no log names marked stopped" in out
    for _, c in dirs[1:]:
        doc = c.store.read("history.json")
        assert doc["state"] == "stopped" and not fold.import_blocks(c)
        assert doc["reason"].startswith("the import stopped before it finished (no heartbeat")
        assert history.summary(c)["status"] == "stopped"
    assert c0.store.read("history.json")["state"] == "done"
    assert store_for(env, repo).read("history.json") is None
    assert history.settle_orphans(env["TOKENECONOMIZER_HOME"]) == 0


# ---- `te history import --all` ---------------------------------------------------------------
@needs_kill
def test_te_history_import_all_detach_resumes_a_dead_import_from_its_place(
        env, repo, claude_home, tmp_path, monkeypatch, capsys):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    ran = []
    monkeypatch.setattr(LiveContext, "spawn", killing_spawn(env, "planned", 3, ran))
    history.maybe_start(ctx)
    assert ran[0].returncode == -signal.SIGKILL
    assert ctx.store.read("history.json")["next"] == 2 and ledger_ids(ctx.store) == OLD[:2]

    detached_runs = []

    def spawn_inline(self, argv, log):
        detached_runs.append(argv)
        return cli.main([*argv, "--host", self.host.name, "--cwd", self.repo.root])
    monkeypatch.setattr(LiveContext, "spawn", spawn_inline)
    for k, v in detached(env).items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import", "--all",
                     "--detach"]) == 0
    assert detached_runs == [["history", "import", "--all", "--auto"]]
    out = capsys.readouterr().out
    assert "acme/widget [claude]: done, 2 session(s) read" in out
    assert ledger_ids(ctx.store) == OLD and ctx.store.read("history.json")["imported"] == 4

    # A run that is alive is left to it.
    ctx.store.write("history.json", dict(ctx.store.read("history.json"), state="running",
                                         heartbeat=time.time(), pid=os.getpid(),
                                         host=history._this_host()))
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import", "--all"]) == 0
    assert "an import is running; left to it" in capsys.readouterr().out
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 1
    assert "is running" in capsys.readouterr().out
    # The same doc with its process gone: taken over by hand, nothing folded twice.
    ctx.store.write("history.json", dict(ctx.store.read("history.json"), pid=dead_pid()))
    assert cli.main(["--cwd", repo, "--host", "claude", "history", "import"]) == 0
    assert "imported 0 earlier session(s)" in capsys.readouterr().out
    assert ledger_ids(ctx.store) == OLD


# ---- the same corpus twice -------------------------------------------------------------------
def test_importing_the_same_corpus_twice_folds_each_session_once(env, repo, claude_home,
                                                                 tmp_path, monkeypatch, capsys):
    corpus(claude_home, repo)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    args = ["--cwd", repo, "--host", "claude", "history", "import"]
    assert cli.main(args) == 0
    assert "imported 4 earlier session(s)" in capsys.readouterr().out
    st = store_for(env, repo)
    first = (ledger_ids(st), folded_keys(st), st.read("injection.json")["sessions"])
    assert first[0] == OLD
    assert cli.main(args) == 0
    assert "imported 0 earlier session(s)" in capsys.readouterr().out
    for _ in range(2):
        assert cli.main(args[:-2] + ["history", "import", "--all"]) == 0
        assert ", 0 session(s) read" in capsys.readouterr().out
    # A session start long after: its catch-up finds nothing new.
    doc = st.read("history.json")
    st.write("history.json", dict(doc, caught_up=time.time() - history.CATCHUP_EVERY_S - 1))
    start_hook(env, repo, "later", tmp_path)
    assert (ledger_ids(st), folded_keys(st), st.read("injection.json")["sessions"]) == first
    doc = st.read("history.json")
    assert doc["state"] == "done" and doc["imported"] == 4
    assert doc["skipped"].get("already_folded") == 4


# ---- at once ---------------------------------------------------------------------------------
@pytest.mark.parametrize("left", ["nothing", "partial", "dead"])
def test_concurrent_session_starts_start_one_worker(env, repo, claude_home, monkeypatch, left):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    if left == "partial":
        ctx.store.write("history.json", {"state": "partial", "plan": [], "next": 0,
                                         "heartbeat": time.time() - 30})
    elif left == "dead":
        ctx.store.write("history.json", {"state": "running", "heartbeat": time.time(),
                                         "pid": dead_pid(), "host": history._this_host()})
    spawned, lock = [], threading.Lock()

    def spawn(self, argv, log):
        with lock:
            spawned.append(argv)
        return True
    monkeypatch.setattr(LiveContext, "spawn", spawn)
    starters = [ctx_for(detached(env), repo) for _ in range(6)]
    gate = threading.Barrier(len(starters))

    def start(c):
        gate.wait()
        history.maybe_start(c)
    threads = [threading.Thread(target=start, args=(c,)) for c in starters]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert spawned == [["history", "import", "--auto"]]
    doc = ctx.store.read("history.json")
    assert doc["state"] == "running" and "pid" not in doc
    assert history.summary(ctx)["status"] == "running"          # what the others see


def test_two_workers_at_once_fold_each_session_once(env, repo, claude_home):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    results = []
    gate = threading.Barrier(2)

    def work():
        c = ctx_for(detached(env), repo)
        gate.wait()
        results.append(history.run(c, auto=True))
    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r["state"] for r in results) == ["done", "done"]
    assert sum(r.get("read", 0) for r in results) == 4
    assert [r.get("reason") for r in results].count("nothing to resume") == 1
    assert ledger_ids(ctx.store) == OLD


HOOK = r"""
import json, sys
from tokeneconomizer.plugin import hook
sys.exit(hook.main(["--host", sys.argv[1]]))
"""


def test_two_agents_starting_in_one_repository_each_import_once(env, repo, claude_home,
                                                                tmp_path):
    """Claude Code and Codex start in the repository at the same moment, each its own hook
    process, twice: each store imports the corpus once (Codex's as foreign sessions)."""
    corpus(claude_home, repo)
    for round_ in ("a", "b"):
        procs = []
        for host in ("claude", "codex"):
            t = str(tmp_path / f"{host}-{round_}.jsonl")
            open(t, "w").close()
            payload = json.dumps({"hook_event_name": "SessionStart", "session_id": host + round_,
                                  "source": "startup", "transcript_path": t, "cwd": repo})
            p = subprocess.Popen([sys.executable, "-c", HOOK, host], env=child_env(env) | {
                "TE_FOLD_SYNC": "1"}, cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            procs.append((p, payload))
        for p, payload in procs:
            p.stdin.write(payload)
            p.stdin.close()
        for p, _ in procs:
            assert p.wait(timeout=120) == 0, p.stderr.read()
            json.loads(p.stdout.read().splitlines()[-1])
        claude, codex = store_for(env, repo), store_for(env, repo, host="codex")
        assert ledger_ids(claude) == OLD
        assert ledger_ids(codex) == [f"claude:{s}" for s in OLD]
        for st in (claude, codex):
            doc = st.read("history.json")
            assert doc["state"] == "done" and doc["imported"] == 4


# ---- what is reported, and the pieces --------------------------------------------------------
def test_a_worker_that_never_got_the_fold_lock_hands_its_claim_back(env, repo, claude_home,
                                                                   monkeypatch):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    spawned = []
    monkeypatch.setattr(LiveContext, "spawn", lambda self, argv, log: spawned.append(argv))
    history.maybe_start(ctx)
    monkeypatch.setattr(fold, "FOLD_LOCK_WAIT_S", 0.2)
    holding, release = threading.Event(), threading.Event()

    def live_fold():
        with ctx_for(detached(env), repo).store.locked("fold") as got:
            assert got
            holding.set()
            release.wait(10)
    th = threading.Thread(target=live_fold)
    th.start()
    assert holding.wait(10)
    try:
        assert history.run(ctx_for(detached(env), repo), auto=True)["state"] == "busy"
    finally:
        release.set()
        th.join()
    doc = ctx.store.read("history.json")
    assert doc["state"] == "partial" and "fold lock was busy" in doc["reason"]
    assert history.describe(history.summary(ctx)).startswith("waiting to resume: 0 session(s)")
    history.maybe_start(ctx)
    assert len(spawned) == 2 and ctx.store.read("history.json")["state"] == "running"
    assert history.run(ctx, auto=True)["state"] == "done"
    doc = ctx.store.read("history.json")
    assert "reason" not in doc and ledger_ids(ctx.store) == OLD


@needs_kill
def test_when_an_import_counts_as_stalled():
    now = time.time()
    me, here = os.getpid(), history._this_host()
    fresh = {"state": "running", "heartbeat": now - 10}
    assert history.stalled(fresh, now) is None
    assert history.stalled(dict(fresh, pid=me, host=here), now) is None
    gone = dead_pid()
    assert history.stalled(dict(fresh, pid=gone, host=here), now) == (
        f"its process, pid {gone}, is gone")
    # Another machine's pid says nothing here, nor does a pid with no host (an older doc).
    assert history.stalled(dict(fresh, pid=gone, host=here + "-elsewhere"), now) is None
    assert history.stalled(dict(fresh, pid=gone), now) is None
    # A quiet heartbeat outweighs a live pid, which another process may have been given.
    quiet = dict(fresh, heartbeat=now - history.HEARTBEAT_STALE_S, pid=me, host=here)
    assert history.stalled(quiet, now) == "no heartbeat for 15 min"
    for state in ("partial", "done", "stopped", None):
        assert history.stalled(dict(quiet, state=state), now) is None
    assert history.stalled({"state": "running"}, now).startswith("no heartbeat for")
    assert history.stalled(None) is None and history.stalled([1]) is None
    for bad in (None, 0, -1, True, "12"):
        assert history._pid_alive(bad) is None
    assert history._pid_alive(me) is True


def test_a_run_refreshes_its_heartbeat_while_it_plans(env, repo, claude_home, monkeypatch):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time()})
    monkeypatch.setattr(history, "BEAT_EVERY_S", 0)
    beats, real = [], history._Heads.head

    def slow_head(self, source, log):
        time.sleep(0.01)
        beats.append(ctx.store.read("history.json")["heartbeat"])
        return real(self, source, log)
    monkeypatch.setattr(history._Heads, "head", slow_head)
    assert history.run(ctx, auto=True)["state"] == "done"
    assert len(beats) == 4 and beats == sorted(beats) and beats[-1] > beats[0]


def test_status_json_says_what_the_import_is_doing(env, repo, claude_home, monkeypatch, capsys):
    corpus(claude_home, repo)
    ctx = ctx_for(detached(env), repo)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time() - 2000,
                                     "plan": [{"id": "x"}] * 3, "next": 1, "imported": 1})
    cli.main(["--cwd", repo, "--host", "claude", "status", "--json"])
    h = json.loads(capsys.readouterr().out)["history"]
    assert h["status"] == "stalled" and h["stalled"] == "no heartbeat for 33 min"
    assert h["state"] == "running" and h["left"] == 2 and h["heartbeat_age_s"] >= 2000
    cli.main(["--cwd", repo, "--host", "claude", "history", "status"])
    assert ("stalled: the import stopped before it finished (no heartbeat for 33 min); 1 "
            "session(s) read, 2 to go") in capsys.readouterr().out
    ctx.store.write("history.json", {"state": "running", "heartbeat": time.time(),
                                     "plan": [{"id": "x"}] * 3, "next": 1, "imported": 1,
                                     "pid": os.getpid(), "host": history._this_host()})
    cli.main(["--cwd", repo, "--host", "claude", "history", "status"])
    assert "reading the agents' history: 1 session(s) so far, 2 to go" in (
        capsys.readouterr().out)
