"""Sessions that ran in an excluded directory (`privacy`): the Messages and Photos libraries, a
family folder, the PARA reference and archive trees, and the directories `history_exclude`
names. None is read past its head, cached, planned, folded or learned from, by the import of
one repository, `te history import --all`, a resumed plan, a live fold's catch-up or the hooks
of a session running there now. The home and every log are synthetic."""
import json
import os
import time

import pytest

from test_history_import import ledger_ids, trap, history_log
from test_history_sources import SID, codex_log, env, home, repo  # noqa: F401
from test_live_hooks import call, store_for
from tokeneconomizer import trace
from tokeneconomizer.trace import sources
from tokeneconomizer.plugin import cli, history, hosts, privacy
from tokeneconomizer.plugin.config import load_config
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.store import Store

EXCLUDED = ("Documents/Family", "Library/Messages", "Pictures/Photos Library.photoslibrary",
            "Documents/03_resources", "Documents/04_archives")


def checkout(path, name):
    (path / ".git" / "info").mkdir(parents=True)
    (path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (path / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = https://github.com/acme/{name}.git\n')
    return str(path)


@pytest.fixture
def private(home):
    """A checkout in each excluded directory class."""
    return [checkout(home / d / "notes", f"private{i}") for i, d in enumerate(EXCLUDED)]


def cache_of(env):
    doc = Store(env["TOKENECONOMIZER_HOME"]).read(history.CACHE)
    return doc if isinstance(doc, dict) else {}


def names_private(text, private):
    return any(p in text or trace.claude_project_folder(p) in text for p in private)


def test_every_class_is_matched_by_path_folder_and_head_and_its_neighbours_are_not(tmp_path):
    h = tmp_path / "h"
    for d in EXCLUDED:
        (h / d / "sub").mkdir(parents=True)
    ex = privacy.Excluded({"HOME": str(h)}, extra=[str(tmp_path / "mine")])
    for d in EXCLUDED:
        p = str(h / d / "sub")
        assert ex.path(p) and ex.head({"cwd": p})
        assert ex.folder(trace.claude_project_folder(p))
        assert ex.log({"path": f"/logs/{trace.claude_project_folder(p)}/s.jsonl"})
    assert ex.path(str(h / "LIBRARY" / "messages"))                   # case is ignored
    assert ex.path("/elsewhere/03_resources/x") and ex.path("/x/Other.photoslibrary/y")
    assert ex.path(str(tmp_path / "mine" / "a"))                      # history_exclude
    os.symlink(h / "Documents" / "Family", tmp_path / "link")
    assert ex.path(str(tmp_path / "link" / "sub"))                    # a link into one
    for ok in ("Documents/Familyx", "Code/widget", "Documents/03_resourcesx"):
        p = str(h / ok)
        assert not ex.path(p), ok
        assert not ex.folder(trace.claude_project_folder(p)), ok
    # A folder name cannot tell `Family-photos` from `Family/photos`: it errs toward leaving out.
    sibling = str(h / "Documents" / "Family-photos")
    assert not ex.path(sibling) and ex.folder(trace.claude_project_folder(sibling))


def test_import_all_reads_no_excluded_session_and_still_reads_the_rest(
        env, home, repo, private, tmp_path, monkeypatch, capsys):
    """The verifier's case: Claude Code logs (filed under a folder named for the directory)
    and Codex logs (the directory only inside the log) in every excluded class, beside one
    ordinary repository."""
    for i, p in enumerate(private):
        for n in range(2):
            history_log(home / ".claude", p, f"x{i}{n}", trap(p, n), age_s=3600 + n)
    codex_log(home, private[0], SID["codex"], 0, age_s=3600)
    history_log(home / ".claude", repo, "c1", trap(repo, 1), age_s=7200)
    read = []
    real = history._Heads.head

    def spy(self, source, log):
        out = real(self, source, log)
        if out is not None:
            read.append(log["path"])
        return out
    monkeypatch.setattr(history._Heads, "head", spy)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all"]) == 0
    assert "1 repository in the agents' history, 1 store(s), 1 session(s) read" in \
        capsys.readouterr().out
    assert ledger_ids(store_for(env, repo)) == ["c1"]
    cache = cache_of(env)
    assert cache and not names_private(json.dumps(cache), private)
    assert not any(names_private(p, private) for p in read)
    for p in private:
        for host in ("claude", "codex"):
            assert not os.path.isdir(store_for(env, p, host=host).base)
    repos_dir = os.path.join(env["TOKENECONOMIZER_HOME"], "repos")
    for root, _, files in os.walk(repos_dir):
        for f in files:
            with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                assert not names_private(fh.read(), private), os.path.join(root, f)


def test_a_cache_an_earlier_version_wrote_is_cleared_of_excluded_sessions(
        env, home, repo, private):
    keep = history_log(home / ".claude", repo, "c1", trap(repo, 1), age_s=3600)
    gone = history_log(home / ".claude", private[1], "x1", trap(private[1], 1), age_s=3600)
    codex = codex_log(home, private[2], SID["codex"], 0, age_s=3600)
    head = {"session_id": "s", "cwd": None, "started": 1.0, "headless": False, "folder": None}
    root = env["TOKENECONOMIZER_HOME"]
    Store(root).write(history.CACHE, {
        keep: {"mtime": 1.0, "size": 1, "head": dict(head, cwd=repo)},
        gone: {"mtime": 1.0, "size": 1, "head": dict(head, cwd=private[1])},
        str(codex): {"mtime": 1.0, "size": 1, "head": dict(head, cwd=private[2])}})
    linked = os.path.join(os.path.dirname(keep), "x2#extra.jsonl")
    os.symlink(gone, linked)
    doc = Store(root).read(history.CACHE)
    doc[linked] = {"mtime": 1.0, "size": 1, "head": dict(head, cwd=repo)}
    Store(root).write(history.CACHE, doc)
    heads = history._Heads(root, privacy.excluded(env))
    assert set(heads.doc) == {keep} and heads.dirty
    history.repositories(root, env, load_config(root, None, env))
    assert set(cache_of(env)) == {keep}


def test_a_repository_in_an_excluded_directory_imports_nothing_and_drops_a_saved_plan(
        env, home, private, tmp_path):
    p = private[3]
    history_log(home / ".claude", p, "x0", trap(p, 0), age_s=3600)
    ctx = LiveContext.build(hosts.ClaudeCode(), p, dict(env, TE_HISTORY_IMPORT="on"))
    ctx.store.write("history.json", {"state": "partial", "next": 0, "imported": 0, "plan": [
        {"id": "x0", "key": "x0", "source": "claude", "path": "/nowhere/x0.jsonl",
         "started": 1.0, "ended": 2.0}]})
    res = history.run(ctx, auto=True)
    assert res["state"] == "stopped" and res["reason"] == history.EXCLUDED
    doc = ctx.store.read("history.json")
    assert "plan" not in doc and doc["state"] == "stopped"
    assert history.maybe_start(ctx) is None
    assert history.catch_up(ctx, time.time()) == 0
    assert not names_private(json.dumps(cache_of(env)), private)
    assert not ctx.store.read("folded.json")


def test_a_saved_plan_elsewhere_leaves_out_a_session_that_ran_in_an_excluded_directory(
        env, home, repo, private):
    """A plan saved before the check: an item in a Family clone of the same repository."""
    clone = checkout(home / "Documents" / "Family" / "widget", "widget")
    path = history_log(home / ".claude", clone, "f1", trap(clone, 1), age_s=3600)
    cpath = codex_log(home, clone, SID["codex"], 0, age_s=3600)
    ctx = LiveContext.build(hosts.ClaudeCode(), repo, dict(env, TE_HISTORY_IMPORT="on"))
    for item in ({"id": "f1", "source": "claude", "path": path},
                 {"id": SID["codex"], "source": "codex", "path": str(cpath)}):
        assert history._still_eligible(ctx, item) == "excluded"


def test_a_hook_in_an_excluded_directory_does_nothing_and_leaves_no_state(
        env, home, private, tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text("")
    mine = checkout(tmp_path / "mine", "mine")
    for cwd, e in [(p, env) for p in private] + [(mine, dict(env, TE_HISTORY_EXCLUDE=mine))]:
        for event in ("SessionStart", "UserPromptSubmit", "SessionEnd"):
            assert call(e, {"hook_event_name": event, "session_id": "s1", "cwd": cwd,
                            "transcript_path": str(t), "prompt": "hi"}) == {}
        out = call(e, {"hook_event_name": "SessionStart", "session_id": "T-1", "cwd": cwd,
                       "transcript_path": str(t)}, host="amp")
        assert out == {privacy.EXCLUDED_KEY: True}
        for host in ("claude", "amp"):
            assert not os.path.isdir(store_for(e, cwd, host=host).base)
    assert not os.path.isdir(os.path.join(env["TOKENECONOMIZER_HOME"], "repos"))


def test_a_hook_elsewhere_answers_as_before(env, repo, tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text("")
    out = call(env, {"hook_event_name": "SessionStart", "session_id": "T-1", "cwd": repo,
                     "transcript_path": str(t)}, host="amp")
    assert privacy.EXCLUDED_KEY not in out
    assert os.path.isdir(store_for(env, repo, host="amp").base)


def test_the_amp_plugin_keeps_no_copy_of_a_thread_in_an_excluded_workspace(
        env, home, private, tmp_path):
    """The plugin file itself under Node, with the real launcher: a thread in a Family
    checkout leaves no copy, no store and no context, and a copy an earlier run left is
    removed."""
    import shutil
    import subprocess
    from test_amp import NODE_DRIVER, TS, TID, plugin_messages
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    copy = os.path.join(env["TOKENECONOMIZER_HOME"], "amp-threads", f"{TID}.jsonl")
    os.makedirs(os.path.dirname(copy))
    with open(copy, "w") as fh:
        fh.write(json.dumps({"type": "thread", "id": TID, "cwd": private[0]}) + "\n")
    run_env = dict(os.environ, **env, TE_AMP_MODE="interactive",
                   TE_TEST_MESSAGES=json.dumps(plugin_messages("fix the tests")))
    for k in ("TE_FOLD_SYNC", "TE_SIDECALL", "TE_DISABLE", "PYTHONHOME", "PYTHONPATH"):
        run_env.pop(k, None)
    run = subprocess.run([node, "--experimental-strip-types", "--no-warnings",
                          "--input-type=module", "-e", NODE_DRIVER, TS, private[0], TID],
                         capture_output=True, text=True, timeout=120, env=run_env)
    if run.returncode and "strip-types" in run.stderr:
        pytest.skip("this node cannot load TypeScript")
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out.get("start") is None and out["tool"] is None        # no context added
    assert not os.path.exists(copy)
    assert not os.path.isdir(store_for(env, private[0], host="amp").base)
    assert not os.path.isdir(os.path.join(env["TOKENECONOMIZER_HOME"], "repos"))


def test_a_log_that_links_into_an_excluded_directory_is_never_read(
        env, home, repo, private, tmp_path, monkeypatch):
    """The verifier's second case: ordinary-looking entries under a normal repository's folder
    whose files live in Family; and a whole project folder that is a link into Messages."""
    family = home / "Documents" / "Family"
    projects = home / ".claude" / "projects"
    folder = projects / trace.claude_project_folder(repo)
    # One link plainly named, one with a `#` in its name (not a database row).
    for n, name in enumerate(("l0.jsonl", "l1#extra.jsonl")):
        real = history_log(tmp_path / "staging", repo, f"l{n}", trap(repo, n), age_s=3600 + n)
        target = family / f"l{n}.jsonl"
        os.replace(real, target)
        folder.mkdir(parents=True, exist_ok=True)
        os.symlink(target, folder / name)
    other = checkout(tmp_path / "other", "other")
    hidden = home / "Library" / "Messages" / "agent-logs"
    history_log(tmp_path / "staging2", other, "m1", trap(other, 1), age_s=3600)
    os.replace(tmp_path / "staging2" / "projects" / trace.claude_project_folder(other), hidden)
    os.symlink(hidden, projects / trace.claude_project_folder(other))
    opened = []
    real_open = open

    def spy(path, *a, **k):
        opened.append(os.path.realpath(path) if isinstance(path, (str, os.PathLike)) else path)
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", spy)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all"]) == 0
    monkeypatch.setattr("builtins.open", real_open)
    assert not any(isinstance(p, str) and (p.startswith(str(family)) or p.startswith(str(hidden)))
                   for p in opened)
    assert cache_of(env) == {} or not any(
        os.path.realpath(p).startswith((str(family), str(hidden))) for p in cache_of(env))
    assert not os.path.isdir(store_for(env, repo).base)
    assert not os.path.isdir(store_for(env, other).base)


def test_a_hook_whose_transcript_links_into_an_excluded_directory_does_nothing(
        env, home, repo, tmp_path):
    target = home / "Documents" / "Family" / "t.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("")
    link = tmp_path / "t.jsonl"
    os.symlink(target, link)
    assert call(env, {"hook_event_name": "SessionStart", "session_id": "s1", "cwd": repo,
                      "transcript_path": str(link)}) == {}
    assert not os.path.isdir(store_for(env, repo).base)


def _connected(monkeypatch):
    """The real path of every database file opened from here on."""
    import sqlite3
    from urllib.parse import unquote, urlparse
    opened, real_connect = [], sqlite3.connect

    def spy(database, *a, **k):
        p = str(database)
        if k.get("uri") or p.startswith("file:"):
            p = unquote(urlparse(p).path)
        opened.append(os.path.realpath(p))
        return real_connect(database, *a, **k)
    monkeypatch.setattr(sqlite3, "connect", spy)
    return opened


def _database_at(name, home, repo):
    import test_history_sources as ths
    if name == "hermes":
        return ths.hermes_db(home, [("h-1", repo, 3600, "cli")])
    if name == "opencode":
        return ths.opencode_db(home, [("o-1", repo, 3600)])
    return ths.zeroclaw_db(home, [("z-1", repo, 3600)])


@pytest.mark.parametrize("name", ["hermes", "opencode", "zeroclaw"])
@pytest.mark.parametrize("linked", [False, True])
def test_a_session_database_in_an_excluded_directory_is_never_opened(
        name, linked, env, home, repo, tmp_path, monkeypatch):
    """The verifier's database case: the agent's store is a link to a file in Family. The import
    must not open it at all, not even to list its rows; the same store where it belongs is read."""
    db = _database_at(name, home, repo)
    family = home / "Documents" / "Family"
    if linked:
        family.mkdir(parents=True)
        target = family / os.path.basename(db)
        os.replace(db, target)
        os.symlink(target, db)
    opened = _connected(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all"]) == 0
    if linked:
        assert not any(p.startswith(str(family)) for p in opened), opened
        assert not any(os.path.realpath(p).startswith(str(family)) for p in cache_of(env))
    else:
        assert os.path.realpath(db) in opened
        assert any(p.startswith(db + "#") for p in cache_of(env))


@pytest.mark.parametrize("name", ["hermes", "zeroclaw"])
def test_a_live_fold_does_not_export_from_a_store_in_an_excluded_directory(
        name, home, tmp_path, monkeypatch):
    family = home / "Documents" / "Family"
    (family / "data" / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TOKENECONOMIZER_HOME", str(tmp_path / "te"))
    calls = []
    monkeypatch.setattr(trace, f"{name}_export", lambda *a, **k: calls.append(a) or True)
    host = hosts.HOSTS[name]()
    session = ({"hermes_home": str(family)} if name == "hermes"
               else {"zeroclaw_config_dir": str(family)})
    copy = str(tmp_path / "copy.json")
    open(copy, "w").close()
    assert host.complete(copy, "s-1", session=session) == copy
    assert calls == []
    ordinary = home / "agent"
    (ordinary / "data" / "sessions").mkdir(parents=True)
    host.complete(copy, "s-1", session={"hermes_home": str(ordinary),
                                        "zeroclaw_config_dir": str(ordinary)})
    assert len(calls) == 1


@pytest.mark.parametrize("name", sorted(sources.SOURCES))
def test_every_source_takes_the_skip_check_the_import_passes(name, env, home, repo):
    """`history.scan` lists every source with `skip=`; a source without it raises, the scan
    records an error and moves on, and that agent's sessions are never imported."""
    import test_history_sources as ths
    ths._write(name, home, repo, ths.SID.get(name, f"{name}-s1"))
    src = sources.SOURCES[name]
    assert src.logs(env, (), skip=lambda p: False) == src.logs(env)


def test_import_all_reads_a_muse_session_and_not_one_run_in_an_excluded_directory(
        env, home, repo, private, tmp_path, monkeypatch, capsys):
    import test_history_sources as ths
    kept = ths.muse_log(home, repo, "muse-ok", 1, age_s=3600)
    ths.muse_log(home, private[0], "muse-private", 2, age_s=3600)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", str(tmp_path), "history", "import", "--all"]) == 0
    assert "1 repository in the agents' history, 1 store(s), 1 session(s) read" in \
        capsys.readouterr().out
    cache = cache_of(env)
    assert kept in cache and not names_private(json.dumps(cache), private)
