"""Directories whose agent sessions the plugin never reads, caches or learns from.

Some of a person's directories are private whatever an agent did there: the Messages and Photos
libraries, a family folder, and the PARA reference and archive trees. A session that ran in one
is left out everywhere: the history import and its head cache (`history.py`), `te history
import --all`, a live fold's catch-up, and the hooks of a session running there now.

A hook of a session running in one does nothing: no context is built, no event recorded, no
store made (`hook.run`), and a plugin that copies its sessions itself is told to keep no copy
(`EXCLUDED_KEY`).

Excluded, always (a setting can add to them, never take them away):

  * under the home directory: `Library/Messages`, `Documents/Family` and
    `Pictures/Photos Library.photoslibrary`;
  * anywhere: a directory named `03_resources` or `04_archives` (the PARA trees, wherever the
    person keeps them), or ending in `.photoslibrary`;
  * the directories `history_exclude` names.

Matching ignores case (macOS's file system does) and looks at a path as written, as its real
path, and as the folder name an agent files it under: Claude Code and Droid turn every
character that is not a letter or a digit into `-`, Grok Build URL-quotes the directory. That
folder is part of the log's path, so for those agents a log is left out before its head is
read. A folder name cannot tell `Family-photos` from `Family/photos`, so it errs toward
leaving such a neighbour out too. The other agents record the directory only inside the log;
there the head (the first lines) is read to learn it, and an excluded one is dropped at once,
never cached, its body never read.

A log is also judged by where it really is: one that is, or sits under, a link into an excluded
directory is left out before its head is read, whatever directory it names. So is a live
session's transcript.
"""
import os
import re
from urllib.parse import unquote

from .. import trace

# What the hook answers, in place of anything else, to a plugin that keeps its own copy of a
# session (Amp's, opencode's): this session is in an excluded directory, so keep none.
EXCLUDED_KEY = "tokeneconomizer_excluded"

HOME_DIRS = ("Library/Messages", "Documents/Family", "Pictures/Photos Library.photoslibrary")
ANY_NAMES = ("03_resources", "04_archives")
ANY_SUFFIXES = (".photoslibrary",)


def _home(env):
    return (env.get("HOME") or env.get("USERPROFILE") or os.path.expanduser("~")) or ""


def _norm(path):
    return os.path.normcase(path).replace(os.sep, "/").replace("\\", "/").casefold().rstrip("/")


def _folder(path):
    return trace.claude_project_folder(path).casefold()


class Excluded:
    """The excluded directories of one environment: `path`, `folder`, `log`, `head` say whether
    a directory, an agent's folder name, a listed log or a log's head falls in one."""

    def __init__(self, env=None, extra=()):
        env = os.environ if env is None else env
        home = _home(env)
        roots = [os.path.join(home, d) for d in HOME_DIRS] if home else []
        roots += [os.path.expanduser(str(p).strip()) for p in (extra or ()) if str(p).strip()]
        roots = [r for r in roots if os.path.isabs(r)]
        self.roots = sorted({_norm(v) for r in roots
                             for v in (os.path.abspath(r), os.path.realpath(r))})
        self.folders = sorted({_folder(v) for r in roots
                               for v in (os.path.abspath(r), os.path.realpath(r))})
        names = "|".join(re.escape(_folder(n)) for n in ANY_NAMES)
        ends = "|".join(re.escape(_folder(s).lstrip("-")) for s in ANY_SUFFIXES)
        self._folder_re = re.compile(rf"(?:^|-)(?:{names})(?:-|$)|-(?:{ends})(?:-|$)")
        self._seen = {}

    def _written(self, path):
        p = _norm(path)
        for root in self.roots:
            if p == root or p.startswith(root + "/"):
                return True
        for part in p.split("/"):
            if part in ANY_NAMES or any(part.endswith(s) for s in ANY_SUFFIXES):
                return True
        return False

    def path(self, path, real=True):
        """Is the directory or file `path` in an excluded directory? `real`: its real path too
        (a link into one), which costs a look at the file system, once per path."""
        if not isinstance(path, str) or not path:
            return False
        key = (path, real)
        if key not in self._seen:
            hit = False
            for p in {path, unquote(path)}:
                if os.path.isabs(p) and (self._written(os.path.abspath(p)) or (
                        real and self._written(os.path.realpath(p)))):
                    hit = True
                    break
            self._seen[key] = hit
        return self._seen[key]

    def folder(self, folder):
        """Is `folder`, the name an agent filed a directory's logs under, an excluded one?"""
        if not isinstance(folder, str) or not folder:
            return False
        decoded = unquote(folder)
        if decoded != folder and os.path.isabs(decoded) and self.path(decoded, real=False):
            return True
        f = folder.casefold()
        if any(f == r or f.startswith(r + "-") for r in self.folders):
            return True
        return bool(self._folder_re.search(f))

    def log(self, log):
        """Is a listed log (`{"path", "folder"?}`) one to leave out before its head is read?"""
        if not isinstance(log, dict):
            return False
        path = log.get("path")
        # A database row is `<db>#<id>`: the file is the database.
        file = path.partition("#")[0] if isinstance(path, str) else None
        # The folder an agent files a log under is the log's parent, named for the directory.
        parent = os.path.basename(os.path.dirname(file)) if file else None
        return (self.path(file) or self.folder(log.get("folder")) or self.folder(parent))

    def head(self, head):
        """Does a log's head say its session ran in an excluded directory?"""
        if not isinstance(head, dict):
            return False
        return self.path(head.get("cwd")) or self.folder(head.get("folder"))


def excluded(env=None, config=None):
    """The `Excluded` of an environment and its settings (`history_exclude`)."""
    return Excluded(env, getattr(config, "history_exclude", ()) if config is not None else ())


def live(cwd, env, *transcripts):
    """Is a running session's directory `cwd`, or one of its `transcripts`, in an excluded
    directory?
    The directories a setting adds come from the global config and the environment: the
    repository's own file lives in its store, which a session here must not make."""
    from .config import load_config
    from .paths import state_root
    ex = excluded(env, load_config(state_root(env), None, env))
    return ex.path(cwd) or any(ex.path(t) for t in transcripts)


def for_ctx(ctx):
    """The `Excluded` of a context, made once and kept on it."""
    ex = getattr(ctx, "_excluded", None)
    if ex is None:
        ex = excluded(ctx.env, ctx.config)
        try:
            ctx._excluded = ex
        except AttributeError:
            pass
    return ex
