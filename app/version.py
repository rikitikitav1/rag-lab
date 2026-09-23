import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import logging_setup

log = logging_setup.get_logger(__name__)

GIT = Path(__file__).resolve().parent.parent / ".git"
APP = Path(__file__).resolve().parent


# read once at import: a run has to record which of our code produced it, not only which libraries
def _read() -> str | None:
    env = os.getenv("CODE_VERSION")
    if env:
        return env[:12]
    try:
        head = (GIT / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:12] or None
        ref = head[5:]
        loose = GIT / ref
        if loose.exists():
            return loose.read_text().strip()[:12] or None
        # after gc the ref lives in packed-refs, and a worktree keeps .git as a file
        for line in (GIT / "packed-refs").read_text().splitlines():
            if line.endswith(f" {ref}"):
                return line.split()[0][:12]
    except OSError as e:
        log.warning("version.unreadable", error=str(e))
        return None
    log.warning("version.ref_not_found")
    return None


CODE_VERSION = _read()


# the config this process loaded, overlay included: a cpu arm and a gpu arm are two stands
def _config_files() -> list:
    import config

    names = [config.CONFIG_PATH, config.CONFIG_OVERLAY]
    return [Path(name) if Path(name).is_absolute() else APP.parent / name for name in names if name]


# the path, not the basename: two `base.py` in two packages are two files
def _name_of(path: Path) -> str:
    try:
        return path.relative_to(APP.parent).as_posix()
    except ValueError:
        return path.as_posix()


# the config decides roles, models and samplers, so an edit to it counts as much as a module
def _walk() -> list:
    return sorted(APP.rglob("*.py")) + _config_files()


# one digest per file, kept apart so a reader can ask which file moved, not only whether the tree did
def _digests() -> dict:
    out = {}
    for path in _walk():
        try:
            # content, not mtime: a checkout that restores the same bytes is not a different stand
            out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            out[str(path)] = "gone"
    return out


# the bytes of every module, because an uncommitted edit changes no commit hash and still changes the run
def tree_stamp(digests: dict | None = None, walk: list | None = None) -> str:
    digests = _digests() if digests is None else digests
    digest = hashlib.sha256()
    for path in walk if walk is not None else _walk():
        name, one = _name_of(Path(path)), digests.get(str(path), "gone")
        if one == "gone":
            digest.update(f"{name}:gone".encode())
        else:
            digest.update(f"{name}:".encode() + bytes.fromhex(one))
    return digest.hexdigest()[:12]


# what the tree held when this process started, per file: the base `loaded_differs` compares against
LOADED_FILES = _digests()
# the walk is the same one `_digests` just made, so the stamp folds it rather than reading the tree again
LOADED_TREE = tree_stamp(LOADED_FILES, walk=list(LOADED_FILES))
STARTED = datetime.now(timezone.utc).isoformat(timespec="seconds")


_ASKED_AT = 0.0
_ANSWERED: str | None = None
# every three seconds an idle worker would read the whole tree to learn nothing
SAY_AGAIN_AFTER = 10.0


# a hash has no order, so this says "not the same", never "older": a revert differs as much as an edit
def differs_from_disk(every: float = SAY_AGAIN_AFTER) -> str | None:
    global _ASKED_AT, _ANSWERED
    now = time.monotonic()
    if now - _ASKED_AT < every:
        return _ANSWERED
    _ASKED_AT = now
    stamp = tree_stamp()
    _ANSWERED = None if stamp == LOADED_TREE else f"loaded {LOADED_TREE}, on disk {stamp}"
    return _ANSWERED


_LOADED_ASKED_AT = 0.0
_LOADED_ANSWERED: dict | None = None
# how many names a finding carries: a flag says "look", a name says where
LOADED_NAMES = 5


# the files this process actually imported, plus the config by declaration: it is read once into settings
def _loaded_paths() -> set:
    import sys

    out = {str(p) for p in _config_files()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        # resolved as APP is, or a module imported through a symlink or a relative path is never looked at
        path = Path(name).resolve()
        if path.suffix == ".py" and path.is_relative_to(APP):
            out.add(str(path))
    return out


# the sharp reading: not "did the tree move" but "does this process run code the tree no longer holds"
def loaded_differs(every: float = SAY_AGAIN_AFTER) -> dict | None:
    global _LOADED_ASKED_AT, _LOADED_ANSWERED
    now = time.monotonic()
    if now - _LOADED_ASKED_AT < every:
        return _LOADED_ANSWERED
    _LOADED_ASKED_AT = now
    moved = []
    for path in sorted(_loaded_paths()):
        try:
            here = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            here = "gone"
        # a file the process imported that the tree did not hold at start is new code it is running
        if LOADED_FILES.get(path, "absent") != here:
            moved.append(_name_of(Path(path)))
    _LOADED_ANSWERED = {"count": len(moved), "files": moved[:LOADED_NAMES]} if moved else None
    return _LOADED_ANSWERED


# set by the container that runs the worker: with no default, nothing on the host can speak for it
SAID = Path(os.environ["WORKER_STAMP"]) if os.getenv("WORKER_STAMP") else None


_LAST_SAID: tuple | None = None


# said at start and again from the loop, so a file edited under a running worker reaches the stand page
def say_loaded(path: Path | None = None) -> None:
    global _LAST_SAID
    path = path or SAID
    if path is None:
        return
    text = json.dumps(mine() | {"pid": os.getpid()})
    if (path, text) == _LAST_SAID:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        _LAST_SAID = (path, text)
    except OSError as e:
        log.warning("version.stamp_unwritten", error=str(e))


def what_the_worker_loaded(path: Path | None = None) -> dict | None:
    path = path or SAID
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# what this process would run, written onto the job it claims: "one code" becomes a query, not a comparison by hand
def mine() -> dict:
    out = {"stamp": LOADED_TREE, "code_version": CODE_VERSION, "at": STARTED}
    # the row says it rather than the queue stalling: a reader drops such a pass, a run does not wait
    said = differs_from_disk()
    # `loaded_differs` is always present and `null` when clean, so a clean process is not an older writer
    return out | ({"differs": said} if said else {}) | {"loaded_differs": loaded_differs()}
