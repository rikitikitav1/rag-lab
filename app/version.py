import hashlib
import json
import os
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


# the bytes of every module, because an uncommitted edit changes no commit hash and still changes the run
def tree_stamp() -> str:
    digest = hashlib.sha256()
    # the config decides roles, models and samplers, so an edit to it counts as much as a module
    for path in sorted(APP.rglob("*.py")) + [APP.parent / "config.yaml"]:
        try:
            # content, not mtime: a checkout that restores the same bytes is not a different stand
            digest.update(f"{path.name}:".encode() + hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(f"{path.name}:gone".encode())
    return digest.hexdigest()[:12]


LOADED_TREE = tree_stamp()
STARTED = datetime.now(timezone.utc).isoformat(timespec="seconds")


# a hash has no order, so this says "not the same", never "older": a revert differs as much as an edit
def differs_from_disk() -> str | None:
    now = tree_stamp()
    return None if now == LOADED_TREE else f"loaded {LOADED_TREE}, on disk {now}"


# set by the container that runs the worker: with no default, nothing on the host can speak for it
SAID = Path(os.environ["WORKER_STAMP"]) if os.getenv("WORKER_STAMP") else None


def say_loaded(path: Path | None = None) -> None:
    path = path or SAID
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mine() | {"pid": os.getpid()}))
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
    return out | ({"differs": said} if said else {})
