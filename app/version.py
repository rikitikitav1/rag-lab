import os
from pathlib import Path

import logging_setup

log = logging_setup.get_logger(__name__)

GIT = Path(__file__).resolve().parent.parent / ".git"


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
