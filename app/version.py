import os
from pathlib import Path

GIT = Path(__file__).resolve().parent.parent / ".git"


# read once at import: a run has to record which of our code produced it, not only which libraries
def _read() -> str | None:
    env = os.getenv("CODE_VERSION")
    if env:
        return env[:12]
    try:
        head = (GIT / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            head = (GIT / head[5:]).read_text().strip()
        return head[:12] or None
    except OSError:
        return None


CODE_VERSION = _read()
