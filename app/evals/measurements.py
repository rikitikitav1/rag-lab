"""Where a job leaves a number so it can be cited later.

Every measurement of this arc was saved by hand from a shell, so a job that computes one had
nowhere to put it and logged it instead. The path is derived here and never taken from options:
a job that writes to a path its caller names writes to someone else's disk.
"""

import json
import os
import re
from datetime import date
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[2] / "datasets" / "measurements"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:60] or "unnamed"


# the worker runs as root over the host's tree, so the file is handed to whoever owns the folder
def hand_back(path: Path) -> None:
    try:
        info = path.parent.stat()
        os.chown(path, info.st_uid, info.st_gid)
        path.chmod(0o664)
    except (PermissionError, OSError):
        pass


def record(kind: str, run_name: str, payload: dict, on: date | None = None) -> str:
    FOLDER.mkdir(parents=True, exist_ok=True)
    stamp = (on or date.today()).strftime("%Y%m%d")
    path = FOLDER / f"{_slug(kind)}_{_slug(run_name)}_{stamp}.json"
    # two passes of one probe in a day are two measurements, and the first was overwritten
    for nth in range(2, 100):
        if not path.exists():
            break
        path = FOLDER / f"{_slug(kind)}_{_slug(run_name)}_{stamp}_{nth}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(path)
    return str(path)
