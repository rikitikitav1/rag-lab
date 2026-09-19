"""Where a job leaves a number so it can be cited later."""

import gzip
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[2] / "datasets" / "measurements"
# inputs of an instrument, not artifacts of a run: they are read by the code and live in git
PANELS = FOLDER.parent / "panels"


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


# the same two lines lived in the anchor, one of the two hands that write a file beside a record
def store_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    hand_back(path)


# a dry read must not leave a number behind, and two scripts wrote this sentence apart
def say_where(kind: str, run_name: str, payload: dict, asked: bool) -> str:
    if not asked:
        return "not recorded; add --record once this is a real reading"
    return f"recorded: {record(kind, run_name, payload)}"


# the rows a number was computed from: needed to recompute it, useless to a reader, heavy in git
def _put_bulk(path: Path, payload: dict, bulk) -> dict:
    out = dict(payload)
    for key in bulk:
        rows = out.pop(key, None)
        if rows is None:
            continue
        blob = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        beside = path.with_name(f"{path.stem}_{key}.json.gz")
        beside.write_bytes(gzip.compress(blob))
        hand_back(beside)
        out[f"{key}_file"] = {"name": beside.name, "count": len(rows),
                              "sha256": hashlib.sha256(blob).hexdigest()[:16]}
    return out


# a reader asks the measurement for its rows and does not care which of the two files holds them
def rows_of(path: str | Path, key: str = "rows") -> list:
    path = Path(path)
    payload = json.loads(path.read_text())
    if key in payload:
        return payload[key]
    said = payload.get(f"{key}_file")
    if not said:
        raise FileNotFoundError(f"{path.name} carries neither {key} nor {key}_file")
    beside = path.with_name(said["name"])
    if not beside.exists():
        raise FileNotFoundError(
            f"{said['name']} is not here: the rows of {path.name} live beside it and are not in git"
        )
    return json.loads(gzip.decompress(beside.read_bytes()))


# the path is derived here and never taken from options: a job writing where its caller says
def record(kind: str, run_name: str, payload: dict, on: date | None = None, bulk=()) -> str:
    FOLDER.mkdir(parents=True, exist_ok=True)
    stamp = (on or date.today()).strftime("%Y%m%d")
    path = FOLDER / f"{_slug(kind)}_{_slug(run_name)}_{stamp}.json"
    # two passes of one probe in a day are two measurements, and the first was overwritten
    for nth in range(2, 100):
        if not path.exists():
            break
        path = FOLDER / f"{_slug(kind)}_{_slug(run_name)}_{stamp}_{nth}.json"
    store_json(path, _put_bulk(path, payload, bulk))
    return str(path)
