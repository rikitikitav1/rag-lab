import os
import shutil

import logging_setup

log = logging_setup.get_logger(__name__)

# what the machine must keep after a download: a full disk takes the database down with it
FLOOR_BYTES = 5 * 2**30


class NotEnoughDisk(RuntimeError):
    pass


def free_bytes(path: str = "/") -> int:
    return shutil.disk_usage(path).free


# the cache is the only place weights land, and it is not always on the same filesystem as the root
def cache_path() -> str:
    seen = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    while seen and not os.path.exists(seen):
        seen = os.path.dirname(seen)
    return seen or "/"


# the filesystem measured is the caller's, a proxy when the engine holds its own volume
def refuse_if_tight(needed: int | None, name: str, path: str | None = None) -> None:
    where = path or cache_path()
    free = free_bytes(where)
    if needed is None:
        # the size is not recorded, so this promises nothing and says so rather than implying room
        log.warning("disk.size_unknown", model=name, free_gib=round(free / 2**30, 1), path=where)
        return
    if free - needed < FLOOR_BYTES:
        raise NotEnoughDisk(
            f"{name} needs {needed / 2**30:.2f} GiB and only {free / 2**30:.2f} GiB is free"
            f" on {where}, keeping {FLOOR_BYTES / 2**30:.0f} GiB in reserve"
        )
    log.info(
        "disk.room",
        model=name,
        needed_gib=round(needed / 2**30, 2),
        free_gib=round(free / 2**30, 1),
        path=where,
    )
