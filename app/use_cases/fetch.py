import hashlib
import os
import subprocess
import zipfile
from pathlib import Path

import requests

TIMEOUT = 120


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# a url to a file, whole or not at all: it lands in a .part and moves into place; False when it was there already
def download(url: str, target: Path) -> bool:
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT) as got:
        got.raise_for_status()
        with part.open("wb") as out:
            for block in got.iter_content(1 << 20):
                out.write(block)
    os.replace(part, target)
    return True


# an archive's files beside it, in a folder of its stem
def unzip(archive: Path) -> list[Path]:
    folder = archive.with_suffix("")
    if not folder.exists():
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file())


def _git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


# blobless and sparse, so a docs folder of a large repository costs its own size only; `--` keeps the url a url
def clone(repo: str, folder: Path, ref: str | None = None, path: str | None = None, also=()) -> dict:
    if not (folder / ".git").exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        branch = ["--branch", ref] if ref else []
        _git("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", *branch, "--", repo, str(folder))
        if path:
            _git("sparse-checkout", "set", path, *also, cwd=folder)
        _git("checkout", cwd=folder)
    elif path and any(not (folder / extra).exists() for extra in also):
        _git("sparse-checkout", "add", *also, cwd=folder)
    return {
        "revision": _git("rev-parse", "HEAD", cwd=folder),
        "committed": _git("log", "-1", "--format=%cI", cwd=folder),
    }


# a folder inside another, never outside it by `..` or an absolute path
def inside(folder: Path, relative: str | None) -> Path | None:
    root = (folder / (relative or "")).resolve()
    return root if root == folder.resolve() or folder.resolve() in root.parents else None
