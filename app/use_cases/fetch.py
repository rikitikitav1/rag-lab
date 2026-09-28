import hashlib
import os
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import requests

TIMEOUT = 120


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


_SLUG = re.compile(r"[^\w.-]+")


# a file or piece key as a flat file name, readable and unique: two keys that slug alike differ by the hash
def file_stem(key: str) -> str:
    return f"{_SLUG.sub('_', key)[:120]}-{short_hash(key)}"


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


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_PAGE_TYPE = re.compile(r"<(?:body|section|div|nav|figure|header)\b[^>]*?(?:epub:type|data-type)=[\"']([^\"']+)[\"']")
# the publisher types its page near the top; deeper in, a chapter holds figures and notes typed on their own
_TYPE_HEAD = 4096


# the publisher's types of a page (cover, index, chapter), from every element near its top that names one
def page_types(html: str) -> set[str]:
    return {kind for found in _PAGE_TYPE.findall(html[:_TYPE_HEAD]) for kind in found.split()}


# an EPUB's chapters in reading order and its skipped pages; the package file stays out, it names a copy's buyer
def epub_chapters(epub: Path, folder: Path, skip: frozenset[str] = frozenset()) -> tuple[list[Path], dict[str, str]]:
    # a book of the owner's own store, and expat 2.6+ caps entity growth while ElementTree loads no external entity
    with zipfile.ZipFile(epub) as opened:
        container = ElementTree.fromstring(opened.read("META-INF/container.xml"))  # nosec B314
        package = next(e.get("full-path") for e in container.iter() if _local(e.tag) == "rootfile")
        opf = ElementTree.fromstring(opened.read(package))  # nosec B314
        base = PurePosixPath(package).parent
        manifest = {e.get("id"): e for e in opf.iter() if _local(e.tag) == "item"}
        spine = [manifest[e.get("idref")] for e in opf.iter() if _local(e.tag) == "itemref"]
        chapters = [str(base / e.get("href")) for e in spine if "html" in (e.get("media-type") or "")]
        folder.mkdir(parents=True, exist_ok=True)
        out, skipped = [], {}
        for n, member in enumerate(chapters, start=1):
            target = folder / f"{n:03d}_{PurePosixPath(member).stem}.html"
            page = opened.read(member)
            if kinds := page_types(page.decode(errors="ignore")) & skip:
                skipped[target.name] = f"epub page type {' '.join(sorted(kinds))}"
                continue
            target.write_bytes(page)
            out.append(target)
    return out, skipped


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
