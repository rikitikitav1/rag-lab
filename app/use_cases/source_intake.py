import re
from datetime import UTC, datetime
from pathlib import Path

from errors import Final
from models.corpus import DataSource, Stage
from sources import files
from sources.declaration import DEFAULT_INCLUDE, Declaration
from use_cases import fetch, site_page

# one vocabulary for the kind column: a folder is `local`, as the code-defined sources already say
KINDS = {"urls": "urls", "folder": "local", "git": "git", "pages": "pages"}
_SLUG = re.compile(r"[^\w.-]+")


# a raw conversion is for a source added by hand; the code-defined ones are accepted as they are
def onboard_refusal(source: DataSource) -> str | None:
    if source.stage == Stage.accepted:
        return f"{source.name} is accepted; a raw conversion is for an added source"
    return None


def onboard_options(name: str, settings: dict | None) -> dict:
    return {"source": name, "settings": settings}


def name_taken(name: str) -> str:
    return f"a source named {name} exists"


# one source whole, as every door shows it
def view(source: DataSource, chunks: int, in_variant: int) -> dict:
    return {
        "id": source.id,
        "name": source.name,
        "kind": source.kind,
        "stage": source.stage,
        "active": source.active,
        "language": source.language,
        "licence": source.licence,
        "origin": source.origin,
        "chunks": chunks,
        "chunks_in_variant": in_variant,
        "ingest_quality": source.ingest_quality,
        "ingest_variant": source.ingest_variant,
        "ingest_checked_at": source.ingest_checked_at,
        # what the raw conversion said: its folder, verdict, reasons and the report's address
        "raw": source.raw or {},
        "raw_verdict": (source.raw or {}).get("verdict"),
        # the file this row answers to, and the variants cut by another version of it
        "file": files.drift(source.name, source.indexed_with),
    }


# the one way a declared source becomes a row, for the door and for the ops server alike
def declared_row(declaration: Declaration) -> DataSource:
    origin = declaration.model_dump(exclude={"name", "language", "licence"}, exclude_none=True)
    return DataSource(
        name=declaration.name,
        kind=next(KINDS[k] for k in KINDS if k in origin),
        git_url=declaration.git.repo if declaration.git else None,
        stage=Stage.declared,
        # nothing of it is searched before it is accepted
        active=False,
        language=declaration.language,
        licence=declaration.licence,
        origin=origin,
    )


# a url to a file under a folder of its own hash, so two index.html stay apart and keep their suffix for the route
def _download(url: str, folder: Path) -> tuple[Path, bool]:
    target = folder / fetch.short_hash(url) / (_SLUG.sub("_", url.rstrip("/").rsplit("/", 1)[-1]) or "index.html")
    return target, fetch.download(url, target)


def _clone(git: dict, folder: Path) -> tuple[Path, dict]:
    if fetch.inside(folder, git.get("path")) is None:
        raise Final(f"{git.get('path')}: not a folder of the repository")
    state = fetch.clone(git["repo"], folder, git.get("ref"), git.get("path"))
    return fetch.inside(folder, git.get("path")), state


# the files of a declared source as the route will read them, fetched into its inbox when they come from outside
def gather(source: DataSource, inbox: Path, stand: Path) -> tuple[Path, list[Path], dict]:
    origin = source.origin or {}
    inbox.mkdir(parents=True, exist_ok=True)
    if "folder" in origin:
        root = (stand / origin["folder"]).resolve()
        if stand.resolve() not in root.parents or not root.is_dir():
            raise Final(f"{origin['folder']}: not a folder of the stand")
        return root, sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")), {}
    if "urls" in origin:
        got = [_download(url, inbox) for url in origin["urls"]]
        # an archive is the files inside it, as the gold's fetch reads it
        files = [p for f, _ in got for p in (fetch.unzip(f) if f.suffix.lower() == ".zip" else [f])]
        return inbox, files, _fetched(any(new for _, new in got))
    if "git" in origin:
        root, state = _clone(origin["git"], inbox / "repo")
        found = {p.resolve() for pattern in origin["git"].get("include", DEFAULT_INCLUDE) for p in root.glob(pattern)}
        return root, sorted(p for p in found if p.is_file() and root in p.parents), state
    site = origin.get("site") or {}
    files, fresh = [], False
    for url in origin["pages"]:
        page, new = _download(url, inbox / "pages")
        fresh |= new
        main = site_page.prepared(page.read_text(errors="ignore"), site["main"], site.get("drop", [])) if site else None
        if main is None:
            files.append(page)
            continue
        target = inbox / "main" / page.parent.name / page.with_suffix(".html").name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(main)
        files.append(target)
    return inbox, files, _fetched(fresh)


# a fetch is stamped when something was fetched; a resume that found every file on disk keeps the earlier stamp
def _fetched(fresh: bool) -> dict:
    return {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds")} if fresh else {}
