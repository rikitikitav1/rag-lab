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
def onboard_refusal(source: DataSource, queued: int | None = None) -> str | None:
    if source.stage == Stage.accepted:
        return f"{source.name} is accepted; a raw conversion is for an added source"
    # two jobs on one source would write one record folder
    if queued:
        return f"{source.name} already has onboard job {queued} queued or running"
    return None


# a source is accepted from raw only; a bad verdict is accepted with a reason said, and the reason stays on the row
def accept_refusal(source: DataSource, reason: str | None, queued: int | None = None) -> str | None:
    if source.stage != Stage.raw:
        return f"{source.name} is {source.stage}; a source is accepted from raw"
    # an onboard finishing after the accept writes the row back to raw and drops the reason
    if queued:
        return f"{source.name} has onboard job {queued} queued or running; accept it after"
    if (source.raw or {}).get("verdict") == "bad" and not reason:
        return f"{source.name}'s raw verdict is bad; accepting it needs a reason"
    return None


def accepted_raw(source: DataSource, reason: str | None) -> dict:
    said = {"accepted_despite": reason} if reason else {}
    return {**(source.raw or {}), "accepted_at": datetime.now(UTC).isoformat(timespec="seconds"), **said}


# a row with no chunks of its own in search reads as a source that answered nothing
def active_refusal(source: DataSource, active: bool) -> str | None:
    if active and source.stage != Stage.accepted:
        return f"{source.name} is {source.stage}; only an accepted source goes into search"
    return None


def onboard_options(name: str, settings: dict | None) -> dict:
    return {"source": name, "settings": settings}


# the queued work that reads a source by name: removing it under a running job leaves the job writing into nothing
SOURCE_JOBS = ("onboard_source", "analyze_source", "index_data")
# an eval_run naming no variant runs on the searched one, which is refused anyway
VARIANT_JOBS = ("index_data", "analyze_source", "eval_run", "build_vector_index")


# a whole-corpus index: the bootstrap names it `all`, a caller may leave the source out
def whole_corpus_index() -> int | None:
    import job_queue

    return job_queue.pending_of_type("index_data", source=None) or job_queue.pending_of_type("index_data", source="all")


def removal_refusal(source: DataSource, queued: int | None, marking: int) -> str | None:
    if queued:
        return f"{source.name} has job {queued} queued or running"
    if marking:
        return f"{marking} questions have their gold in {source.name}; remove their sets first"
    return None


def variant_refusal(variant: str, live: str, queued: int | None) -> str | None:
    from use_cases.index import VARIANT_RE

    if not VARIANT_RE.fullmatch(variant):
        return f"{variant}: not a variant name"
    if variant == live:
        return f"{variant} is the corpus variant the stand searches"
    if queued:
        return f"{variant} has job {queued} queued or running"
    return None


# the stand's own files of a source, its raw conversions and what its job fetched; a folder origin is the owner's
def stand_files(source: DataSource, raw: Path) -> list[Path]:
    own = re.compile(rf"{re.escape(source.name)}_[0-9a-f]{{8}}")
    if not raw.is_dir():
        return []
    found = [raw / "_fetched" / source.name, *(p for p in raw.iterdir() if own.fullmatch(p.name))]
    return sorted(p for p in found if p.is_dir())


# one removal for both doors: refused while a job reads it or questions stand on it, then the row, chunks, own files
def remove_source(source: DataSource, raw: Path) -> dict:
    import shutil

    import job_queue

    import db

    # a whole-corpus index names no source and reads every file-defined one, so it counts as reading this one too
    queued = next((j for t in SOURCE_JOBS if (j := job_queue.pending_of_type(t, source=source.name))), None)
    queued = queued or whole_corpus_index()
    if refusal := removal_refusal(source, queued, db.questions_marking(source.id)):
        raise Final(refusal)
    own = stand_files(source, raw)
    chunks = db.remove_source(source.id)
    for folder in own:
        shutil.rmtree(folder)
    return {"source": source.name, "chunks": chunks, "folders": [p.name for p in own]}


def remove_variant(variant: str, live: str) -> dict:
    import job_queue

    import db

    queued = next((j for t in VARIANT_JOBS if (j := job_queue.pending_of_type(t, variant=variant))), None)
    queued = queued or veto_reading(variant)
    if refusal := variant_refusal(variant, live, queued):
        raise Final(refusal)
    return {"variant": variant, "chunks": db.remove_variant(variant)}


# a pending veto build that reads the variant, by name or through its defaults (no variants, no cut_from)
def veto_reading(variant: str) -> int | None:
    import job_queue
    from evals.build_veto import CUT_FROM

    found = job_queue.pending_listing("build_veto_set", "variants", variant)
    found = found or job_queue.pending_of_type("build_veto_set", cut_from=variant)
    if variant in ("baseline", CUT_FROM):
        found = found or job_queue.pending_of_type("build_veto_set", variants=None)
    if variant == CUT_FROM:
        found = found or job_queue.pending_of_type("build_veto_set", cut_from=None)
    return found


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


# the knobs a source sets for itself: its file's when it has one, the file winning over the row, else its row's
def intake_block(name: str, origin: dict | None) -> dict:
    found = files.source_files().get(name) if name else None
    if found is not None and found.intake is not None:
        return found.intake.model_dump(exclude_none=True)
    return (origin or {}).get("intake") or {}


# the stand's route rules and settings names with a source's own knobs over them
def intake_rule(block: dict) -> tuple:
    import config
    from config import RouteCfg

    route = config.settings.intake.route
    knobs = {k: v for k, v in block.items() if k != "settings" and v is not None}
    rule = RouteCfg(**{**route.model_dump(), **knobs})
    return rule, {**config.settings.intake.settings, **(block.get("settings") or {})}


# a source's own knobs go on its row, unless a job reads it now or a file speaks for it, the file winning over the row
def intake_refusal(source: DataSource, queued: int | None) -> str | None:
    if queued:
        return f"{source.name} has job {queued} queued or running"
    found = files.source_files().get(source.name)
    if found is not None:
        return f"{source.name} has a source file; its knobs go in the file's `intake:` block"
    return None


def with_intake(origin: dict | None, block: dict) -> dict:
    kept = {k: v for k, v in (origin or {}).items() if k != "intake"}
    return {**kept, "intake": block} if block else kept


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


# a folder origin read at the door as its job would read it, so a typo fails before the job's turn in the queue
def stand_folder(folder: str, stand: Path) -> tuple[Path, list[Path]]:
    root = (stand / folder).resolve()
    if stand.resolve() not in root.parents or not root.is_dir():
        raise Final(f"{folder}: not a folder of the stand")
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        raise Final(f"{folder}: a folder with no files")
    return root, files


# what a declaration names that the door can check now; the job's turn may be hours away in the queue
def declaration_refusal(declaration: Declaration, stand: Path) -> str | None:
    if declaration.folder:
        try:
            stand_folder(declaration.folder, stand)
        except Final as e:
            return str(e)
    return None


# the files of a declared source as the route will read them, fetched into its inbox when they come from outside
def gather(source: DataSource, inbox: Path, stand: Path) -> tuple[Path, list[Path], dict]:
    origin = source.origin or {}
    inbox.mkdir(parents=True, exist_ok=True)
    if "folder" in origin:
        return *stand_folder(origin["folder"], stand), {}
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


# every file under its report name, and what was left out; an EPUB stands for its chapters when the source reads them
def named_files(
    root: Path, files: list[Path], inbox: Path, skip=frozenset(), epub: bool = False
) -> tuple[dict[Path, str], dict[str, str]]:
    names, skipped = {}, {}
    for file in files:
        rel = str(file.relative_to(root))
        if file.suffix.lower() != ".epub":
            names[file] = rel
            continue
        if not epub:
            skipped[rel] = "EPUB reading is off until a second EPUB is measured (knob epub_chapters)"
            continue
        chapters, left_out = fetch.epub_chapters(file, inbox / "epub" / rel, skip)
        names |= {chapter: f"{rel}/{chapter.name}" for chapter in chapters}
        skipped |= {f"{rel}/{name}": why for name, why in left_out.items()}
    return names, skipped


# a fetch is stamped when something was fetched; a resume that found every file on disk keeps the earlier stamp
def _fetched(fresh: bool) -> dict:
    return {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds")} if fresh else {}
