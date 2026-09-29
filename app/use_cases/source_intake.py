import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import corpus_keys
import logging_setup
from errors import Final
from models.corpus import DataSource, Stage
from paths import FETCHED, RAW, ROOT
from sources import files
from sources.declaration import DEFAULT_INCLUDE, Declaration, GitFamily, site_of
from use_cases import fetch, route, site_page

log = logging_setup.get_logger(__name__)

_SLUG = re.compile(r"[^\w.-]+")
# a reason kept beside a bad verdict, bounded alike at every door
REASON_CHARS = (3, 500)
# outside the alphabet of a source name, so one source's folder never reads as another's
RAW_MARK = "@"


def onboard_refusal(source: DataSource, queued: int | None = None) -> str | None:
    # two jobs on one source would write one record folder
    if queued:
        return f"{source.name} already has onboard job {queued} queued or running"
    return None


# the run the owner is asked about: the one waiting beside the accepted run, else the row's own
def run_under_review(source: DataSource) -> dict:
    raw = source.raw or {}
    return raw.get("candidate") or raw


# a source is accepted from raw, or an accepted one with a new run waiting; a bad verdict needs a reason said
def accept_refusal(
    source: DataSource, reason: str | None, queued: int | None = None, indexing: int | None = None
) -> str | None:
    waiting = source.stage == Stage.accepted and (source.raw or {}).get("candidate")
    if source.stage != Stage.raw and not waiting:
        return f"{source.name} is {source.stage}; a source is accepted from raw or with a new run waiting"
    # an onboard finishing after the accept writes the row back to raw and drops the reason
    if queued:
        return f"{source.name} has onboard job {queued} queued or running; accept it after"
    # the accepted run's folder goes with the accept, and an index reading it would lose it midway
    if waiting and indexing:
        return f"{source.name} is read by index job {indexing}; accept its new run after"
    if run_under_review(source).get("verdict") == "bad" and not reason:
        return f"{source.name}'s raw verdict is bad; accepting it needs a reason"
    if reason is not None and not REASON_CHARS[0] <= len(reason.strip()) <= REASON_CHARS[1]:
        return f"a reason is {REASON_CHARS[0]} to {REASON_CHARS[1]} characters, kept on the row as said"
    return None


def accepted_raw(source: DataSource, reason: str | None) -> dict:
    said = {"accepted_despite": reason} if reason else {}
    return {**run_under_review(source), "accepted_at": datetime.now(UTC).isoformat(timespec="seconds"), **said}


# the waiting run, or the raw one, becomes the accepted one; the raw folder it replaced is named for after the commit
def promote(source: DataSource, reason: str | None, by: str | None = None) -> str | None:
    raw = source.raw or {}
    earlier = raw.get("folder") if raw.get("candidate") else None
    source.raw = {**accepted_raw(source, reason), **({"accepted_by": by} if by else {})}
    source.stage = Stage.accepted
    source.language = source.raw.get("language") or source.language
    # a variant cut from the run this one replaces reads as moved until it is cut again
    if source.indexed_with:
        source.indexed_with = {variant: files.RUN_REPLACED for variant in source.indexed_with}
    return earlier if earlier != source.raw.get("folder") else None


# a finished run onto its row: a raw source's own, or waiting beside the accepted one; ok and accepted if the stand says
def take_run(row: DataSource, run: dict) -> list[str]:
    import config

    gone = []
    if row.stage == Stage.accepted:
        # a run that waited unaccepted is replaced by this one, and its folder goes with it
        earlier = ((row.raw or {}).get("candidate") or {}).get("folder")
        if earlier not in (run["folder"], (row.raw or {}).get("folder")):
            gone.append(earlier)
        row.raw = {**(row.raw or {}), "candidate": run}
    else:
        row.stage = Stage.raw
        row.language = run["language"]
        row.raw = run
    # the door's refusals bar it, the onboard job that calls this aside
    if run["verdict"] == "ok" and config.settings.intake.quality.auto_accept_ok:
        indexing = index_waiting(row.name) if row.stage == Stage.accepted else None
        if not accept_refusal(row, None, None, indexing):
            gone.append(promote(row, None, by="auto"))
    return [folder for folder in gone if folder]


# a raw folder no row names any more, removed once the row that dropped it is committed and no index reads the source
def drop_folder(folder: str | None, source: str | None = None) -> None:
    if not folder or not (ROOT / folder).resolve().is_relative_to(RAW.resolve()):
        return
    # an index claimed between the accept and its commit reads the old folder; the removal door finds it later
    if source is not None and (indexing := index_waiting(source)):
        log.warning("intake.folder_kept", folder=folder, index_job=indexing)
        return
    shutil.rmtree(ROOT / folder, ignore_errors=True)


def index_waiting(name: str) -> int | None:
    import job_queue

    return job_queue.pending_of_type("index_data", source=name) or whole_corpus_index()


# a row with no chunks of its own in search reads as a source that answered nothing
def active_refusal(source: DataSource, active: bool) -> str | None:
    if active and source.stage != Stage.accepted:
        return f"{source.name} is {source.stage}; only an accepted source goes into search"
    return None


def _onboard_waiting(name: str) -> int | None:
    import job_queue

    return job_queue.pending_of_type("onboard_source", source=name)


# each transition checks and changes the row, or raises Final with why not; the door commits and answers
def accept(source: DataSource, reason: str | None) -> str | None:
    if refusal := accept_refusal(source, reason, _onboard_waiting(source.name), index_waiting(source.name)):
        raise Final(refusal)
    return promote(source, reason)


def set_active(source: DataSource, active: bool) -> None:
    if refusal := active_refusal(source, active):
        raise Final(refusal)
    source.active = active


def set_intake(source: DataSource, block: dict) -> None:
    if refusal := intake_refusal(source, _onboard_waiting(source.name)):
        raise Final(refusal)
    source.declaration = with_intake(source.declaration, block)


def check_onboard(source: DataSource) -> None:
    if refusal := onboard_refusal(source, _onboard_waiting(source.name)):
        raise Final(refusal)


def onboard_options(name: str, settings: dict | None) -> dict:
    return {"source": name, "settings": settings}


# the queued work that reads a source by name: removing it under a running job leaves the job writing into nothing
SOURCE_JOBS = ("onboard_source", "analyze_source", "index_data", "probe_intake")
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
    from corpus_keys import VARIANT_RE

    if not VARIANT_RE.fullmatch(variant):
        return f"{variant}: not a variant name"
    if variant == live:
        return f"{variant} is the corpus variant the stand searches"
    if queued:
        return f"{variant} has job {queued} queued or running"
    return None


# a run's folder: the source's name, then its hashes after a mark no source name can hold
def raw_folder(raw: Path, name: str, arm_hash: str, fingerprint: str | None = None) -> Path:
    marks = [arm_hash, *([fingerprint[:8]] if fingerprint else [])]
    older = raw / "_".join([name, *marks])
    # a run begun under the older underscore mark resumes where it lies; its row names it for removal
    return older if older.is_dir() else raw / RAW_MARK.join([name, *marks])


def is_raw_folder_of(name: str, folder: str) -> bool:
    return re.fullmatch(rf"{re.escape(name)}{RAW_MARK}[0-9a-f]{{8}}(?:{RAW_MARK}[0-9a-f]{{8}})?", folder) is not None


# the stand's own files of a source, its raw conversions and what its job fetched; a folder origin is the owner's
def stand_files(source: DataSource, raw: Path) -> list[Path]:
    if not raw.is_dir():
        return []
    run = source.raw or {}
    named = {Path(f).name for f in (run.get("folder"), (run.get("candidate") or {}).get("folder")) if f}
    found = [
        raw / FETCHED.name / source.name,
        *(p for p in raw.iterdir() if is_raw_folder_of(source.name, p.name) or p.name in named),
    ]
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
        "declaration": source.declaration,
        "chunks": chunks,
        "chunks_in_variant": in_variant,
        "ingest_quality": source.ingest_quality,
        "ingest_variant": source.ingest_variant,
        "ingest_checked_at": source.ingest_checked_at,
        # what the raw conversion said: its folder, verdict, reasons and the report's address
        "raw": source.raw or {},
        "raw_verdict": run_under_review(source).get("verdict"),
        # the variants cut by another version of this row's declaration
        "drift": files.drift(source.name, source.indexed_with, source.declaration),
    }


# the knobs a source sets for itself, in its declaration; the seed wrote a source file's there
def intake_block(declaration: dict | None) -> dict:
    return (declaration or {}).get("intake") or {}


# the stand's route rules and settings names with a source's own knobs over them
def intake_rule(block: dict) -> tuple:
    import config
    from config import RouteCfg

    route = config.settings.intake.route
    knobs = {k: v for k, v in block.items() if k != "settings" and v is not None}
    rule = RouteCfg(**{**route.model_dump(), **knobs})
    return rule, {**config.settings.intake.settings, **(block.get("settings") or {})}


# a source's own knobs go on its row, unless a job reads it now or the seed writes it from a file, which wins
def intake_refusal(source: DataSource, queued: int | None) -> str | None:
    if queued:
        return f"{source.name} has job {queued} queued or running"
    if source.seeded:
        return f"{source.name} has a source file; its knobs go in the file's `intake:` block"
    return None


def with_intake(declaration: dict | None, block: dict) -> dict:
    kept = {k: v for k, v in (declaration or {}).items() if k != "intake"}
    return {**kept, "intake": block} if block else kept


# the one way a declared source becomes a row, for the door and for the ops server alike
def declared_row(declaration: Declaration) -> DataSource:
    written = declaration.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    # nothing of it is searched before it is accepted
    return DataSource(
        **files.row_of(declaration, declaration.name), stage=Stage.declared, active=False, declaration=written
    )


# a url to a file under a folder of its own hash, so two index.html stay apart and keep their suffix for the route
def _download(url: str, folder: Path) -> tuple[Path, bool]:
    target = folder / corpus_keys.short_hash(url) / (_SLUG.sub("_", url.rstrip("/").rsplit("/", 1)[-1]) or "index.html")
    return target, fetch.download(url, target)


def _clone(git: dict, folder: Path) -> tuple[Path, dict]:
    if fetch.inside(folder, git.get("path")) is None:
        raise Final(f"{git.get('path')}: not a folder of the repository")
    # a repeated intake reads the upstream as it is now; the gold keeps its clone as it came, for a stable arm
    state = fetch.clone(git["repo"], folder, git.get("ref"), git.get("path"), update=True)
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
    if refusal := files.index_refusal(declaration):
        return refusal
    if declaration.folder:
        try:
            stand_folder(declaration.folder, stand)
        except Final as e:
            return str(e)
    return None


# the files of a declared source as the route will read them, fetched into its inbox when they come from outside
def gather(source: DataSource, inbox: Path, stand: Path) -> tuple[Path, list[Path], dict]:
    origin = source.declaration or {}
    inbox.mkdir(parents=True, exist_ok=True)
    if "git_family" in origin:
        family = GitFamily.model_validate(origin["git_family"])
        origin = {**origin, "git": {"repo": family.repo_of(source.name), "include": family.include}}
    if "folder" in origin:
        return *stand_folder(origin["folder"], stand), {}
    if "urls" in origin:
        got = [_download(url, inbox) for url in origin["urls"]]
        # an archive is the files inside it, as the gold's fetch reads it
        files = [p for f, _ in got for p in (fetch.unzip(f) if f.suffix.lower() == ".zip" else [f])]
        return inbox, files, _fetched(any(new for _, new in got))
    if "git" in origin:
        root, state = _clone(origin["git"], inbox / "repo")
        # a repeated intake reads the upstream anew, so a clone is always fetched now
        state = {**state, **_fetched(True)}
        found = {p.resolve() for pattern in origin["git"].get("include", DEFAULT_INCLUDE) for p in root.glob(pattern)}
        return root, sorted(p for p in found if p.is_file() and root in p.parents), state
    site = site_of(origin)
    release = site.release if site else None
    _drop_pages_of_another_release(inbox, release)
    files, fresh = [], False
    for url in origin["pages"]:
        page, new = _download(url, inbox / "pages")
        fresh |= new
        main = site_page.prepared(page.read_text(errors="ignore"), site.main, site.drop) if site else None
        if main is None:
            files.append(page)
            continue
        target = inbox / "main" / page.parent.name / page.with_suffix(".html").name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(main)
        files.append(target)
    if release:
        (inbox / "release").write_text(release)
    return inbox, files, _fetched(fresh)


# pages kept from another release, or from before one was declared, would take its name unread: they are fetched anew
def _drop_pages_of_another_release(inbox: Path, release: str | None) -> None:
    stamp = inbox / "release"
    if not release or (stamp.is_file() and stamp.read_text() == release):
        return
    for folder in ("pages", "main", "flat"):
        shutil.rmtree(inbox / folder, ignore_errors=True)


# an HTML file with its highlighting flat, beside the fetched one, for any HTML and not only a site's own element
def _flat(file: Path, inbox: Path, rel: str) -> Path:
    if file.suffix.lower() not in route.HTML:
        return file
    text = file.read_text(errors="ignore")
    flat = site_page.flat_pre(text)
    if flat == text:
        return file
    target = inbox / "flat" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(flat)
    return target


# every file under its report name, and what was left out; an EPUB stands for its chapters when the source reads them
def named_files(
    root: Path, files: list[Path], inbox: Path, skip=frozenset(), epub: bool = False, generated: list[str] = ()
) -> tuple[dict[Path, str], dict[str, str]]:
    names, skipped = {}, {}
    for file in files:
        rel = str(file.relative_to(root))
        page = generated and file.suffix.lower() in route.HTML
        if page and (built := site_page.generated_by(file.read_text(errors="ignore"), generated)):
            skipped[rel] = f"the site builds this page: {built}"
            continue
        if file.suffix.lower() != ".epub":
            names[_flat(file, inbox, rel)] = rel
            continue
        if not epub:
            skipped[rel] = "EPUB reading is off until a second EPUB is measured (knob epub_chapters)"
            continue
        chapters, left_out = fetch.epub_chapters(file, inbox / "epub" / rel, skip)
        names |= {_flat(chapter, inbox, f"{rel}/{chapter.name}"): f"{rel}/{chapter.name}" for chapter in chapters}
        skipped |= {f"{rel}/{name}": why for name, why in left_out.items()}
    return names, skipped


# a fetch is stamped when something was fetched; a resume that found every file on disk keeps the earlier stamp
def _fetched(fresh: bool) -> dict:
    return {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds")} if fresh else {}


# the rows whose declaration says they drift on their own
def drifting_rows() -> list[str]:
    from models.corpus import DataSource
    from orm.sync_db import Session
    from sqlalchemy import select

    with Session() as session:
        rows = session.execute(select(DataSource.name, DataSource.declaration)).all()
    return sorted(name for name, declaration in rows if (declaration or {}).get("drifts"))


# the source's language before any piece is read, since a scan's OCR is told it: declared, else from its own text
def source_language(source: DataSource, files: list[Path], layers: dict) -> str:
    # the row's column holds what an earlier run found; only the declaration says what was declared
    if declared := (source.declaration or {}).get("language"):
        return declared
    text = []
    for file in files:
        if file.suffix.lower() == ".pdf":
            # a file the reader cannot open is marked in the report, not a reason to stop before it
            try:
                layers.setdefault(file, route.layer_texts(file))
            except route.Unreadable:
                continue
            text += layers[file]
        elif file.suffix.lower() in route.MARKDOWN | route.HTML:
            text.append(file.read_text(errors="ignore"))
    joined = "".join(text)
    if not any(c.isalpha() for c in joined):
        raise Final(f"{source.name}: no text layer to read its language from; declare its language")
    return corpus_keys.language_by_alphabet(joined)
