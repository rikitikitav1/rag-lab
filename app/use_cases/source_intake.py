import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import gold_match
import logging_setup
from corpus_keys import HAS_GOLD_SQL, Gold, vector_index_name
from errors import Final, Refusal
from models.corpus import DataChunk, DataSource, Stage
from orm.sync_db import engine
from paths import FETCHED, RAW, ROOT
from sources import files
from sources.declaration import Declaration
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from use_cases.intake_fetch import stand_folder

log = logging_setup.get_logger(__name__)

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
    knobs = {"accepted_after_knobs": raw["knobs_tried"]} if raw.get("knobs_tried") else {}
    source.raw = {**accepted_raw(source, reason), **({"accepted_by": by} if by else {}), **knobs}
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
        tried = (row.raw or {}).get("knobs_tried")
        row.stage = Stage.raw
        row.language = run["language"]
        row.raw = {**run, **({"knobs_tried": tried} if tried else {})}
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
def accept(source: DataSource, reason: str | None, by: str | None = None) -> str | None:
    if refusal := accept_refusal(source, reason, _onboard_waiting(source.name), index_waiting(source.name)):
        raise Final(refusal)
    return promote(source, reason, by=by)


def set_active(source: DataSource, active: bool) -> None:
    if refusal := active_refusal(source, active):
        raise Final(refusal)
    source.active = active


def set_intake(source: DataSource, block: dict) -> None:
    if refusal := intake_refusal(source, _onboard_waiting(source.name)) or knob_refusal(source):
        raise Final(refusal)
    source.declaration = with_intake(source.declaration, block)
    _note_knob(source, {"intake": block})


def _knob_rounds(raw: dict) -> set:
    return {k.get("run") for k in raw.get("knobs_tried", [])}


# past the agent's rounds a person approves a new knob or refuses the source
def knob_refusal(source: DataSource) -> str | None:
    import config

    limit = config.settings.intake.quality.agent_knob_rounds
    raw = source.raw or {}
    rounds = _knob_rounds(raw)
    if raw.get("folder") not in rounds and len(rounds) >= limit:
        return f"{source.name} had {limit} rounds of knobs; a person approves a new knob or refuses the source"
    return None


# an ok the stand accepts after an agent turned knobs says so on the row: the gate went quiet, nobody looked
def _note_knob(source: DataSource, knob: dict) -> None:
    raw = source.raw or {}
    if not raw:
        return
    tried = [*raw.get("knobs_tried", []), {**knob, "run": raw.get("folder"), "verdict_before": raw.get("verdict")}]
    source.raw = {**raw, "knobs_tried": tried}


# the fields of a declaration a door may set in place; the rest go through the source file or a new declaration
KNOB_FIELDS = frozenset({"skip_paths", "markup", "markup_values", "section_root_by_path"})
# what the agent guessed when it declared, mended before the source is cut: a wrong language picks the wrong OCR
GUESSED_FIELDS = frozenset({"language", "licence", "categories"})
SETTABLE_FIELDS = KNOB_FIELDS | GUESSED_FIELDS


# declared fields set on the row for its next onboarding or index; an empty value clears one, the declaration checks all
def set_fields(source: DataSource, fields: dict) -> None:
    from pydantic import ValidationError

    if unknown := sorted(set(fields) - SETTABLE_FIELDS):
        raise Refusal("invalid", f"not settable in place: {unknown}; settable: {sorted(SETTABLE_FIELDS)}")
    # the fields move the cut, so a queued index of the source would cut by what the door is changing
    if queued := _onboard_waiting(source.name) or index_waiting(source.name):
        raise Final(f"{source.name} has job {queued} queued or running")
    if source.seeded:
        raise Final(f"{source.name} has a source file; set {sorted(fields)} there")
    knobs = {k: v for k, v in fields.items() if k in KNOB_FIELDS}
    if knobs and (refusal := knob_refusal(source)):
        raise Final(refusal)
    if set(fields) & GUESSED_FIELDS and source.indexed_with:
        raise Final(f"{source.name} is cut in {sorted(source.indexed_with)}; its language, licence and categories "
                    "are mended by removing it and declaring it again")
    declared = {k: v for k, v in (source.declaration or {}).items() if k not in fields}
    declared |= {k: v for k, v in fields.items() if v}
    try:
        checked = Declaration.model_validate(declared)
    except ValidationError as e:
        raise Refusal("invalid", str(e)) from e
    if refusal := files.index_refusal(checked):
        raise Refusal("invalid", refusal)
    source.declaration = declared
    source.language, source.licence = checked.language, checked.licence
    if knobs:
        _note_knob(source, {"fields": knobs})


def check_onboard(source: DataSource) -> None:
    if refusal := onboard_refusal(source, _onboard_waiting(source.name)):
        raise Final(refusal)


def onboard_options(name: str, settings: dict | None, fresh: bool = False) -> dict:
    return {"source": name, "settings": settings, **({"fresh": True} if fresh else {})}


# the queued work that reads a source by name: removing it under a running job leaves the job writing into nothing
SOURCE_JOBS = (
    "onboard_source", "analyze_source", "index_data", "probe_intake", "generate_questions", "load_questions",
    "accept_questions", "judge_questions", "anchor_questions", "reparse_questions",
)
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

    # a whole-corpus index names no source and reads every file-defined one, so it counts as reading this one too
    queued = next((j for t in SOURCE_JOBS if (j := job_queue.pending_of_type(t, source=source.name))), None)
    queued = queued or whole_corpus_index()
    if refusal := removal_refusal(source, queued, _questions_marking(source.id)):
        raise Final(refusal)
    own = stand_files(source, raw)
    chunks = _drop_source_row(source.id)
    for folder in own:
        shutil.rmtree(folder)
    return {"source": source.name, "chunks": chunks, "folders": [p.name for p in own]}


def remove_variant(variant: str, live: str) -> dict:
    import job_queue

    queued = next((j for t in VARIANT_JOBS if (j := job_queue.pending_of_type(t, variant=variant))), None)
    queued = queued or veto_reading(variant)
    if refusal := variant_refusal(variant, live, queued):
        raise Final(refusal)
    return {"variant": variant, "chunks": _drop_variant_rows(variant)}


# a pending veto build that reads the variant, by name or through its defaults (no variants, no cut_from)
def veto_reading(variant: str) -> int | None:
    import job_queue
    from corpus_keys import VETO_CUT_FROM

    found = job_queue.pending_listing("build_veto_set", "variants", variant)
    found = found or job_queue.pending_of_type("build_veto_set", cut_from=variant)
    if variant == VETO_CUT_FROM:
        found = found or job_queue.pending_of_type("build_veto_set", variants=None)
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
        "drift": files.drift(source.name, source.indexed_with, source.declaration, source.indexed_rules),
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


def _settings_sha(name: str) -> str | None:
    from use_cases.converting import load_settings

    try:
        return load_settings(name)[1]
    # a settings file gone since the run is that tool's edit, not a failed check
    except Final:
        return None


# a converted source against the stand now: the settings chosen, their files or the route moved since it was read
def conversion_drift(source: DataSource) -> dict | None:
    from use_cases import reading

    raw = source.raw or {}
    if not raw.get("folder") or raw.get("root_kind") == "tree":
        return None
    record_path = ROOT / raw["folder"] / "record.json"
    if not record_path.exists():
        return {"record": "missing"}
    record = json.loads(record_path.read_text())
    rule, names = intake_rule(intake_block(source.declaration))
    names = {**names, **(record.get("settings_override") or {})}
    was = record.get("settings") or {}
    chosen = sorted(t for t in was if names.get(t) != was[t])
    hashes = record.get("settings_sha256") or {}
    edited = sorted(t for t, name in was.items() if _settings_sha(name) != hashes.get(t))
    moved = {"chosen": chosen, "edited": edited, "route": record.get("route_sha256") != reading.route_sha(rule)}
    if moved["route"] and "route" in record:
        moved["route_knobs"] = files.changed_fields(record["route"], reading.route_rules(rule))
    return moved if chosen or edited or moved["route"] else {}


# the rows whose declaration says they drift on their own
def drifting_rows() -> list[str]:
    from models.corpus import DataSource
    from orm.sync_db import Session
    from sqlalchemy import select

    with Session() as session:
        rows = session.execute(select(DataSource.name, DataSource.declaration)).all()
    return sorted(name for name, declaration in rows if (declaration or {}).get("drifts"))


# the list's fields of one source, the same at the REST list and the MCP one
def line(source: DataSource, chunks: int, in_variant: int) -> dict:
    return {
        "id": source.id,
        "name": source.name,
        "kind": source.kind,
        "active": source.active,
        "chunks": chunks,
        "chunks_in_variant": in_variant,
        "ingest_quality": source.ingest_quality,
        "ingest_variant": source.ingest_variant,
        "ingest_checked_at": source.ingest_checked_at,
        "stage": source.stage,
        "language": source.language,
        "raw_verdict": run_under_review(source).get("verdict"),
    }


# a source's chunks, all and those of the variant it was last indexed as
def chunk_counts(session, source: DataSource) -> tuple[int, int]:
    from sqlalchemy import func, select

    count = select(func.count()).select_from(DataChunk).where(DataChunk.source_id == source.id)
    chunks = session.scalar(count) or 0
    in_variant = session.scalar(count.where(DataChunk.variant == source.ingest_variant)) if source.ingest_variant else 0
    return chunks, in_variant or 0


# a page of sources by name, each with its counts; `ingest_variant`, so two numbers side by side describe one cut
def listed(session, stage: Stage | None, limit: int, offset: int) -> list[dict]:
    from sqlalchemy import func, select

    stmt = select(DataSource).order_by(DataSource.name).limit(limit).offset(offset)
    if stage is not None:
        stmt = stmt.where(DataSource.stage == stage)
    found = list(session.scalars(stmt))
    ids = [s.id for s in found]
    per_variant = {
        (source_id, variant): n
        for source_id, variant, n in session.execute(
            select(DataChunk.source_id, DataChunk.variant, func.count())
            .where(DataChunk.source_id.in_(ids))
            .group_by(DataChunk.source_id, DataChunk.variant)
        )
    }
    totals: dict[int, int] = {}
    for (source_id, _), n in per_variant.items():
        totals[source_id] = totals.get(source_id, 0) + n
    return [line(s, totals.get(s.id, 0), per_variant.get((s.id, s.ingest_variant), 0)) for s in found]


# a declared row written for both doors; two declarations of one name at once meet the unique name at commit
def declare(session, declaration: Declaration) -> DataSource:
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    if refusal := declaration_refusal(declaration, ROOT):
        raise Refusal("malformed", refusal)
    if session.scalar(select(DataSource.id).where(DataSource.name == declaration.name)):
        raise Refusal("taken", name_taken(declaration.name))
    source = declared_row(declaration)
    session.add(source)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise Refusal("taken", name_taken(declaration.name)) from e
    session.refresh(source)
    return source


def _drop_variant(conn, variant) -> int:
    dropped = conn.execute(text("DELETE FROM data_chunks WHERE variant = :variant"), {"variant": variant}).rowcount
    # an empty partial index left behind makes the next index of the name insert row by row
    conn.execute(text(f"DROP INDEX IF EXISTS {vector_index_name(variant)}"))
    # a row kept by another variant must not say this one was cut by some file
    conn.execute(
        text(
            "UPDATE data_sources SET indexed_with = indexed_with - :variant, indexed_rules = indexed_rules - :variant"
        ),
        {"variant": variant},
    )
    return dropped


def _drop_variant_rows(variant) -> int:
    with engine.begin() as conn:
        return _drop_variant(conn, variant)


# the row and its chunks in every variant; the chunks go by the foreign key's cascade
def _drop_source_row(source_id: int) -> int:
    try:
        with engine.begin() as conn:
            chunks = conn.execute(
                text("SELECT count(*) FROM data_chunks WHERE source_id = :id"), {"id": source_id}
            ).scalar()
            conn.execute(text("DELETE FROM data_sources WHERE id = :id"), {"id": source_id})
    except IntegrityError as e:
        raise Final(f"source {source_id} was taken by another row while it was being removed; nothing removed") from e
    return chunks


# questions whose gold lies in the source, by the stand's own predicate: a mark is a file or a folder prefix
def _questions_marking(source_id: int) -> int:
    with engine.connect() as conn:
        files = (
            conn.execute(text("SELECT DISTINCT source FROM data_chunks WHERE source_id = :id"), {"id": source_id})
            .scalars()
            .all()
        )
        if not files:
            return 0
        rows = conn.execute(
            text(f"SELECT marked_sources, gold FROM questions q WHERE {HAS_GOLD_SQL.format(q='q')}")
        ).all()
    return gold_match.count_marking(files, [Gold.of(marks, gold) for marks, gold in rows])
