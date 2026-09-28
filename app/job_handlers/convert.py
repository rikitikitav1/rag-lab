import hashlib
import json
from pathlib import Path

import job_queue
import logging_setup
from engines import converter
from engines.converter_tools import KEPT_STATUSES, sha256, tool_version
from use_cases import route, source_intake

from . import reading
from .base import Final, register
from .card import take
from .converting import ROOT, SETTINGS, ceiling, code_lines_of, convert, converter_for, fields, load_settings, pieces

log = logging_setup.get_logger(__name__)

GOLD = ROOT / "datasets" / "converter_gold" / "files"
INBOX = ROOT / "datasets" / "inbox"
RUNS = GOLD / "runs"


# the image stamps what it copied in; the tree may have moved on since, as it did once unseen
def _build_differs(tool: str, build: dict | None) -> list[str] | str | None:
    # no stamp is the lagging image itself, never a clean one
    if not build:
        return "unstamped"
    here = {name: SETTINGS / name if (SETTINGS / name).is_file() else SETTINGS / tool / name for name in build["files"]}
    moved = [name for name, path in here.items() if not path.is_file() or build["files"][name] != sha256(path)]
    return moved or None


# a resumed run is the same configuration or it is not resumed: new settings, language or any rebuild is a new `out`
def _refuse_another_configuration(record: dict, now: dict) -> None:
    if not record.get("converted"):
        return
    for key in ("tool", "settings_sha256", "build", "language", "route_sha256", "pages_per_chunk", "knobs"):
        if record.get(key) != now[key]:
            raise Final(
                f"{key} was {record.get(key)!r} for what is converted here, now {now[key]!r}; "
                "a new configuration goes to a new out"
            )


def _pages(path: Path) -> int | None:
    from pypdf import PdfReader

    return len(PdfReader(path).pages) if path.suffix.lower() == ".pdf" else None


# a book goes in pieces of the whole file by page range, so the outline and its heading levels stay
def _chunks(pages: int | None, size: int) -> list[tuple[int, int] | None]:
    return pieces(1, pages, size) if pages else [None]


def _chunk_key(chunk: tuple[int, int] | None) -> str:
    return "all" if chunk is None else f"{chunk[0]}-{chunk[1]}"


def _origin(name: str) -> dict | None:
    from models.corpus import DataSource
    from orm.sync_db import Session
    from sqlalchemy import select

    with Session() as session:
        return session.scalar(select(DataSource.origin).where(DataSource.name == name))


# a source's knobs as its onboarding reads them, the run's `knobs` over them; `settings` is the arm's base, a default
def _input_rule(key: str, options: dict, tool: str) -> tuple[dict, object, dict]:
    sources = options.get("sources") or {}
    block = source_intake.intake_block(sources[key], _origin(sources[key])) if key in sources else {}
    block = {**block, **(options.get("knobs") or {})}
    rule, names = source_intake.intake_rule(block)
    return block, rule, {**names, tool: block.get("settings", {}).get(tool, options["settings"])}


# an intake arm's route over every input's own rule and settings file, so a source's moved knob is a new out too
def _arm_route_sha(options: dict, tool: str) -> str:
    each = {}
    for key in options["inputs"]:
        _, rule, names = _input_rule(key, options, tool)
        each[key] = [reading.route_sha(rule), names[tool]]
    return hashlib.sha256(json.dumps(each, sort_keys=True).encode()).hexdigest()[:12]


# the gold's inputs through the corpus's own reading path, a piece at a time and resumable, joined as the corpus joins
def _read_as_intake(options: dict, settings: dict, record: dict, record_path: Path, out: Path) -> None:
    loaded: dict = {}
    failed = []
    base = INBOX if options.get("root") == "inbox" else GOLD
    ranges = options.get("pages") or {}
    for path in [base / p for p in options["inputs"]]:
        key = str(path.relative_to(base))
        parts = out / (key.replace("/", "__") + ".parts")
        parts.mkdir(exist_ok=True)
        entry = record["converted"].setdefault(key, {"input_sha256": sha256(path), "pages": _pages(path), "chunks": {}})
        span = tuple(ranges[key]) if key in ranges else None
        entry["range"] = span
        block, rule, names = _input_rule(key, options, settings["tool"])
        layers = route.layer_texts(path) if path.suffix.lower() == ".pdf" else None
        entry["intake"] = block
        units = reading.plan(path, names, loaded, span, options.get("pages_per_chunk"), rule)
        for run, piece, name in units:
            chunk = _chunk_key(piece)
            if (parts / f"{chunk}.md").exists() and entry["chunks"].get(chunk, {}).get("status") in KEPT_STATUSES:
                continue
            if options.get("_job_id") is not None and job_queue.is_cancelled(options["_job_id"]):
                record["cancelled_before"] = f"{key} {chunk}"
                record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
                return
            layer = reading.layer_of(layers, piece)
            done, taken, reread = reading.read_piece(
                path, run.engine, piece, name, loaded, options["language"], layer, rule
            )
            (parts / f"{chunk}.md").write_text(done["markdown"])
            # the piece's own structure beside it, so a defect in a run is traced without converting again
            if done.get("structure") is not None:
                (parts / f"{chunk}.docling.json").write_text(json.dumps(done["structure"]))
            status = (done["status"] or {}).get("status", "success")
            if status not in KEPT_STATUSES:
                failed.append(f"{key} {chunk}: {status}")
            entry["chunks"][chunk] = {
                "status": status,
                "seconds": done["seconds"],
                "settings": taken,
                "reread": reread,
                "code_from_layer": done.get("code"),
            }
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        texts = [(parts / f"{_chunk_key(piece)}.md") for _, piece, _ in units]
        if all(p.exists() for p in texts):
            html = all(run.why == "html" for run, _, _ in units)
            whole, healed = reading.assemble([p.read_text() for p in texts], html, rule.html_one_title)
            entry["joins_healed"] = healed
            (out / (key.replace("/", "__") + ".md")).write_text(whole)
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    if failed:
        raise Final(f"{len(failed)} pieces failed, first: {failed[:3]}")


# resumable for free: an input whose markdown is already written is not converted again
@register("convert_source")
def convert_source(options: dict) -> None:
    settings, settings_sha256 = load_settings(options["settings"])
    base = INBOX if options.get("root") == "inbox" else GOLD
    inputs = [base / p for p in options["inputs"]]
    absent = [str(p.relative_to(base)) for p in inputs if not p.is_file()]
    if absent:
        raise Final(f"inputs not under {base.relative_to(ROOT)}: {absent[:3]}")
    spec = converter_for(settings["tool"])
    take(spec)
    state, supervisor = converter.reading(spec)
    if supervisor.get("tool") != settings["tool"]:
        raise Final(f"the converter runs {supervisor.get('tool')}, the settings are for {settings['tool']}")
    out = RUNS / options["out"]
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"converted": {}}
    now = {
        "tool": settings["tool"],
        "settings_sha256": settings_sha256,
        "build": supervisor.get("build"),
        "language": options["language"],
        # the plain arm runs the stand's code chain under the live rules too, so a rule moved between resumes is seen
        "route_sha256": _arm_route_sha(options, settings["tool"]) if options.get("intake") else reading.route_sha(),
        "pages_per_chunk": options.get("pages_per_chunk"),
        "knobs": options.get("knobs"),
    }
    _refuse_another_configuration(record, now)
    record.update(
        {
            **now,
            "settings": options["settings"],
            "build_differs": _build_differs(settings["tool"], supervisor.get("build")),
            "residency_started_at": supervisor.get("started_at"),
            "tool_version": tool_version(spec, settings["tool"]),
            "word_rules": bool(options.get("intake")),
        }
    )
    if options.get("intake"):
        _read_as_intake(options, settings, record, record_path, out)
        return
    tool_fields = fields(settings, options["language"])
    # a tool's piece bound is set from its smoke and lives beside its piece size
    bound = ceiling(settings)
    failed = []
    for path in inputs:
        key = str(path.relative_to(GOLD))
        target = out / (key.replace("/", "__") + ".md")
        parts = out / (key.replace("/", "__") + ".parts")
        parts.mkdir(exist_ok=True)
        pages = _pages(path)
        entry = record["converted"].setdefault(key, {"input_sha256": sha256(path), "pages": pages, "chunks": {}})
        chunks = _chunks(pages, settings["pages_per_chunk"])
        for chunk in chunks:
            name = _chunk_key(chunk)
            part = parts / f"{name}.md"
            if part.exists() and entry["chunks"].get(name, {}).get("status") in KEPT_STATUSES:
                continue
            # a cancel is read between pieces: a book of hours otherwise holds the queue after it was called off
            if options.get("_job_id") is not None and job_queue.is_cancelled(options["_job_id"]):
                record["cancelled_before"] = f"{key} {name}"
                record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
                log.info("convert.cancelled", input=key, chunk=name)
                return
            result = convert(spec, settings["tool"], path, tool_fields, chunk, bound)
            markdown, code = code_lines_of(settings, path, result) if result["markdown"] is not None else (None, None)
            result.pop("markdown")
            result.pop("structure", None)
            if code:
                result["code_from_layer"] = code
            if markdown is not None:
                part.write_text(markdown)
            if result["status"] not in KEPT_STATUSES:
                failed.append(f"{key} {name}: {result['status']} {result['errors'][:1]}")
            entry["chunks"][name] = {**result, "residency_started_at": converter.reading(spec)[1].get("started_at")}
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            log.info("convert.chunk", input=key, chunk=name, status=result["status"], seconds=result["seconds"])
        done = [parts / f"{_chunk_key(c)}.md" for c in chunks]
        if all(p.exists() for p in done):
            target.write_text("\n\n".join(p.read_text() for p in done))
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    # the pieces that failed are in the record with their errors; a rerun of the job retries only them
    if failed:
        raise Final(f"{len(failed)} pieces failed, first: {failed[:3]}")
