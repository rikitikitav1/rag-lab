import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import config
import job_queue
import logging_setup
import version
from engines import converter
from engines.converter_tools import KEPT_STATUSES, sha256
from evals import measurements
from models.corpus import DataSource, Stage
from orm.sync_db import Session
from sqlalchemy import select
from tool_names import Tool
from use_cases import fetch, raw_quality, route, source_intake

from .base import Final, register
from .card import take
from .converting import ROOT, ceiling, convert, converter_for, fields, load_settings, pieces

log = logging_setup.get_logger(__name__)

RAW = ROOT / "datasets" / "raw_sources"
# what the job fetches from outside; a folder placed by hand lives under datasets/inbox instead
FETCHED = RAW / "_fetched"
_SLUG = re.compile(r"[^\w.-]+")


# a file name for a key: its readable slug plus a hash of the key, so two keys never share a file
def _stem(key: str) -> str:
    return f"{_SLUG.sub('_', key)[:120]}-{fetch.short_hash(key)}"


def _write_json(path: Path, value) -> None:
    part = path.with_suffix(".part")
    part.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    os.replace(part, path)


def _pieces(pages: tuple[int, int] | None, size: int) -> list[tuple[int, int] | None]:
    return [None] if pages is None else pieces(pages[0], pages[1], size)


# one piece of a run through its engine: its markdown, and Docling's structure when it gave one
def _convert_piece(file: Path, engine: str | None, piece, settings: dict, language: str) -> dict:
    if engine is None:
        return {"markdown": file.read_text(errors="ignore"), "seconds": 0.0, "status": None, "structure": None}
    tool_settings, _ = settings[engine]
    spec = converter_for(engine)
    take(spec)
    tool_fields = fields(tool_settings, language)
    if engine == Tool.docling:
        tool_fields = tool_fields + [("to_formats", "json")]
    result = convert(spec, engine, file, tool_fields, piece, ceiling(tool_settings))
    reading = converter.reading(spec)[1]
    return {
        "markdown": result["markdown"] or "",
        "structure": result.get("structure"),
        "seconds": result["seconds"],
        "status": {"status": result["status"], "errors": result["errors"][:3]},
        "build": (reading.get("build") or {}).get("built_at"),
        "residency_started_at": reading.get("started_at"),
    }


def _key(rel: str, piece) -> str:
    return f"{rel}#{piece[0]}-{piece[1]}" if piece else rel


# a converted unit is kept on a resume when its file is the same and it came back, or needed no engine
def _kept(row: dict | None, sha: str) -> bool:
    if row is None or row.get("file_sha256") != sha:
        return False
    return row.get("piece_status") is None or row["piece_status"]["status"] in KEPT_STATUSES


def _unreadable_row(rel: str, sha: str, error: str) -> dict:
    return {
        "key": rel,
        "file": rel,
        "file_sha256": sha,
        "pages": None,
        "engine": None,
        "breached": ["file.unreadable"],
        "piece_status": {"status": "unreadable", "errors": [error]},
    }


# the share of the text in breaching rows; a row without words, as an empty or unreadable piece, weighs as the mean
def bad_share(rows: list[dict], key: str) -> float:
    weights = [r.get(key) or 0 for r in rows]
    known = [w for w in weights if w]
    weights = [w or (sum(known) / len(known) if known else 1) for w in weights]
    total = sum(weights)
    return round(sum(w for w, r in zip(weights, rows, strict=True) if r["breached"]) / total, 4) if total else 0.0


# bad when either the conversion's pieces or the chunker's chapters breach over the declared share of the text
def _verdict(pieces: list[dict], sections: list[dict]) -> tuple[str, dict, dict]:
    if not pieces:
        return "bad", {"no files": 1}, {}
    reasons = Counter(b for r in pieces + sections for b in r["breached"])
    shares = {"pieces": bad_share(pieces, "output_words"), "sections": bad_share(sections, "words")}
    if max(shares.values()) > config.settings.intake.quality.bad_share:
        return "bad", dict(reasons), shares
    return ("dirty" if reasons else "ok"), dict(reasons), shares


# a declared source to a raw folder: every file through the engine its route names, a report row a run, no index
@register("onboard_source")
def onboard_source(options: dict) -> None:
    with Session() as session:
        source = session.scalar(select(DataSource).where(DataSource.name == options["source"]))
        if source is None:
            raise Final(f"no source named {options['source']}")
        if refusal := source_intake.onboard_refusal(source):
            raise Final(refusal)
        session.expunge(source)
    names = {**config.settings.intake.settings, **(options.get("settings") or {})}
    settings = {tool: load_settings(name) for tool, name in names.items()}
    arm_hash = hashlib.sha256(json.dumps({t: s[1] for t, s in sorted(settings.items())}).encode()).hexdigest()[:8]
    folder = RAW / f"{source.name}_{arm_hash}"
    (folder / "files").mkdir(parents=True, exist_ok=True)
    (folder / "pieces").mkdir(exist_ok=True)
    root, files, fetched = source_intake.gather(source, FETCHED / source.name, ROOT)
    record_path = folder / "record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"units": {}}
    record.update({"source": source.name, "settings": names, "settings_sha256": {t: s[1] for t, s in settings.items()}})
    shas = {file: sha256(file) for file in files}
    # a unit is a piece of a run, so a bad stretch of a long book is named by its pages, not hidden in the whole
    units, unreadable = [], {}
    for file in files:
        rel = str(file.relative_to(root))
        try:
            runs = route.route(file)
        except route.Unreadable as error:
            unreadable[rel] = _unreadable_row(rel, shas[file], str(error))
            continue
        for run in runs:
            size = settings[run.engine][0]["pages_per_chunk"] if run.engine else 0
            units += [(file, rel, run, piece) for piece in _pieces(run.pages, size)]
    planned = {_key(rel, piece) for _, rel, _, piece in units}
    # a row whose file left, changed its route or its pieces is not this source any more
    record["units"] = {k: v for k, v in record["units"].items() if k in planned} | unreadable
    # one engine's units together: every switch between the two converters is a handover of the card
    units.sort(key=lambda u: u[2].engine or "")
    layers: dict[Path, list[str]] = {}
    for file, rel, run, piece in units:
        key = _key(rel, piece)
        # a piece whose markdown is gone from the folder, or was written under an older name, is converted again
        if _kept(record["units"].get(key), shas[file]) and (folder / "pieces" / f"{_stem(key)}.md").exists():
            continue
        # a cancel is read between pieces: a long book otherwise holds the queue and the card after it was called off
        if options.get("_job_id") is not None and job_queue.is_cancelled(options["_job_id"]):
            _write_json(record_path, record)
            log.info("onboard.cancelled", source=source.name, unit=key)
            return
        if file.suffix.lower() == ".pdf" and file not in layers:
            layers[file] = route.layer_texts(file)
        started = time.monotonic()
        done = _convert_piece(file, run.engine, piece, settings, source.language)
        stem = _stem(key)
        (folder / "pieces" / f"{stem}.md").write_text(done["markdown"])
        if done["structure"] is not None:
            (folder / "pieces" / f"{stem}.docling.json").write_text(json.dumps(done["structure"]))
        pages_text = layers.get(file)
        layer = "\n".join(pages_text[piece[0] - 1 : piece[1]]) if pages_text is not None and piece else None
        arm = {
            "engine": run.engine,
            "settings": names.get(run.engine) if run.engine else None,
            "settings_sha256": settings[run.engine][1] if run.engine else None,
            "build": done.get("build"),
            "residency_started_at": done.get("residency_started_at"),
        }
        unit = {"file": rel, "file_sha256": shas[file], "pages": list(piece) if piece else None}
        row = raw_quality.unit_row(done["markdown"], layer, unit, arm, done["seconds"])
        if done["status"] and done["status"]["status"] not in KEPT_STATUSES:
            row["breached"].append("conversion.failed")
        absorbed = [p for p in run.signals.get("absorbed") or [] if piece and piece[0] <= p <= piece[1]]
        row.update({"route": run.why, "absorbed": absorbed, "piece_status": done["status"], "key": key})
        record["units"][key] = row
        _write_json(record_path, record)
        log.info("onboard.unit", source=source.name, unit=key, seconds=round(time.monotonic() - started, 1))
    # a file's markdown whole, its pieces in page order, as the loader will read it
    sections: list[dict] = []
    for file in files:
        rel = str(file.relative_to(root))
        if rel in unreadable:
            continue
        mine = sorted((r for r in record["units"].values() if r["file"] == rel), key=lambda r: (r["pages"] or [0])[0])
        parts = [(folder / "pieces" / f"{_stem(r['key'])}.md").read_text() for r in mine]
        whole = "\n\n".join(parts)
        (folder / "files" / f"{_stem(rel)}.md").write_text(whole)
        # the chunker's gates read the file whole, as the index will cut it, a row a chapter
        sections += raw_quality.section_rows(whole, rel)
    rows = list(record["units"].values())
    verdict, reasons, shares = _verdict(rows, sections)
    pages_by_engine = Counter()
    builds: dict[str, set] = {}
    for r in rows:
        pages_by_engine[r["engine"] or "none"] += (r["pages"][1] - r["pages"][0] + 1) if r["pages"] else 1
        if r["engine"]:
            builds.setdefault(r["engine"], set()).add(r.get("build"))
    summary = {
        "schema": 2,
        "reads": (
            "rows: a piece of a file (a page range of a PDF, a whole page or file otherwise) through an engine, or "
            "none for markdown, with the conversion's signals; layer signals are judged for the engines in "
            "intake.quality.layer_band_engines. sections: a chapter of a file's whole markdown with the chunker's "
            "gates. The verdict is bad when the words in breaching pieces or chapters pass intake.quality.bad_share "
            "of the text, dirty with any breach, ok with none"
        ),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source.name,
        # the way back from a report to the job that wrote it and the code it ran
        "job_id": options.get("_job_id"),
        "code": version.mine(),
        "folder": str(folder.relative_to(ROOT)),
        "verdict": verdict,
        "reasons": reasons,
        "units": len(rows),
        "chapters": len(sections),
        "bad_share": shares,
        "pages_by_engine": dict(pages_by_engine),
        # a resume across a rebuild keeps the earlier rows, so a report can hold two builds of one engine
        "builds": {engine: sorted(b or "unstamped" for b in found) for engine, found in builds.items()},
        "settings": names,
        "signals": {k: {"better": v.better, "source": v.source} for k, v in raw_quality.SIGNALS.items()},
    }
    report = measurements.record(
        "raw_source", source.name, {**summary, "rows": rows, "sections": sections}, bulk=("rows", "sections")
    )
    provenance_path = folder / "provenance.json"
    earlier = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    fetched = {"fetched_at": earlier["fetched_at"], **fetched} if "fetched_at" in earlier else fetched
    provenance = {"origin": source.origin, **fetched, "settings": record["settings_sha256"], "files": len(files)}
    _write_json(provenance_path, provenance)
    with Session() as session:
        row = session.get(DataSource, source.id)
        row.stage = Stage.raw
        row.raw = {
            **summary,
            "report": str(Path(report).relative_to(measurements.ROOT)),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        session.commit()
    log.info("onboard.done", source=source.name, verdict=verdict, units=len(rows))
