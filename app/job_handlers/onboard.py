import hashlib
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import config
import corpus_keys
import job_queue
import logging_setup
import version
from engines.converter_tools import KEPT_STATUSES, sha256
from evals import measurements
from models.corpus import DataSource, Stage
from orm.sync_db import Session
from paths import FETCHED, RAW, ROOT
from sources.declaration import site_of
from sqlalchemy import select
from use_cases import raw_quality, route, source_intake

from . import reading
from .base import Final, register
from .converting import load_settings, pieces

log = logging_setup.get_logger(__name__)



_stem = corpus_keys.file_stem


def _write_json(path: Path, value) -> None:
    part = path.with_suffix(".part")
    part.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    os.replace(part, path)


def _key(rel: str, piece) -> str:
    return f"{rel}#{piece[0]}-{piece[1]}" if piece else rel


# a converted unit is kept on a resume when its file, its planned settings and the route are the same and it came back
def _kept(row: dict | None, sha: str, settings_sha: str | None, same_route: bool) -> bool:
    if row is None or not same_route or row.get("file_sha256") != sha:
        return False
    if row.get("planned_settings_sha256", row.get("settings_sha256")) != settings_sha:
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


# every unit of every file: a file no engine reads is skipped, one that cannot be opened is a marked row
def _plan(files: list[Path], named: dict, shas: dict, names: dict, loaded: dict, skipped: dict, rule) -> tuple:
    units, unreadable = [], {}
    for file in files:
        rel = named[file]
        try:
            planned = reading.plan(file, names, loaded, rule=rule)
        except route.Unsupported as error:
            skipped[rel] = str(error)
            continue
        except route.Unreadable as error:
            unreadable[rel] = _unreadable_row(rel, shas[file], str(error))
            continue
        units += [(file, rel, run, piece, name) for run, piece, name in planned]
    return units, unreadable


# one unit read and written to the folder; its row carries the reading taken and what the rules did
def _read_unit(unit, folder: Path, shas: dict, loaded: dict, language: str, layers: dict, rule) -> dict:
    file, rel, run, piece, planned = unit
    key = _key(rel, piece)
    if file.suffix.lower() == ".pdf" and file not in layers:
        layers[file] = route.layer_texts(file)
    layer = reading.layer_of(layers.get(file), piece)
    done, name, reread = reading.read_piece(file, run.engine, piece, planned, loaded, language, layer, rule)
    stem = _stem(key)
    (folder / "pieces" / f"{stem}.md").write_text(done["markdown"])
    if done["structure"] is not None:
        (folder / "pieces" / f"{stem}.docling.json").write_text(json.dumps(done["structure"]))
    arm = {
        "engine": run.engine,
        "settings": name,
        "settings_sha256": loaded[name][1] if name else None,
        "build": done.get("build"),
        "residency_started_at": done.get("residency_started_at"),
    }
    unit_keys = {"file": rel, "file_sha256": shas[file], "pages": list(piece) if piece else None}
    row = raw_quality.unit_row(done["markdown"], layer, unit_keys, arm, done["seconds"])
    row["planned_settings_sha256"] = loaded[planned][1] if planned else None
    if reread:
        row["reread"] = reread
    if done["status"] and done["status"]["status"] not in KEPT_STATUSES:
        row["breached"].append("conversion.failed")
    # kept, but said: none seen yet, and the first one should show in the report rather than pass as a success
    if done["status"] and done["status"]["status"] == "partial_success":
        row["breached"].append("conversion.partial")
    absorbed = [p for p in run.signals.get("absorbed") or [] if piece and piece[0] <= p <= piece[1]]
    row.update({"route": run.why, "absorbed": absorbed, "piece_status": done["status"], "key": key})
    if done.get("code"):
        row["code_from_layer"] = done["code"]
    return row


# each file's markdown whole, its pieces in page order as the loader will read it, and the chunker's rows a chapter
def _assemble(files: list[Path], named: dict, record: dict, folder: Path, left: set, rule, layers: dict) -> tuple:
    sections: list[dict] = []
    joins = Counter({"fences": 0, "tables": 0, "headings": 0, "running_heads": 0})
    check: Counter = Counter()
    for file in files:
        rel = named[file]
        if rel in left:
            continue
        mine = sorted((r for r in record["units"].values() if r["file"] == rel), key=lambda r: (r["pages"] or [0])[0])
        parts = [(folder / "pieces" / f"{_stem(r['key'])}.md").read_text() for r in mine]
        if file.suffix.lower() == ".pdf" and file not in layers:
            layers[file] = route.layer_texts(file)
        whole, healed = reading.whole_file(parts, [r.get("route") for r in mine], rule, layers.get(file))
        joins.update(healed)
        (folder / "files" / f"{_stem(rel)}.md").write_text(whole)
        sections += raw_quality.section_rows(whole, rel)
        check.update(raw_quality.self_check(whole, route.outline_titles(file)))
    return sections, joins, check


# piece ends the seam rule moved off a page break that code runs over, against a plain cut of the same run
def _seams_moved(units: list, loaded: dict) -> int:
    runs: dict = {}
    for file, _, run, piece, name in units:
        if piece and name:
            runs.setdefault((file, run.pages, name), []).append(piece)
    plain = {key: pieces(*key[1], loaded[key[2]][0]["pages_per_chunk"]) for key in runs}
    return sum(len({end for _, end in cut} - {end for _, end in plain[key]}) for key, cut in runs.items())


_CODE_RULES = (
    "rebuilt",
    "kept",
    "joined",
    "tables_joined",
    "fenced",
    "relevelled",
    "numbered_levels",
    "paragraphs_joined",
    "entities_decoded",
    "pipes_dropped",
    "rows_run_on",
    "rows_once",
    "dashes_restored",
    "words_joined",
    "bullets_unescaped",
    "underscores_unescaped",
    "pictures_addressed",
    "formulas_from_layer",
    "captions_demoted",
    "running_heads_dropped",
    "split_words_joined",
    "duplicates_dropped",
)


def _unlisted(code: list[dict]) -> list[str]:
    return sorted({k for c in code for k in c} - set(_CODE_RULES))


# what the rules did across the source, so a book where one never fired reads apart from one where it did
def _rules_fired(rows: list[dict]) -> dict:
    rereads = [r["reread"] for r in rows if r.get("reread")]
    code = [r["code_from_layer"] for r in rows if r.get("code_from_layer")]
    return {
        "rereads": {
            "tried": len(rereads),
            # a splice keeps the second reading's prose, so it counts with the readings taken
            "taken": sum(bool(r["taken"] or r.get("tables_spliced")) for r in rereads),
            "tables_spliced": sum(r.get("tables_spliced", 0) for r in rereads),
            "formulas_spliced": sum(r.get("formulas_spliced", 0) for r in rereads),
        },
        # the known rules first, then any counter a writer added and this list does not name yet
        "code_from_layer": {k: sum(c.get(k, 0) for c in code) for k in [*_CODE_RULES, *_unlisted(code)]},
    }




def _settings_sha(name: str) -> str | None:
    try:
        return load_settings(name)[1]
    # a settings file gone since the run is that tool's edit, not a failed check
    except Final:
        return None


# a converted source against the stand now: the settings chosen, their files or the route moved since it was read
def conversion_drift(source: DataSource) -> dict | None:
    raw = source.raw or {}
    if not raw.get("folder") or raw.get("root_kind") == "tree":
        return None
    record_path = ROOT / raw["folder"] / "record.json"
    if not record_path.exists():
        return {"record": "missing"}
    record = json.loads(record_path.read_text())
    rule, names = source_intake.intake_rule(source_intake.intake_block(source.declaration))
    names = {**names, **(record.get("settings_override") or {})}
    was = record.get("settings") or {}
    chosen = sorted(t for t in was if names.get(t) != was[t])
    hashes = record.get("settings_sha256") or {}
    edited = sorted(t for t, name in was.items() if _settings_sha(name) != hashes.get(t))
    moved = {"chosen": chosen, "edited": edited, "route": record.get("route_sha256") != reading.route_sha(rule)}
    return moved if chosen or edited or moved["route"] else {}


def _fingerprint(arm_hash: str, route_sha: str, file_shas: dict) -> str:
    return hashlib.sha256(json.dumps([arm_hash, route_sha, sorted(file_shas.items())]).encode()).hexdigest()


# where the index reads a source: its own tree when every file is markdown, else the raw folder a converter wrote
def _index_root(units: list, tree: Path, folder: Path) -> tuple[str, str]:
    kind = "tree" if units and all(run.engine is None for _, _, run, _, _ in units) else "converted"
    root = tree if kind == "tree" else folder
    return (str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root)), kind


# a declared source to a raw folder: every file through the engine its route names, a report row a run, no index
@register("onboard_source")
def onboard_source(options: dict) -> dict | None:
    with Session() as session:
        source = session.scalar(select(DataSource).where(DataSource.name == options["source"]))
        if source is None:
            raise Final(f"no source named {options['source']}")
        if refusal := source_intake.onboard_refusal(source):
            raise Final(refusal)
        session.expunge(source)
    # the job's own settings over the source's knobs over the stand's; the knobs from its file, else its row
    rule, names = source_intake.intake_rule(source_intake.intake_block(source.declaration))
    names = {**names, **(options.get("settings") or {})}
    settings = {tool: load_settings(name) for tool, name in names.items()}
    arm_hash = hashlib.sha256(json.dumps({t: s[1] for t, s in sorted(settings.items())}).encode()).hexdigest()[:8]
    # every settings file a run names is loaded once, keyed by its own name rather than its tool
    loaded = {name: settings[tool] for tool, name in names.items()}
    route_sha = reading.route_sha(rule)
    origin = source.declaration or {}
    root, gathered, fetched = source_intake.gather(source, FETCHED / source.name, ROOT)
    # what the gather says it fetched now, a folder read from the tree as it lies: unchanged speaks for that
    refetched = "folder" in origin or "fetched_at" in fetched
    epub_skip = frozenset(rule.epub_skip)
    site = site_of(origin)
    generated = site.generated if site else []
    named, left_out = source_intake.named_files(
        root, gathered, FETCHED / source.name, epub_skip, rule.epub_chapters, generated
    )
    files = list(named)
    shas = {file: sha256(file) for file in files}
    fingerprint = _fingerprint(arm_hash, route_sha, {named[f]: sha for f, sha in shas.items()})
    accepted = source.stage == Stage.accepted
    # the files, settings and route of the accepted run: nothing to read again
    if accepted and (source.raw or {}).get("fingerprint") == fingerprint:
        log.info("onboard.unchanged", source=source.name)
        return {"unchanged": True, "refetched": refetched}
    # a new run of an accepted source goes beside the accepted folder, which search keeps reading until it is accepted
    folder = source_intake.raw_folder(RAW, source.name, arm_hash, fingerprint if accepted else None)
    (folder / "files").mkdir(parents=True, exist_ok=True)
    (folder / "pieces").mkdir(exist_ok=True)
    record_path = folder / "record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"units": {}}
    same_route = record.get("route_sha256") == route_sha
    record.update(
        {
            "source": source.name,
            "settings": names,
            # what the job itself chose over the source's knobs, so the drift check does not read it as a move
            "settings_override": options.get("settings") or {},
            "settings_sha256": {t: s[1] for t, s in settings.items()},
            "route_sha256": route_sha,
        }
    )
    # a unit is a piece of a run, so a bad stretch of a long book is named by its pages, not hidden in the whole
    skipped = dict(left_out)
    units, unreadable = _plan(files, named, shas, names, loaded, skipped, rule)
    planned = {_key(rel, piece) for _, rel, _, piece, _ in units}
    # a row whose file left, changed its route or its pieces is not this source any more
    record["units"] = {k: v for k, v in record["units"].items() if k in planned} | unreadable
    # one engine's units together: every switch between the two converters is a handover of the card
    units.sort(key=lambda u: u[2].engine or "")
    layers: dict[Path, list[str]] = {}
    language = source_intake.source_language(source, files, layers)
    for unit in units:
        file, rel, _, piece, name = unit
        key = _key(rel, piece)
        # a piece whose markdown is gone from the folder, or was written under an older name, is converted again
        kept = _kept(record["units"].get(key), shas[file], loaded[name][1] if name else None, same_route)
        if kept and (folder / "pieces" / f"{_stem(key)}.md").exists():
            continue
        # a cancel is read between pieces: a long book otherwise holds the queue and the card after it was called off
        if options.get("_job_id") is not None and job_queue.is_cancelled(options["_job_id"]):
            _write_json(record_path, record)
            log.info("onboard.cancelled", source=source.name, unit=key)
            return
        started = time.monotonic()
        record["units"][key] = _read_unit(unit, folder, shas, loaded, language, layers, rule)
        _write_json(record_path, record)
        log.info("onboard.unit", source=source.name, unit=key, seconds=round(time.monotonic() - started, 1))
    sections, joins, check = _assemble(files, named, record, folder, set(unreadable) | set(skipped), rule, layers)
    index_root, root_kind = _index_root(units, root, folder)
    rows = list(record["units"].values())
    verdict, reasons, shares = _verdict(rows, sections)
    pages_by_engine, pages_by_settings = Counter(), Counter()
    builds: dict[str, set] = {}
    for r in rows:
        pages = (r["pages"][1] - r["pages"][0] + 1) if r["pages"] else 1
        pages_by_engine[r["engine"] or "none"] += pages
        pages_by_settings[r.get("settings") or "none"] += pages
        if r["engine"]:
            builds.setdefault(r["engine"], set()).add(r.get("build"))
    provenance_path = folder / "provenance.json"
    earlier = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    fetched = {"fetched_at": earlier["fetched_at"], **fetched} if "fetched_at" in earlier else fetched
    # a site's pages are keyed by the product's release; the fetch date stands beside it as a label
    release = site.release if site else None
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
        # where the index reads it: its own tree (markdown) or this raw folder (a converter's output)
        "root": index_root,
        "root_kind": root_kind,
        "language": language,
        "fingerprint": fingerprint,
        "verdict": verdict,
        "reasons": reasons,
        "units": len(rows),
        # files no engine reads, by suffix; they weigh nothing in the verdict
        "skipped": skipped,
        "chapters": len(sections),
        # chapters that are a file's root alone, where the coverage gate is not read
        "coverage_by_shape": sum(1 for s in sections if s["coverage_by_shape"]),
        "bad_share": shares,
        "pages_by_engine": dict(pages_by_engine),
        "pages_by_settings": dict(pages_by_settings),
        # code blocks and tables a page break cut in two, joined again, and HTML section headings set one level down
        "joins_healed": dict(joins),
        "rules_fired": _rules_fired(rows),
        # the source read against itself with no gold: outline titles found as headings, fences, one-line code, seams
        "self_check": {**dict(check), "seams_moved": _seams_moved(units, loaded)},
        # a resume across a rebuild keeps the earlier rows, so a report can hold two builds of one engine
        "builds": {engine: sorted(b or "unstamped" for b in found) for engine, found in builds.items()},
        "settings": names,
        # the knobs this source set for itself over the stand's
        "intake": source_intake.intake_block(source.declaration),
        "version": {"release": release, "fetched_at": fetched.get("fetched_at")} if release else None,
        "signals": {k: {"better": v.better, "source": v.source} for k, v in raw_quality.SIGNALS.items()},
    }
    report = measurements.record(
        "raw_source", source.name, {**summary, "rows": rows, "sections": sections}, bulk=("rows", "sections")
    )
    provenance = {
        "declaration": source.declaration, **fetched, "settings": record["settings_sha256"], "files": len(files)
    }
    _write_json(provenance_path, provenance)
    with Session() as session:
        row = session.get(DataSource, source.id)
        run = {
            **summary,
            "report": str(Path(report).relative_to(measurements.ROOT)),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        gone = source_intake.take_run(row, run)
        session.commit()
    for folder in gone:
        source_intake.drop_folder(folder, source.name)
    log.info("onboard.done", source=source.name, verdict=verdict, units=len(rows))
    return {"verdict": verdict, "candidate": accepted}
