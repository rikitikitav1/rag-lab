import json
from pathlib import Path

import job_queue
import logging_setup
from engines import converter
from engines.converter_tools import KEPT_STATUSES, sha256, tool_version

from .base import Final, register
from .card import take
from .converting import ROOT, SETTINGS, ceiling, convert, converter_for, fields, load_settings, pieces

log = logging_setup.get_logger(__name__)

GOLD = ROOT / "datasets" / "converter_gold" / "files"
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
    for key in ("tool", "settings_sha256", "build", "language"):
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


# resumable for free: an input whose markdown is already written is not converted again
@register("convert_source")
def convert_source(options: dict) -> None:
    settings, settings_sha256 = load_settings(options["settings"])
    inputs = [GOLD / p for p in options["inputs"]]
    absent = [str(p.relative_to(GOLD)) for p in inputs if not p.is_file()]
    if absent:
        raise Final(f"inputs not in the gold: {absent[:3]}")
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
    }
    _refuse_another_configuration(record, now)
    record.update(
        {
            **now,
            "settings": options["settings"],
            "build_differs": _build_differs(settings["tool"], supervisor.get("build")),
            "residency_started_at": supervisor.get("started_at"),
            "tool_version": tool_version(spec, settings["tool"]),
        }
    )
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
            markdown = result.pop("markdown")
            result.pop("structure", None)
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
