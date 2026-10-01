import contextlib
import contextvars
import functools
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import config
import engines
import logging_setup
from engines import converter
from engines.converter_tools import PIECE, sha256
from errors import Final
from models.registry import EngineKind
from paths import READINGS, ROOT
from tool_names import Tool

log = logging_setup.get_logger(__name__)

SETTINGS = ROOT / "converters"
# a piece of fifty pages takes minutes; this long without a result is a hang, unless the settings bound it closer
CHUNK_CEILING = 1800


# a settings file by its name, with the hash that stands for it in every record
def load_settings(name: str) -> tuple[dict, str]:
    tool, file = name.split("/")
    path = SETTINGS / tool / "settings" / f"{file}.json"
    if not path.is_file():
        raise Final(f"no settings file {path.relative_to(ROOT)}")
    settings = json.loads(path.read_text())
    if settings["tool"] not in PIECE:
        raise Final(f"no adapter for the converter {settings['tool']}")
    return settings, sha256(path)


def fields(settings: dict, language: str) -> list[tuple[str, str]]:
    found = list(settings["fields"].items())
    found += [("ocr_lang", lang) for lang in settings.get("ocr_lang", {}).get(language, [])]
    # Docling's own structure rides along: the code items' boxes for the rebuild, and the pieces keep it
    if settings["tool"] == Tool.docling and ("to_formats", "json") not in found:
        found.append(("to_formats", "json"))
    return found


# the markdown with its code blocks' lines from the PDF's layer when the settings ask for it, and what was rebuilt
def code_lines_of(settings: dict, path: Path, result: dict, rule) -> tuple[str, dict | None]:
    from use_cases import code_lines, docling_structure, markdown_cleanup, route

    spread = rule.mono_spread

    markdown = result.get("markdown") or ""
    if not settings.get("code_from_layer") or path.suffix.lower() != ".pdf" or not result.get("structure"):
        return markdown, None
    from use_cases import heading_rules, piece_join

    markdown, entities = markdown_cleanup.decode_entities(markdown) if rule.decode_entities else (markdown, 0)
    markdown, pipes = markdown_cleanup.drop_lone_pipes(markdown) if rule.drop_lone_pipes else (markdown, 0)
    rows_by = frozenset(rule.code_row_rules)
    callouts = route.mono_names(rule) if rule.listing_callouts else None
    markdown, counts = code_lines.rebuild(
        markdown, result["structure"], path, spread, callouts, rule.mono_by_step, rows_by
    )
    counts.update(entities_decoded=entities, pipes_dropped=pipes)
    goes_on = docling_structure.continued(result["structure"])["table"]
    markdown, counts["tables_joined"], counts["rows_continued"] = piece_join.join_tables(
        markdown, goes_on, rule.join_continued_rows
    )
    outline = route.outline(path)
    titles = [title for _, title, _ in outline]
    markdown, counts["fenced"] = code_lines.fence_mono(
        markdown, result["structure"], path, route.mono_names(rule), titles, spread=spread, rows_by=rows_by
    )
    counts["relevelled"] = counts["numbered_levels"] = 0
    if outline and rule.outline_levels:
        markdown, counts["relevelled"] = heading_rules.relevel(
            markdown, result["structure"], outline, rule.headings_by_number
        )
    elif not outline and rule.numbered_levels:
        markdown, counts["numbered_levels"] = heading_rules.relevel(markdown, result["structure"], [], by_number=True)
    markdown, counts["paragraphs_joined"] = piece_join.join_paragraphs(markdown, result["structure"])
    counts["pictures_addressed"] = 0
    if rule.picture_addresses:
        markdown, counts["pictures_addressed"] = code_lines.picture_addresses(markdown, result["structure"])
    counts["formulas_from_layer"] = 0
    if rule.formula_text:
        markdown, counts["formulas_from_layer"] = code_lines.formula_text(markdown, result["structure"])
    counts["man_page_titles"] = 0
    if rule.man_page_titles:
        markdown, counts["man_page_titles"] = heading_rules.man_page_titles(markdown)
    return markdown, counts


def ceiling(settings: dict) -> float:
    return settings.get("piece_ceiling_seconds", CHUNK_CEILING)


# a page range in pieces of a size, first and last inclusive
def pieces(first: int, last: int, size: int) -> list[tuple[int, int]]:
    return [(a, min(a + size - 1, last)) for a in range(first, last + 1, size)]


# the engine a tool is declared on; its supervisor is asked only whether it is up and runs that tool
def converter_for(tool: str):
    name = config.settings.intake.engines.get(Tool(tool)) if tool in Tool else None
    if name is None:
        raise Final(f"no converter engine is declared for {tool} in intake.engines")
    spec = next((s for s in engines.card_engines(EngineKind.converter) if s.name == name), None)
    if spec is None:
        raise Final(f"{name} is declared for {tool} and is not a registered converter engine")
    try:
        runs = converter.reading(spec)[1].get("tool")
    except Exception as e:
        raise Final(f"{name} does not answer: {e}") from e
    if runs != tool:
        raise Final(f"{name} runs {runs}, not {tool}")
    return spec


@functools.lru_cache(maxsize=64)
def _file_sha(path: str, mtime: int, size: int) -> str:
    return sha256(Path(path))


def _stat(path: str) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


# the key of a piece's reading: the same file, pages, settings and converter build read the same; unstamped is no key
def reading_key(spec, tool: str, path: Path, fields: list[tuple[str, str]], chunk) -> str | None:
    build = converter.reading(spec)[1].get("build")
    if not build:
        return None
    body = {
        "tool": tool,
        "file": _file_sha(str(path), *_stat(str(path))),
        "chunk": list(chunk) if chunk else None,
        "fields": sorted([list(f) for f in fields]),
        "build": build,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# a run that measures the tool itself reads every piece again; a kept reading would measure the cache
_FRESH: contextvars.ContextVar[bool] = contextvars.ContextVar("converting_fresh", default=False)


# how a job takes the card for a piece its tool must read and restarts a tool that hung; the job hands it over
class CardHold(NamedTuple):
    take: Callable
    restart: Callable


_HOLD: contextvars.ContextVar[CardHold | None] = contextvars.ContextVar("converting_hold", default=None)


@contextlib.contextmanager
def card_hold(hold: CardHold):
    token = _HOLD.set(hold)
    try:
        yield
    finally:
        _HOLD.reset(token)


@contextlib.contextmanager
def reading_fresh(fresh: bool):
    token = _FRESH.set(bool(fresh))
    try:
        yield
    finally:
        _FRESH.reset(token)


# a piece through its tool's adapter, or its kept reading; a piece that hung restarts the child for the next one
def convert(spec, tool: str, path: Path, fields: list[tuple[str, str]], chunk, ceiling: float) -> dict:
    if tool not in PIECE:
        raise Final(f"no adapter for the converter {tool}")
    key = reading_key(spec, tool, path, fields, chunk)
    kept = READINGS / key[:2] / f"{key}.json" if key else None
    if kept is not None and kept.is_file() and not _FRESH.get():
        reading = json.loads(kept.read_text())
        # a kept reading spent no tool time now; its own time stays beside it for whoever prices the tool
        return {**reading, "seconds": 0.0, "read_seconds": reading.get("seconds"), "cached": True}
    # the card is taken only for a piece the tool must read: a rerun of the rules leaves the card to whoever holds it
    hold = _HOLD.get()
    if hold is None:
        raise Final(f"{tool} must read {path.name}, and no job handed over the card")
    hold.take(spec)
    result = PIECE[Tool(tool)](spec, path, fields, chunk, ceiling)
    if result["status"] == "timeout":
        hold.restart(spec)
        result["errors"] = [*result["errors"], "child restarted"]
    # only a whole reading is kept: a partial one is for the reread to replace, a timeout or failure is read again
    if kept is not None and result["status"] == "success":
        kept.parent.mkdir(parents=True, exist_ok=True)
        # a name of its own per writer: two jobs reading one piece must not write into one file
        part = kept.with_name(f"{kept.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
        part.write_text(json.dumps(result, ensure_ascii=False))
        part.replace(kept)
    return result
