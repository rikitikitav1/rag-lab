import hashlib
import json
from pathlib import Path

from engines import converter
from tool_names import Tool
from use_cases import piece_join, raw_quality, route

from .card import take
from .converting import ceiling, code_lines_of, convert, converter_for, fields, load_settings, pieces

# the route's keys that shape a piece or its reading; the rest (a file skipped, a signal's floor) moves no piece
_SHAPES = (
    "min_layer_chars",
    "min_raster_run",
    "reread_below_layer_f1",
    "reread_settings",
    "reread_cells_slack",
    "seam_window",
    "seam_margin",
    "mono_spread",
    "mono_faces",
    "headings_by_number",
    "listing_callouts",
    "mono_by_step",
    "code_row_rules",
    "outline_levels",
    "html_one_title",
    "epub_chapters",
    "numbered_levels",
    "decode_entities",
    "drop_lone_pipes",
    "join_layer_hyphens",
    "restore_dashes",
    "join_broken_words",
    "unescape_bullets",
    "unescape_underscores",
    "picture_addresses",
    "join_split_words",
)

# the one path a file is read by, for the corpus and for the gold's intake arm alike: plan, read each piece, assemble


# a file's pieces over a page range or all of it, each with its settings file; raises route.Unsupported or Unreadable
def plan(file: Path, names: dict, loaded: dict, pages=None, size: int | None = None, rule=None) -> list:
    units = []
    for run in route.route(file, rule):
        name = (run.settings or names[run.engine]) if run.engine else None
        if name and name not in loaded:
            loaded[name] = load_settings(name)
        size_here = size or (loaded[name][0]["pages_per_chunk"] if name else 0)
        span = run.pages
        # a page range reads only its own pages of each run, as onboarding hands a converter a piece of the whole file
        if pages and span:
            span = (max(span[0], pages[0]), min(span[1], pages[1]))
            if span[0] > span[1]:
                continue
        if span is None:
            cut = [None]
        elif file.suffix.lower() == ".pdf":
            cut = route.seamless_pieces(file, span, size_here, rule)
        else:
            cut = pieces(span[0], span[1], size_here)
        units += [(run, piece, name) for piece in cut]
    return units


# one piece through its engine: its markdown with code rebuilt from the layer, and Docling's structure when it gave one
def convert_piece(file: Path, engine: str | None, piece, settings: tuple | None, language: str, rule=None) -> dict:
    if engine is None:
        return {"markdown": file.read_text(errors="ignore"), "seconds": 0.0, "status": None, "structure": None}
    tool_settings, _ = settings
    spec = converter_for(engine)
    take(spec)
    tool_fields = fields(tool_settings, language)
    result = convert(spec, engine, file, tool_fields, piece, ceiling(tool_settings))
    reading = converter.reading(spec)[1]
    markdown, code = code_lines_of(tool_settings, file, result, rule)
    return {
        "markdown": markdown,
        "code": code,
        "structure": result.get("structure"),
        "seconds": result["seconds"],
        "status": {"status": result["status"], "errors": result["errors"][:3]},
        "build": (reading.get("build") or {}).get("built_at"),
        "residency_started_at": reading.get("started_at"),
    }


# a dash the converter dropped at a line end put back from the piece's own layer, counted with the code rules
def _dashes_back(done: dict, layer, rule, piece=None) -> dict:
    counts = {}
    # the Docling path decodes before its code rules; a reading without them, as MinerU's, is decoded here
    if rule.decode_entities and done.get("code") is None:
        done["markdown"], counts["entities_decoded"] = piece_join.decode_entities(done["markdown"])
    if rule.picture_addresses and done.get("code") is None:
        done["markdown"], counts["pictures_addressed"] = piece_join.inline_pictures(done["markdown"], piece)
    if rule.unescape_bullets:
        done["markdown"], counts["bullets_unescaped"] = piece_join.unescape_bullets(done["markdown"])
    if rule.unescape_underscores:
        done["markdown"], counts["underscores_unescaped"] = piece_join.unescape_underscores(done["markdown"])
    if not layer:
        return _counted(done, counts)
    # only the word rules read the joined layer; the reread's floor was set on the layer as PDFium gives it
    if rule.join_layer_hyphens:
        layer = route.joined_hyphens(layer)
    if rule.restore_dashes:
        done["markdown"], counts["dashes_restored"] = piece_join.restore_dashes(done["markdown"], layer)
    if rule.join_broken_words:
        done["markdown"], counts["words_joined"] = piece_join.join_broken_words(done["markdown"], layer)
    if rule.join_split_words:
        done["markdown"], counts["split_words_joined"] = piece_join.join_split_words(done["markdown"], layer)
    return _counted(done, counts)


def _counted(done: dict, counts: dict) -> dict:
    if any(counts.values()):
        done["code"] = {**(done.get("code") or {}), **counts}
    return done


# a piece read by its planned settings, and again by the reread settings when it agrees badly with its layer
def read_piece(file: Path, engine: str | None, piece, name: str | None, loaded: dict, language: str, layer, rule=None):
    rule = route.rule_of(rule)
    done = convert_piece(file, engine, piece, loaded.get(name), language, rule)
    # a file read as it is is its author's text: the word rules were measured on converters' output only
    if engine is not None:
        done = _dashes_back(done, layer, rule, piece)
    first = raw_quality.conversion_signals(done["markdown"], layer)["layer_f1"]
    if engine != Tool.docling or name == rule.reread_settings or first is None or first >= rule.reread_below_layer_f1:
        return done, name, None
    if rule.reread_settings not in loaded:
        loaded[rule.reread_settings] = load_settings(rule.reread_settings)
    again = _dashes_back(
        convert_piece(file, engine, piece, loaded[rule.reread_settings], language, rule), layer, rule, piece
    )
    second = raw_quality.conversion_signals(again["markdown"], layer)["layer_f1"]
    cells = [raw_quality.table_cells(done["markdown"]), raw_quality.table_cells(again["markdown"])]
    taken = raw_quality.better_reading(
        {"layer_f1": first, "table_cells": cells[0]},
        {"layer_f1": second, "table_cells": cells[1]},
        rule.reread_cells_slack,
    )
    reread = {"settings": rule.reread_settings, "layer_f1": [first, second], "table_cells": cells, "taken": taken}
    return (again, rule.reread_settings, reread) if taken else (done, name, reread)


# a file's pieces in page order as one markdown, with what the joins healed
def assemble(parts: list[str], html: bool, one_title: bool = False) -> tuple[str, dict]:
    whole, healed = piece_join.join(parts)
    healed["headings"] = 0
    if html and one_title:
        whole, healed["headings"] = piece_join.one_title(whole)
    return whole, healed


# the PDF's own text over a piece's pages, or None for a file with no layer
def layer_of(layers: list[str] | None, piece) -> str | None:
    return "\n".join(layers[piece[0] - 1 : piece[1]]) if layers is not None and piece else None


# the route's fingerprint for a resume: what shapes a piece, plus the reread file's own content under its name
def route_sha(rule=None) -> str:
    rule = route.rule_of(rule)
    shaping = {key: getattr(rule, key) for key in _SHAPES} | {"reread_sha256": load_settings(rule.reread_settings)[1]}
    return hashlib.sha256(json.dumps(shaping, sort_keys=True).encode()).hexdigest()[:12]
