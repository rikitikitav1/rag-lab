import hashlib
import json
import re
from pathlib import Path

from config import SHAPE_NO_PIECE, RouteCfg
from engines import converter
from engines.converter_tools import KEPT_STATUSES
from tool_names import Tool
from use_cases import (
    code_lines,
    docling_structure,
    heading_rules,
    layer_words,
    markdown_cleanup,
    piece_join,
    raw_quality,
    route,
    rule_counts,
)
from use_cases.converting import ceiling, code_lines_of, convert, converter_for, fields, load_settings, pieces

# the route's keys that shape a piece or its reading; a file skipped or a signal's floor moves no piece
_SHAPES = tuple(name for name in RouteCfg.model_fields if name not in SHAPE_NO_PIECE)

# the one path a file is read by, for the corpus and for the gold's intake arm alike: plan, read each piece, assemble


# a file's pieces over a page range or all of it, each with its settings file; raises route.Unsupported or Unreadable
def plan(file: Path, names: dict, loaded: dict, rule, pages=None, size: int | None = None) -> list:
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
def convert_piece(file: Path, engine: str | None, piece, settings: tuple | None, language: str, rule) -> dict:
    if engine is None:
        fired = rule_counts.Fired(rule)
        markdown = markdown_cleanup.markdown_of(file)
        markdown = fired.run("drop_repeated_code", markdown, markdown_cleanup.drop_repeated_code)
        done = {"markdown": markdown, "seconds": 0.0, "status": None, "structure": None}
        return {**done, "code": fired.counts} if any(fired.counts.values()) else done
    tool_settings, _ = settings
    spec = converter_for(engine)
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
        "cached": result.get("cached", False),
        "build": (reading.get("build") or {}).get("built_at"),
        "residency_started_at": reading.get("started_at"),
    }


# a dash the converter dropped at a line end put back from the piece's own layer, counted with the code rules
def _dashes_back(done: dict, layer, rule, piece=None) -> dict:
    fired = rule_counts.Fired(rule)
    md = done["markdown"]
    # the Docling path decodes before its code rules; a reading without them, as MinerU's, is decoded here
    if done.get("code") is None:
        md = fired.run("decode_entities", md, markdown_cleanup.decode_entities)
        md = fired.run("picture_addresses", md, markdown_cleanup.inline_pictures, piece)
    md = fired.run("unescape_bullets", md, markdown_cleanup.unescape_bullets)
    md = fired.run("unescape_underscores", md, markdown_cleanup.unescape_underscores)
    md = fired.run("demote_caption_headings", md, heading_rules.demote_caption_headings)
    md = fired.run("drop_inherited_members", md, markdown_cleanup.drop_inherited_members)
    if layer:
        md = fired.run("drop_running_headings", md, heading_rules.drop_running_headings, layer)
        marked = layer_words.marked_joins(layer)
        # only the word rules read the joined layer; the reread's floor was set on the layer as PDFium gives it
        if rule.join_layer_hyphens:
            layer = route.joined_hyphens(layer)
        md = fired.run("restore_dashes", md, layer_words.restore_dashes, layer, marked)
        md = fired.run("join_broken_words", md, layer_words.join_broken_words, layer)
        md = fired.run("join_split_words", md, layer_words.join_split_words, layer)
    done["markdown"] = md
    return _counted(done, fired.counts)


def _counted(done: dict, counts: dict) -> dict:
    if any(counts.values()):
        done["code"] = {**(done.get("code") or {}), **counts}
    return done


def _partial(done: dict) -> bool:
    return bool(done.get("status")) and done["status"]["status"] == "partial_success"


# a piece the tool read in part is read once more by the reread settings; the whole reading is taken, else the first
def _reread_partial(file, engine, piece, name, loaded, language, layer, rule, done):
    if rule.reread_settings not in loaded:
        loaded[rule.reread_settings] = load_settings(rule.reread_settings)
    again = _dashes_back(
        convert_piece(file, engine, piece, loaded[rule.reread_settings], language, rule), layer, rule, piece
    )
    taken = not _partial(again) and again["status"]["status"] in KEPT_STATUSES
    reread = {"settings": rule.reread_settings, "for": "partial", "taken": taken}
    return (again, rule.reread_settings, reread) if taken else (done, name, reread)


# a piece read by its planned settings, and again by the reread settings when it agrees badly with its layer
def read_piece(file: Path, engine: str | None, piece, name: str | None, loaded: dict, language: str, layer, rule):
    done = convert_piece(file, engine, piece, loaded.get(name), language, rule)
    # a file read as it is is its author's text: the word rules were measured on converters' output only
    if engine is not None:
        done = _dashes_back(done, layer, rule, piece)
    if engine == Tool.docling and name != rule.reread_settings and _partial(done):
        return _reread_partial(file, engine, piece, name, loaded, language, layer, rule, done)
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
    if taken:
        # the prose reading taken whole may still have lost a formula's operators the first reading kept
        if rule.splice_tables:
            markdown, reread["formulas_spliced"] = splice_formulas(again, done)
            again = {**again, "markdown": markdown}
        return again, rule.reread_settings, reread
    # lost on cells alone: the second reading's prose, and the first's tables and formulas where they read more
    if rule.splice_tables and first is not None and second is not None and second > first and cells[1] < cells[0]:
        markdown, spliced = splice_tables(again, done)
        if spliced:
            markdown, reread["formulas_spliced"] = splice_formulas({**again, "markdown": markdown}, done)
            reread["tables_spliced"] = spliced
            return {**again, "markdown": markdown}, rule.reread_settings, reread
    return done, name, reread


def _tables_by_page(done: dict) -> list[tuple[int, tuple[int, int]]] | None:
    order = docling_structure.reading_order(done.get("structure") or {})
    items = [item for _, item in order if item.get("label") == "table"]
    spans = piece_join.table_spans(done["markdown"].split("\n"))
    if len(items) != len(spans):
        return None
    return [(docling_structure.page_of(item), span) for item, span in zip(items, spans, strict=True)]


# a table of the prose reading swapped for the table reading's one on its page, when that one keeps more cells
def splice_tables(prose: dict, tables: dict) -> tuple[str, int]:
    ours, theirs = _tables_by_page(prose), _tables_by_page(tables)
    if not ours or not theirs:
        return prose["markdown"], 0
    lines, other = prose["markdown"].split("\n"), tables["markdown"].split("\n")
    by_page: dict = {}
    for page, span in theirs:
        by_page.setdefault(page, []).append(span)
    out, at, seen, spliced = [], 0, {}, 0
    for page, (start, end) in ours:
        nth = seen[page] = seen.get(page, -1) + 1
        mine, found = lines[start : end + 1], by_page.get(page, [])
        if nth < len(found):
            theirs_rows = other[found[nth][0] : found[nth][1] + 1]
            if raw_quality.table_cells("\n".join(theirs_rows)) > raw_quality.table_cells("\n".join(mine)):
                mine, spliced = theirs_rows, spliced + 1
        out += lines[at:start] + mine
        at = end + 1
    return "\n".join(out + lines[at:]), spliced


# operators a math font draws; the second reading's backend reads that font's glyphs as control marks and loses them
_OPERATORS = re.compile(r"[=+<>≤≥≠×÷∑∫√±∧∨¬→←↔]")


# a formula the prose reading lost operators in, taken back from the table reading at the same page and place
def splice_formulas(prose: dict, tables: dict) -> tuple[str, int]:
    ours = code_lines.formulas_by_page(prose.get("structure") or {})
    theirs = code_lines.formulas_by_page(tables.get("structure") or {})
    markdown, spliced, at = prose["markdown"], 0, 0
    for page, texts in ours.items():
        other = theirs.get(page, [])
        if len(other) != len(texts):
            continue
        for mine, better in zip(texts, other, strict=True):
            # the formula's own line after the one before it: a short one like `x = 1` also stands in prose
            found = re.compile(rf"^[ \t$]*{re.escape(mine)}[ \t$]*$", re.M).search(markdown, at) if mine else None
            if found is None:
                continue
            at = found.end()
            if len(_OPERATORS.findall(better)) > len(_OPERATORS.findall(mine)):
                line = found.group(0).replace(mine, better, 1)
                markdown, at = markdown[: found.start()] + line + markdown[found.end() :], found.start() + len(line)
                spliced += 1
    return markdown, spliced


# a file's pieces in page order as one markdown, with what the joins healed
def assemble(parts: list[str], html: bool, one_title: bool = False) -> tuple[str, dict]:
    whole, healed = piece_join.join(parts)
    healed["headings"] = 0
    if html and one_title:
        whole, healed["headings"] = heading_rules.one_title(whole)
    return whole, healed


# one file whole for the corpus, the gold's intake arm and the probe alike; a running head is read over the whole file
def whole_file(parts: list[str], routes: list[str], rule, layers: list[str] | None) -> tuple[str, dict]:
    html = bool(routes) and all(why == "html" for why in routes)
    whole, healed = assemble(parts, html, rule.html_one_title)
    healed["running_heads"] = 0
    # a piece drops the heads it repeats itself; the first of each in a later piece is only seen from the whole
    if rule.drop_running_headings and layers:
        whole, healed["running_heads"] = heading_rules.drop_running_headings(whole, "\f".join(layers))
    return whole, healed


# the PDF's own text over a piece's pages, or None for a file with no layer
def layer_of(layers: list[str] | None, piece) -> str | None:
    # pages apart by a form feed, whitespace to the word rules, so a rule can still find a page's first line
    return "\f".join(layers[piece[0] - 1 : piece[1]]) if layers is not None and piece else None


# what shapes a piece and the reread file's content; a knob that is off is left out, so adding one moves nothing
def route_rules(rule) -> dict:
    shaping = {key: getattr(rule, key) for key in _SHAPES} | {"reread_sha256": load_settings(rule.reread_settings)[1]}
    return {key: value for key, value in shaping.items() if value not in (None, False, [])}


# the route's fingerprint for a resume
def route_sha(rule) -> str:
    return hashlib.sha256(json.dumps(route_rules(rule), sort_keys=True).encode()).hexdigest()[:12]
