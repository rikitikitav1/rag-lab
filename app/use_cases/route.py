import re
import unicodedata
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import config
from tool_names import Tool
from use_cases import code_lines

MARKDOWN = {".md", ".markdown", ".txt"}
IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
HTML = {".html", ".htm", ".xhtml"}
OFFICE = {".docx", ".pptx", ".xlsx", ".odt", ".rtf"}
_WORD = re.compile(r"\w+")
_BROKEN_WORD = re.compile("\ufffe(?:\r?\n)?")


@dataclass(frozen=True)
class Run:
    # None is a file the loader reads as it is, no converter between
    engine: str | None
    # 1-based and inclusive; None is the whole file
    pages: tuple[int, int] | None
    why: str
    signals: dict = field(default_factory=dict, compare=False)
    # a settings file of its own; None reads the engine's settings of the source's arm
    settings: str | None = None


# a word whose letters come from two scripts, as a Cyrillic word with a Latin letter an OCR put in
def mixed_script(word: str) -> bool:
    return len({unicodedata.name(c, "?").split(" ")[0] for c in word if c.isalpha()}) > 1


# NFKC before the split, so a ligature of a TeX layer reads as the letters the converter gives back
def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(unicodedata.normalize("NFKC", text).lower()) if any(c.isalpha() for c in w)]


# the share of words mixing two scripts; below the floor a share says nothing and is left out rather than read as zero
def mixed_share(found: list[str], floor: int = 1) -> float | None:
    return round(sum(map(mixed_script, found)) / len(found), 4) if found and len(found) >= floor else None


class Unreadable(Exception):
    pass


# a file no engine reads, as a stylesheet or a font beside the chapters; left out of the source, never retried
class Unsupported(Exception):
    pass


# PDFium marks a hyphen that breaks a word at a line end with U+FFFE: the word is joined, as the converter prints it
def joined_hyphens(text: str) -> str:
    return _BROKEN_WORD.sub("", text)


# a PDF's own text layer page by page; pypdf glued the words of LaTeX and Sphinx books, PDFium keeps their spaces
def layer_texts(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        raise Unreadable(f"{path.name}: {e}") from e
    try:
        texts = []
        for i in range(len(document)):
            with closing(document[i]) as page, closing(page.get_textpage()) as text:
                texts.append(text.get_text_range())
        return texts
    finally:
        document.close()


# a PDF's outline as (depth, title, page), every level, page 1-based or None; a file with no outline gives none
def outline(path: Path) -> list[tuple[int, str, int | None]]:
    import pypdfium2 as pdfium

    if path.suffix.lower() != ".pdf":
        return []
    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError:
        return []
    try:
        entries = []
        for item in document.get_toc(max_depth=15):
            dest = item.get_dest()
            page = dest.get_index() + 1 if dest is not None and dest.get_index() is not None else None
            entries.append((item.level, item.get_title().strip(), page))
        return entries
    finally:
        document.close()


def outline_titles(path: Path) -> list[str]:
    return [title for _, title, _ in outline(path)]


# the route's rules for a source: its own when it has them, the stand's otherwise
def rule_of(rule=None):
    return rule or config.settings.intake.route


# what a PDF's own text layer says about each page, read before any converter touches it
def page_signals(path: Path, rule=None) -> list[dict]:
    texts = layer_texts(path)
    per_page = [words(t) for t in texts]
    seen = Counter(w for page in per_page for w in page)
    floor = rule_of(rule).suspect_min_words
    return [
        {
            "page": n,
            "layer_chars": len(text.strip()),
            "layer_words": len(page),
            "mixed_script": mixed_share(page, floor),
            "once_in_file": round(sum(seen[w] == 1 for w in page) / len(page), 4) if len(page) >= floor else None,
        }
        for n, (text, page) in enumerate(zip(texts, per_page, strict=True), start=1)
    ]


# consecutive pages of one kind as runs; a short raster run inside text stays with its neighbours
def _pdf_runs(signals: list[dict], rule=None) -> list[Run]:
    rule = rule_of(rule)
    layered = [s["layer_chars"] >= rule.min_layer_chars for s in signals]
    runs: list[list] = []
    for page, has_layer in enumerate(layered, start=1):
        if runs and runs[-1][0] == has_layer:
            runs[-1][2] = page
        else:
            runs.append([has_layer, page, page])
    if any(layered):
        for run in runs:
            if not run[0] and run[2] - run[1] + 1 < rule.min_raster_run:
                run[0] = True
    merged: list[list] = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        else:
            merged.append(list(run))
    out = []
    for has_layer, first, last in merged:
        pages = signals[first - 1 : last]
        # a scanned page kept with its text neighbours is read by Docling's OCR, and the report counts what that cost
        absorbed = [s["page"] for s in pages if has_layer and s["layer_chars"] < rule.min_layer_chars]
        why = "text layer" if has_layer else "no text layer"
        if absorbed:
            why += f", {len(absorbed)} raster pages read by docling"
        out.append(
            Run(Tool.docling if has_layer else Tool.mineru, (first, last), why, {"pages": pages, "absorbed": absorbed})
        )
    return out


# a monospace face by its name: the layer's font flags do not say fixed pitch reliably, the names do
def mono_names(rule):
    return re.compile("|".join(re.escape(face) for face in rule.mono_faces), re.I)


# the page's rows top to bottom, each with whether most of its glyphs are monospace; running heads and feet left out
def mono_rows(page, rule=None) -> list[bool]:
    import ctypes

    import pypdfium2.raw as pdfium_c

    rule = rule_of(rule)
    margin, mono_name = rule.seam_margin, mono_names(rule)
    height = page.get_height()
    name = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    glyphs = []
    with closing(page.get_textpage()) as text:
        for i in range(text.count_chars()):
            if text.get_text_range(i, 1).isspace():
                continue
            _, bottom, _, top = text.get_charbox(i, loose=True)
            middle = (bottom + top) / 2
            if not margin * height < middle < (1 - margin) * height:
                continue
            pdfium_c.FPDFText_GetFontInfo(text.raw, i, name, 128, ctypes.byref(flags))
            left, _, right, _ = text.get_charbox(i, loose=True)
            mono = bool(mono_name.search(name.value.decode(errors="ignore")))
            glyphs.append((middle, top - bottom, mono, right - left))
    rows: list[list] = []
    for middle, size, mono, width in sorted(glyphs, key=lambda g: -g[0]):
        if rows and abs(rows[-1][0] - middle) < size * 0.5:
            rows[-1][1].append((mono, width))
        else:
            rows.append([middle, [(mono, width)]])

    # a face with no name or a name outside the list still sets code at one advance; a short row of digits would too
    def monospace(marks):
        by_name = sum(m for m, _ in marks) * 2 > len(marks)
        return by_name or (len(marks) >= 8 and code_lines.uniform([w for _, w in marks], rule.mono_spread))

    return [monospace(marks) for _, marks in rows]


# code runs over the page break when the last row of one page and the first of the next are both monospace
def code_crosses(before: list[bool], after: list[bool]) -> bool:
    return bool(before) and bool(after) and before[-1] and after[0]


# pieces of a run whose ends are moved, within the window, off a page break that code runs over
def seamless_pieces(path: Path, pages: tuple[int, int], size: int, rule=None) -> list[tuple[int, int]]:
    import pypdfium2 as pdfium

    rule = rule_of(rule)
    document = pdfium.PdfDocument(str(path))
    rows: dict[int, list[bool]] = {}

    def crosses(end: int) -> bool:
        for page in (end, end + 1):
            if page not in rows:
                with closing(document[page - 1]) as pdf_page:
                    rows[page] = mono_rows(pdf_page, rule)
        return code_crosses(rows[end], rows[end + 1])

    try:
        out, start = [], pages[0]
        while start <= pages[1]:
            end = min(start + size - 1, pages[1])
            if end < pages[1]:
                shifts = sorted(range(-rule.seam_window, rule.seam_window + 1), key=abs)
                candidates = [end + d for d in shifts if start <= end + d < pages[1]]
                end = next((e for e in candidates if not crosses(e)), end)
            out.append((start, end))
            start = end + 1
        return out
    finally:
        document.close()


# which engine reads each part of a file: the kind is read from the file itself, never declared
def route(path: Path, rule=None) -> list[Run]:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN:
        return [Run(None, None, "markdown")]
    if suffix in IMAGE:
        return [Run(Tool.mineru, None, "image")]
    if suffix in HTML:
        return [Run(Tool.docling, None, "html")]
    if suffix in OFFICE:
        return [Run(Tool.docling, None, f"office: {suffix}")]
    if suffix == ".pdf":
        return _pdf_runs(page_signals(path, rule), rule)
    raise Unsupported(suffix or "no suffix")
