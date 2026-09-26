import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import config
from tool_names import Tool

MARKDOWN = {".md", ".markdown", ".txt"}
IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
HTML = {".html", ".htm"}
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Run:
    # None is a file the loader reads as it is, no converter between
    engine: str | None
    # 1-based and inclusive; None is the whole file
    pages: tuple[int, int] | None
    why: str
    signals: dict = field(default_factory=dict, compare=False)


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


# a PDF's own text layer page by page; pypdf glued the words of LaTeX and Sphinx books, PDFium keeps their spaces
def layer_texts(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        raise Unreadable(f"{path.name}: {e}") from e
    try:
        return [document[i].get_textpage().get_text_range() for i in range(len(document))]
    finally:
        document.close()


# what a PDF's own text layer says about each page, read before any converter touches it
def page_signals(path: Path) -> list[dict]:
    texts = layer_texts(path)
    per_page = [words(t) for t in texts]
    seen = Counter(w for page in per_page for w in page)
    floor = config.settings.intake.route.suspect_min_words
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
def _pdf_runs(signals: list[dict]) -> list[Run]:
    rule = config.settings.intake.route
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


# which engine reads each part of a file: the kind is read from the file itself, never declared
def route(path: Path) -> list[Run]:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN:
        return [Run(None, None, "markdown")]
    if suffix in IMAGE:
        return [Run(Tool.mineru, None, "image")]
    if suffix in HTML:
        return [Run(Tool.docling, None, "html")]
    if suffix == ".pdf":
        return _pdf_runs(page_signals(path))
    return [Run(Tool.docling, None, f"other: {suffix or 'no suffix'}")]
