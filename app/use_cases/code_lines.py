import re
import statistics
from contextlib import closing
from pathlib import Path

from use_cases.docling_structure import continued, page_of, reading_order
from use_cases.markup import FENCE, HYPHEN_MARK

# a code row runs on to the line's end, is given once a page, and a listing's line numbers go
ROW_RULES = frozenset({"run_on", "once", "numbers"})
# the mark at a line end is a real `-` in code, as in `app-`
_LINE_END_HYPHEN = {HYPHEN_MARK: "-"}


def _chars(page) -> list[tuple[float, float, float, float, str]]:
    with closing(page.get_textpage()) as text:
        out = []
        for i in range(text.count_chars()):
            ch = text.get_text_range(i, 1)
            if ch and ch not in ("\r", "\n"):
                # the loose box is the font's full height, so a comma or an underscore sits on its line's row
                out.append((*text.get_charbox(i, loose=True), _LINE_END_HYPHEN.get(ch, ch)))
        return out


# glyphs of one advance as a monospace face sets them: nine in ten near the median, so a number column does not count
def uniform(widths: list[float], spread: float) -> bool:
    middle = statistics.median(widths) if widths else 0
    if middle <= 0:
        return False
    return sum(1 for w in widths if abs(w - middle) / middle <= spread) >= 0.9 * len(widths)


# enough glyphs at one advance to call a face monospace though its name is not on the list; a short digit row is not
def one_advance(widths: list[float], spread: float) -> bool:
    return len(widths) >= 8 and uniform(widths, spread)


# rows top to bottom, each its glyphs left to right
def glyph_rows(chars: list) -> list[list]:
    rows: list[list] = []
    for c in sorted(chars, key=lambda c: -(c[1] + c[3]) / 2):
        middle = (c[1] + c[3]) / 2
        if rows and abs(rows[-1][0] - middle) < (c[3] - c[1]) * 0.5:
            rows[-1][1].append(c)
        else:
            rows.append([middle, [c]])
    return [(middle, sorted(row, key=lambda c: c[0])) for middle, row in rows]


# a row's glyphs run on to the right while the line goes on: a box drawn short of the line's end cut it mid-word
def _run_on(row: list, page_row: list, advance: float) -> list:
    kept = list(row)
    for c in sorted(page_row, key=lambda c: c[0]):
        if c[0] > kept[-1][0] and c[0] - kept[-1][2] <= 2 * advance and not c[4].isspace():
            kept.append(c)
    return kept


_NUMBER = re.compile(r"^(\s*)(\d+)(\s+)(?=\S)")


# a listing's line-number column, set before each row, is not code; it goes when nearly every row carries one in order
def _without_numbers(lines: list[str]) -> list[str]:
    found = [_NUMBER.match(line) for line in lines]
    numbered = [int(m.group(2)) for m in found if m]
    if len(lines) < 3 or len(numbered) < 0.8 * len(lines) or numbered != sorted(numbered):
        return lines
    rest = [line[m.end(2) :] if m else line for line, m in zip(lines, found, strict=True)]
    shift = min((len(r) - len(r.lstrip()) for r in rest if r.strip()), default=0)
    return [r[shift:] for r in rest]


# a code block as the layer draws it, rows by height and spaces by the monospace advance; a row is given once a page
def layer_code(
    page,
    box: dict,
    taken: set | None = None,
    *,
    spread: float,
    drop=frozenset(),
    by_step: bool = False,
    rows_by: frozenset = ROW_RULES,
    tally: dict | None = None,
) -> str | None:
    chars = [c for c in _chars(page) if not c[4].isspace() and c[:4] not in drop]
    visible = _inside(chars, _box_edges(page, box))
    if not visible:
        return None
    # a proportional face has no single advance to count spaces by, so its words would glue: Docling's text stays
    if by_step:
        advance = _step(visible, spread)
        if advance is None:
            return None
    elif not uniform([c[2] - c[0] for c in visible], spread):
        return None
    else:
        advance = statistics.median(c[2] - c[0] for c in visible)
    page_rows = glyph_rows(chars)
    taken = set() if taken is None else taken
    tally = {} if tally is None else tally
    rows = []
    for middle, row in glyph_rows(visible):
        key = round(middle)
        # a row is given again unless every glyph of it was drawn: a fragment that took its tail does not take it all
        if "once" in rows_by and all((key, round(c[0])) in taken for c in row):
            tally["rows_once"] = tally.get("rows_once", 0) + 1
            continue
        if "run_on" in rows_by:
            whole = next((r for m, r in page_rows if abs(m - middle) < 1), row)
            longer = _run_on(row, whole, advance)
            if len(longer) > len(row):
                tally["rows_run_on"] = tally.get("rows_run_on", 0) + 1
            row = longer
        taken.update((key, round(c[0])) for c in row)
        rows.append(row)
    # every row given already by a box drawn before: the block is a duplicate, not a failed rebuild
    if not rows:
        return ""
    edge = min(row[0][0] for row in rows)
    lines = []
    for row in rows:
        line = " " * round((row[0][0] - edge) / advance)
        for before, c in zip([None, *row[:-1]], row, strict=True):
            if before is not None and by_step:
                line += " " * max(round((c[0] - before[0]) / advance) - 1, 0)
            elif before is not None:
                line += " " * max(round((c[0] - before[2]) / advance), 0)
            line += c[4]
        lines.append(line.rstrip())
    return "\n".join(_without_numbers(lines) if "numbers" in rows_by else lines)


# the advance by glyph origins, for a face whose boxes are wider than its step: nine steps in ten a whole number of it
def _step(visible: list, spread: float) -> float | None:
    steps = [b[0] - a[0] for _, row in glyph_rows(visible) for a, b in zip(row, row[1:], strict=False) if b[0] > a[0]]
    if not steps:
        return None
    floor = sorted(steps)[len(steps) // 10]
    advance = statistics.median(x for x in steps if x <= 1.5 * floor)
    whole = sum(1 for x in steps if abs(x / advance - round(x / advance)) <= spread * 1.5)
    return advance if whole >= 0.9 * len(steps) else None


# Docling's code blocks with their lines from the PDF's own layer, and a block the page break cut in two made one again
def rebuild(
    markdown: str,
    structure: dict | None,
    pdf: Path,
    spread: float,
    callouts=None,
    by_step: bool = False,
    rows_by: frozenset = ROW_RULES,
) -> tuple[str, dict]:
    import pypdfium2 as pdfium

    codes = [item for _, item in reading_order(structure or {}) if item.get("label") == "code"]
    fences = list(FENCE.finditer(markdown))
    counts = {"rebuilt": 0, "kept": 0, "joined": 0, "duplicates_dropped": 0}
    tally: dict = {}
    if not fences:
        return markdown, counts
    # the fences and the code items pair by position; when their counts differ the pairing is not trusted at all
    if len(fences) != len(codes):
        return markdown, {**counts, "kept": len(fences)}
    document = pdfium.PdfDocument(str(pdf))
    try:
        texts, notes, taken = [], [], {}
        for item in codes:
            prov = (item.get("prov") or [{}])[0]
            page_no = prov.get("page_no")
            box = prov.get("bbox")
            if not (page_no and box):
                notes.append("")
                texts.append(None)
                continue
            with closing(document[page_no - 1]) as page:
                drop, note = _callouts(page, box, callouts) if callouts else (frozenset(), "")
                notes.append(note)
                taken_here = taken.setdefault(page_no, set())
                texts.append(
                    layer_code(page, box, taken_here, spread=spread, drop=drop, by_step=by_step, rows_by=rows_by,
                               tally=tally)
                )
    finally:
        document.close()
    # None keeps Docling's own block, "" drops a duplicate block, any other text replaces the block
    bodies = [fence.group(1) if t is None else t + "\n" if t else "" for fence, t in zip(fences, texts, strict=True)]
    goes_on = continued(structure)["code"]
    out, last, i = [], 0, 0
    while i < len(fences):
        body, j = bodies[i], i
        while j + 1 < len(fences) and goes_on[j] and not markdown[fences[j].end() : fences[j + 1].start()].strip():
            j += 1
            body = body.rstrip("\n") + "\n" + bodies[j]
        said = " ".join(n for n in notes[i : j + 1] if n)
        if not body.strip():
            out.append(markdown[last : fences[i].start()])
        else:
            out += [markdown[last : fences[i].start(1)], body, "```" + (f"\n\n{said}" if said else "")]
        last = fences[j].end()
        counts["joined"] += j - i
        i = j + 1
    out.append(markdown[last:])
    counts.update(rebuilt=sum(bool(t) for t in texts), kept=sum(t is None for t in texts), **tally)
    counts["duplicates_dropped"] = sum(1 for t in texts if t == "")
    return "".join(out), counts


# a listing's callouts, set in a text face inside the code box: their glyphs to leave out and their words as one line
def _callouts(page, box: dict, mono_name) -> tuple[frozenset, str]:
    dropped, words = set(), []
    for x0, y0, x1, y1, face, ch in _inside(page_glyphs(page, spaces=True), _box_edges(page, box)):
        if ch.isspace():
            words.append(" ")
        elif face and not mono_name.search(face):
            dropped.add((x0, y0, x1, y1))
            words.append(ch)
        else:
            words.append(" ")
    return frozenset(dropped), " ".join("".join(words).split())


_MARKS = re.compile(r"^\s*(#{1,7}|[-*+•]|\d+[.)])\s+")
_PROSE_LABELS = ("text", "section_header", "list_item", "paragraph")
_MONO_SHARE = 0.9


def _box_edges(page, box: dict) -> tuple[float, float, float, float]:
    height = page.get_height()
    t, b = (height - box["t"], height - box["b"]) if box.get("coord_origin") == "TOPLEFT" else (box["t"], box["b"])
    return box["l"], box["r"], max(t, b), min(t, b)


# glyphs whose middle falls in the box, a point of slack each side
def _inside(glyphs: list, edges: tuple) -> list:
    left, right, top, bottom = edges
    return [
        g for g in glyphs if left - 1 <= (g[0] + g[2]) / 2 <= right + 1 and bottom - 1 <= (g[1] + g[3]) / 2 <= top + 1
    ]


# the share of an item's visible glyphs set in a monospace face by name; a face with no name counts by its one advance
def mono_share(page, box: dict, mono_name, glyphs: list | None = None, *, spread: float) -> float:
    inside = _inside(glyphs if glyphs is not None else _faces(page), _box_edges(page, box))
    if not inside:
        return 0.0
    if one_advance([g[2] - g[0] for g in inside], spread):
        return 1.0
    return sum(bool(mono_name.search(g[4])) for g in inside) / len(inside)


# a page's glyphs as (left, bottom, right, top, face, char), spaces only when asked; the one reader of glyph faces
def page_glyphs(page, spaces: bool = False) -> list[tuple[float, float, float, float, str, str]]:
    import ctypes

    import pypdfium2.raw as pdfium_c

    with closing(page.get_textpage()) as text:
        name, flags = ctypes.create_string_buffer(128), ctypes.c_int()
        out = []
        for i in range(text.count_chars()):
            ch = text.get_text_range(i, 1)
            if ch.isspace() and not spaces:
                continue
            pdfium_c.FPDFText_GetFontInfo(text.raw, i, name, 128, ctypes.byref(flags))
            face = name.value.decode(errors="ignore")
            out.append((*text.get_charbox(i, loose=True), face, _LINE_END_HYPHEN.get(ch, ch)))
        return out


# a page's visible glyphs with their face names, read once for all the items on it
def _faces(page) -> list[tuple[float, float, float, float, str, str]]:
    return page_glyphs(page)


def _flat(text: str) -> str:
    return " ".join(_MARKS.sub("", text.replace("\\", "")).split())


# a term in the code face (`hostname` in a definition list) stays text; a short line is code when it opens with a prompt
def _long_or_prompt(text: str) -> bool:
    text = text.strip()
    return len(text.split()) >= 3 or text[:1] in ("$", "#", ">") and len(text) > 1


def _share_on(pages: dict, document, prov: dict, mono_name, spread: float) -> float:
    page, glyphs = _page_faces(pages, document, prov)
    return mono_share(page, prov["bbox"], mono_name, glyphs, spread=spread)


def _page_faces(pages: dict, document, prov: dict) -> tuple:
    if prov["page_no"] not in pages:
        page = document[prov["page_no"] - 1]
        pages[prov["page_no"]] = (page, _faces(page))
    return pages[prov["page_no"]]


# a heading in the code face on one line is a signature or a command name and stays a heading; a session runs longer
def _one_line(pages: dict, document, prov: dict) -> bool:
    page, glyphs = _page_faces(pages, document, prov)
    return len(glyph_rows([(*g[:4], "x") for g in _inside(glyphs, _box_edges(page, prov["bbox"]))])) <= 1


# runs of Docling text items set wholly in a monospace face, in reading order on one page; an outline heading stays one
def _mono_runs(structure: dict, document, mono_name, outline: set[str], spread: float) -> list[list[dict]]:
    runs, last, pages = [], None, {}
    try:
        for _, item in reading_order(structure):
            prov = (item.get("prov") or [{}])[0]
            goes_on = last is not None and page_of(last) == page_of(item)
            mono = (
                item.get("label") in _PROSE_LABELS
                and item.get("text", "").strip()
                and (goes_on or _long_or_prompt(item["text"]))
                and prov.get("bbox")
                and prov.get("page_no")
                and not (item["label"] == "section_header" and _flat(item["text"]).lower() in outline)
                and not (item["label"] == "section_header" and _one_line(pages, document, prov))
                and _share_on(pages, document, prov, mono_name, spread) >= _MONO_SHARE
            )
            if mono and goes_on:
                runs[-1].append(item)
            elif mono:
                runs.append([item])
            last = item if mono else None
        return runs
    # closed here, in this thread: a page left to the collector is closed by whichever thread collects
    finally:
        for page, _ in pages.values():
            page.close()


# code Docling gave no code box, as a shell prompt it made a heading: fenced with its lines from the layer
def fence_mono(
    markdown: str, structure: dict | None, pdf: Path, mono_name, outline=(), *, spread: float, rows_by=ROW_RULES
) -> tuple[str, int]:
    import pypdfium2 as pdfium

    if not structure:
        return markdown, 0
    document = pdfium.PdfDocument(str(pdf))
    try:
        runs = _mono_runs(structure, document, mono_name, {_flat(t).lower() for t in outline}, spread)
        blocks = markdown.split("\n\n")
        flat = [_flat(b) for b in blocks]
        fenced, at, taken = 0, 0, {}
        for run in runs:
            heads = [_flat(item["text"])[:30] for item in run]
            a = next((k for k in range(at, len(blocks)) if heads[0] and flat[k].startswith(heads[0])), None)
            if a is None or a + len(run) > len(blocks):
                continue
            if not all(flat[a + n].startswith(h) for n, h in enumerate(heads)):
                continue
            lines = []
            for item in run:
                prov = item["prov"][0]
                with closing(document[prov["page_no"] - 1]) as page:
                    drawn = layer_code(
                        page, prov["bbox"], taken.setdefault(prov["page_no"], set()), spread=spread, rows_by=rows_by
                    )
                if drawn != "":
                    lines.append(drawn if drawn is not None else item["text"])
            fence = "```\n" + "\n".join(lines) + "\n```"
            blocks[a : a + len(run)] = [fence]
            flat[a : a + len(run)] = [_flat(fence)]
            fenced, at = fenced + 1, a + 1
        return "\n\n".join(blocks), fenced
    finally:
        document.close()


_PLACEHOLDER = "<!-- image -->"
_FORMULA = "<!-- formula-not-decoded -->"
# private-use glyphs of a math font (brace pieces) and control marks carry nothing a reader or a search can use
_FORMULA_NOISE = re.compile(r"[\ue000-\uf8ff\x00-\x08\x0b-\x1f\ufffe\uffff]")


def _caption(structure: dict, picture: dict) -> str:
    texts = structure.get("texts", [])
    found = [texts[int(c["$ref"].rsplit("/", 1)[1])] for c in picture.get("captions", []) if "/texts/" in c["$ref"]]
    # brackets escaped, not changed: `[2010]` stays as printed and the alt text still closes where it should
    return " ".join(" ".join(t.get("text", "").split()) for t in found).replace("[", "\\[").replace("]", "\\]")


def formula_line(item: dict) -> str:
    return " ".join(_FORMULA_NOISE.sub(" ", item.get("orig") or "").split())


# a piece's formulas as their text lines, page by page in reading order
def formulas_by_page(structure: dict) -> dict[int | None, list[str]]:
    found: dict = {}
    for _, item in reading_order(structure):
        if item.get("label") == "formula":
            found.setdefault(page_of(item), []).append(formula_line(item))
    return found


# each formula placeholder gets the text the layer holds under it, flattened; all left alone when the counts disagree
def formula_text(markdown: str, structure: dict | None) -> tuple[str, int]:
    formulas = [item for _, item in reading_order(structure or {}) if item.get("label") == "formula"]
    if not formulas or markdown.count(_FORMULA) != len(formulas):
        return markdown, 0
    parts = markdown.split(_FORMULA)
    out, filled = [parts[0]], 0
    for formula, rest in zip(formulas, parts[1:], strict=True):
        text = formula_line(formula)
        filled += bool(text)
        out += [text or _FORMULA, rest]
    return "".join(out), filled


# each placeholder gets its picture's page, place on the page and caption; all left alone when the counts disagree
def picture_addresses(markdown: str, structure: dict | None) -> tuple[str, int]:
    pictures = [item for ref, item in reading_order(structure or {}) if ref.startswith("#/pictures/")]
    if not pictures or markdown.count(_PLACEHOLDER) != len(pictures):
        return markdown, 0
    seen: dict[int | None, int] = {}
    parts = markdown.split(_PLACEHOLDER)
    for n, picture in enumerate(pictures):
        page = page_of(picture)
        seen[page] = seen.get(page, 0) + 1
        parts[n] += f"![{_caption(structure, picture)}](picture:p{page}-{seen[page]})"
    return "".join(parts), len(pictures)
