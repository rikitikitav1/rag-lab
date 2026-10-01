import re
import unicodedata

_FENCE_LINE = re.compile(r"^\s*```")
_HEADING = re.compile(r"^(#{1,6}) ")


def _open_fence(text: str) -> bool:
    return sum(1 for line in text.splitlines() if _FENCE_LINE.match(line)) % 2 == 1


def _cells(line: str) -> int:
    return len(line.strip().strip("|").split("|"))


def _separator(line: str) -> bool:
    return line.strip().startswith("|") and set(line.strip()) <= set("|-: ")


# a code block the page break left open goes on in the next piece's first fence, which is dropped to join them
def _heal_fence(before: str, after: str) -> tuple[str, str, bool]:
    if not _open_fence(before):
        return before, after, False
    lines = after.lstrip("\n").splitlines()
    if lines and _FENCE_LINE.match(lines[0]):
        return before.rstrip("\n"), "\n".join(lines[1:]), True
    return before.rstrip("\n") + "\n```", after, True


# a table the break cut in two: the second half's header is either the first half's again or a row read as a header
def _join_table(before: str, after: str) -> tuple[str, str, bool]:
    head = before.rstrip("\n").splitlines()
    tail = after.lstrip("\n").splitlines()
    if len(head) < 2 or len(tail) < 2 or not head[-1].strip().startswith("|") or not _separator(tail[1]):
        return before, after, False
    top = len(head) - 1
    while top > 0 and head[top - 1].strip().startswith("|"):
        top -= 1
    header = head[top]
    if _cells(tail[0]) != _cells(header):
        return before, after, False
    rest = tail[2:] if tail[0].split() == header.split() else [tail[0], *tail[2:]]
    head_text = before.rstrip("\n")
    # a row the page break cut goes on in a row whose first cell is empty: its cells finish the row above
    if rest and (merged := _continued_row(head[-1], rest[0])):
        head_text, rest = "\n".join([*before.rstrip("\n").splitlines()[:-1], merged]), rest[1:]
    return head_text, "\n".join(rest), True


def _continued_row(above: str, row: str) -> str | None:
    if _separator(above) or not row.strip().startswith("|"):
        return None
    top, low = above.strip().strip("|").split("|"), row.strip().strip("|").split("|")
    if len(top) != len(low) or low[0].strip() or not any(cell.strip() for cell in low):
        return None
    cells = [f"{a.strip()} {b.strip()}".strip() for a, b in zip(top, low, strict=True)]
    return "| " + " | ".join(cells) + " |"


# a file's pieces joined in page order, with what the joins healed
def join(parts: list[str]) -> tuple[str, dict]:
    healed = {"fences": 0, "tables": 0}
    if not parts:
        return "", healed
    whole = parts[0]
    for part in parts[1:]:
        whole, part, fence = _heal_fence(whole, part)
        table = False
        if not fence:
            whole, part, table = _join_table(whole, part)
        healed["fences"] += fence
        healed["tables"] += table
        whole = whole + ("\n" if fence or table else "\n\n") + part
    return whole, healed


# sections an HTML page sets at its title's level (each section's h1) go one level down, where the chunker cuts
def one_title(markdown: str) -> tuple[str, int]:
    lines, fenced, marks = markdown.split("\n"), False, []
    for i, line in enumerate(lines):
        if _FENCE_LINE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line)):
            marks.append((i, len(m.group(1))))
    if not marks or [level for _, level in marks].count(1) < 2 or marks[0][1] != 1:
        return markdown, 0
    # a `##` before the second title means the page already nests its sections, and nothing moves
    second = next(i for i, (_, level) in enumerate(marks) if i and level == 1)
    if any(level == 2 for _, level in marks[1:second]):
        return markdown, 0
    for i, level in marks[1:]:
        lines[i] = "#" * min(level + 1, 6) + lines[i][level:]
    return "\n".join(lines), len(marks) - 1


# a piece's tables as line ranges, fenced code left out
def table_spans(lines: list[str]) -> list[tuple[int, int]]:
    found, fenced, start = [], False, None
    for i, line in enumerate([*lines, ""]):
        if _FENCE_LINE.match(line):
            fenced = not fenced
        is_row = not fenced and line.strip().startswith("|")
        if is_row and start is None:
            start = i
        elif not is_row and start is not None:
            found.append((start, i - 1))
            start = None
    return found


# tables the structure says go on over a page break, joined when only blank lines lie between them in the markdown
def join_tables(markdown: str, goes_on: list[bool]) -> tuple[str, int]:
    lines = markdown.split("\n")
    tables = table_spans(lines)
    if len(tables) != len(goes_on):
        return markdown, 0
    joined = 0
    for k in range(len(tables) - 2, -1, -1):
        (start, end), (next_start, next_end) = tables[k], tables[k + 1]
        if not goes_on[k] or any(line.strip() for line in lines[end + 1 : next_start]):
            continue
        first, second = "\n".join(lines[start : end + 1]), "\n".join(lines[next_start : next_end + 1])
        before, after, merged = _join_table(first, second)
        if merged:
            lines[start : next_end + 1] = (before + ("\n" + after if after else "")).split("\n")
            joined += 1
    return "\n".join(lines), joined


_ENDS = ".!?…:;"
_CLOSERS = ")]»\"”’'"
# page feet and notes stand between a paragraph's halves without ending it; a float between does end it
_PASSED = {"page_header", "page_footer", "footnote"}
_MARGIN_WORDS = 6


def _flat(text: str) -> str:
    return " ".join(text.replace("\\", "").split())


# a paragraph goes on over the page break when its half ends mid-sentence and the next half opens in lowercase
def _goes_on(before: str, after: str) -> bool:
    end, start = before.rstrip().rstrip(_CLOSERS), after.lstrip()
    return bool(end and start) and end[-1] not in _ENDS and start[0].islower()


def _span(item: dict) -> tuple[float, float] | None:
    box = (item.get("prov") or [{}])[0].get("bbox")
    return (box["l"], box["r"]) if box else None


# two items share a column when their widths overlap by half the narrower; with no boxes nothing is said against it
def _one_column(a: dict, b: dict) -> bool:
    x, y = _span(a), _span(b)
    if x is None or y is None:
        return True
    overlap = min(x[1], y[1]) - max(x[0], y[0])
    return overlap >= 0.5 * min(x[1] - x[0], y[1] - y[0])


# a margin note beside the text column, as «с. 211»: a few words out of the paragraph's column
def _in_margin(item: dict, paragraph: dict) -> bool:
    if item.get("label") != "text":
        return False
    return len(item.get("text", "").split()) <= _MARGIN_WORDS and not _one_column(item, paragraph)


# text items a page break cut, as pairs in reading order with only feet, notes and margin notes between
def _cut_paragraphs(structure: dict) -> list[tuple[str, str]]:
    from use_cases.code_lines import page_of, reading_order

    order = [item for _, item in reading_order(structure)]
    pairs = []
    for i, item in enumerate(order):
        if item.get("label") != "text" or page_of(item) is None:
            continue
        j = i + 1
        while j < len(order) and (order[j].get("label") in _PASSED or _in_margin(order[j], item)):
            j += 1
        after = order[j] if j < len(order) else {}
        if (
            after.get("label") == "text"
            and page_of(after) == page_of(item) + 1
            and _one_column(item, after)
            and len(item["text"].split()) > _MARGIN_WORDS
            and _goes_on(item["text"], after["text"])
        ):
            pairs.append((item["text"], after["text"]))
    return pairs


# a paragraph the page break cut, made one again in the markdown; notes between move after it, a hyphen is dropped
def join_paragraphs(markdown: str, structure: dict | None) -> tuple[str, int]:
    blocks = markdown.split("\n\n")
    flat = [_flat(b) for b in blocks]
    joined, at = 0, 0
    for before, after in _cut_paragraphs(structure or {}):
        tail, head = _flat(before)[-30:], _flat(after)[:30]
        a = next((k for k in range(at, len(blocks)) if flat[k].endswith(tail)), None)
        b = next((k for k in range((a or 0) + 1, len(blocks)) if a is not None and flat[k].startswith(head)), None)
        if a is None or b is None:
            continue
        first, second = blocks[a].rstrip(), blocks[b].lstrip()
        hyphen = first.endswith(("-", "\xad")) and first[-2:-1].isalpha()
        merged = first[:-1] + second if hyphen else f"{first} {second}"
        blocks[a : b + 1] = [merged, *blocks[a + 1 : b]]
        flat[a : b + 1] = [_flat(merged), *flat[a + 1 : b]]
        joined, at = joined + 1, a
    return "\n\n".join(blocks), joined


_MD_HEADING = re.compile(r"^(#{1,})\s+(.*\S)\s*$")
# a heading's own section number says its depth, as «1.4.22» is third, where a source asks for it
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\.?\s")
# a heading may sit a page off its outline entry, as one that closes the page before its section
_OUTLINE_PAGES = 1


# a title's words with its section number kept: «1.2 Overview» is not a unit's «Overview»
def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))


# the outline depth of each Docling heading that matches an entry by title and page
def _depths(structure: dict, outline: list) -> list[tuple[str, int]]:
    from use_cases.code_lines import page_of, reading_order

    entries = [(depth, _words(title), page) for depth, title, page in outline if page and _words(title)]
    found = []
    for _, item in reading_order(structure):
        if item.get("label") != "section_header" or not (title := _words(item.get("text", ""))):
            continue
        page = page_of(item)
        near = sorted((abs(p - page), d, t) for d, t, p in entries if page and abs(p - page) <= _OUTLINE_PAGES)
        depth = next((d for _, d, t in near if t == title or t.endswith(title) or title.endswith(t)), None)
        if depth is not None:
            found.append((title, depth))
    return found


# heading levels from the outline, its shallowest on `##` where the chunker starts cutting; past level six a bold line
def relevel(markdown: str, structure: dict | None, outline: list, by_number: bool = False) -> tuple[str, int]:
    matched = _depths(structure or {}, outline)
    # the whole outline's top, not the piece's: a piece inside a chapter levels its sections as the chapter's piece does
    base = min((d for d, title, page in outline if page and _words(title)), default=0) - 1
    lines, inside, changed, at, shift = markdown.split("\n"), False, 0, 0, 0
    for n, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            inside = not inside
        if inside or not (m := _MD_HEADING.match(line)):
            continue
        title = _words(m.group(2))
        hit = next((k for k in range(at, len(matched)) if matched[k][0] == title), None)
        if hit is not None:
            at = hit + 1
        if by_number and (number := _NUMBERED.match(m.group(2))):
            level = min(number.group(1).count(".") + 1, 6)
        elif hit is not None:
            level = min(max(matched[hit][1] - base + 1, 1), 6)
            shift = max(level - len(m.group(1)), 0)
        elif len(m.group(1)) > 6:
            lines[n], changed = f"**{m.group(2)}**", changed + 1
            continue
        else:
            # a heading the outline lacks goes down with the outline's heading above it, never up to a chapter's level
            level = min(len(m.group(1)) + shift, 6)
        if level != len(m.group(1)):
            lines[n], changed = f"{'#' * level} {m.group(2)}", changed + 1
    return "\n".join(lines), changed


_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_TABLE_PIPE = re.compile(r"&(?:#124|#x7[cC]|vert|verbar|VerticalLine);")


# Docling writes `<`, `>` and `&` as entities; a book never means them, and they hide a command from the fence rule
def decode_entities(markdown: str) -> tuple[str, int]:
    import html

    found = len(_ENTITY.findall(markdown))
    if not found:
        return markdown, 0
    # a pipe inside a table cell stays escaped, or it splits the row into one more column
    lines = [_TABLE_PIPE.sub(r"\\|", line) if line.lstrip().startswith("|") else line for line in markdown.split("\n")]
    # each whole entity alone: `html.unescape` over the text reads `&section=2` in a URL as `§ion=2`
    return _ENTITY.sub(lambda m: html.unescape(m.group(0)), "\n".join(lines)), found


# a paragraph that is only `|`, a running foot's rule or a box edge, carries nothing and splits a listing in two
def drop_lone_pipes(markdown: str) -> tuple[str, int]:
    blocks, inside, kept, dropped = markdown.split("\n\n"), False, [], 0
    for block in blocks:
        if not inside and block.strip() == "|":
            dropped += 1
            continue
        kept.append(block)
        inside ^= block.count("```") % 2 == 1
    return "\n\n".join(kept), dropped


# a dash of any kind across a line end, or a hyphen inside a line; a hyphen at a line end is a word's hyphenation
_DASHED = re.compile(r"(\w+)(?:([\u2014\u2013])\s*|(-))(\w+)")


# Docling reads a dash at a line end as a hyphenation and glues the two words; the layer still has the dash
def restore_dashes(markdown: str, layer: str | None, marked: frozenset = frozenset()) -> tuple[str, int]:
    if not layer:
        return markdown, 0
    # a word the join made from a mid-line hyphen mark is no proof the layer writes it glued
    words = set(re.findall(r"\w+", layer)) - marked
    glued = {a + b: f"{a}{dash or hyphen}{b}" for a, dash, hyphen, b in _DASHED.findall(layer) if a + b not in words}
    # a dash Docling prints as a hyphen, where the layer never hyphenates the pair
    hyphened = {f"{a}-{b}" for a, _, hyphen, b in _DASHED.findall(layer) if hyphen}
    glued |= {f"{a}-{b}": f"{a}{dash}{b}" for a, dash, _, b in _DASHED.findall(layer)
              if dash and f"{a}-{b}" not in hyphened}
    if not glued:
        return markdown, 0
    count = 0

    def put_back(match):
        nonlocal count
        count += 1
        return glued[match.group(0)]

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(glued, key=len, reverse=True))) + r")\b")
    return pattern.sub(put_back, markdown), count


_MID_LINE_MARK = re.compile("(\\w+)\ufffe(\\w+)")


# the words a mid-line hyphen mark joins: soft hyphens mostly, a compound's own hyphen where the layer spells it so too
def marked_joins(layer: str | None) -> frozenset:
    return frozenset(a + b for a, b in _MID_LINE_MARK.findall(layer or ""))


# a space or a line end beside the hyphen: a hyphen inside a line (`well-known`) is the word's own
_BROKEN = re.compile(r"\b(\w+)(?: - ?| ?-\n ?|- )(\w+)\b")


def _letters(half: str) -> bool:
    return half.replace("_", "").isalpha()


# a word the converter left broken at its hyphenation («за - кономерн»), joined when the layer has it whole
def join_broken_words(markdown: str, layer: str | None) -> tuple[str, int]:
    if not layer:
        return markdown, 0
    words = set(re.findall(r"\w+", layer))
    count = 0

    def join(match):
        nonlocal count
        a, b = match.group(1), match.group(2)
        # letters or `_` on both sides, one half no word of its own: `N - 2` and `WordPress - based` stay apart
        broken = _letters(a) and _letters(b) and (a not in words or b not in words)
        if broken and a + b in words and f"{a}-{b}" not in layer and f"{a} - {b}" not in layer:
            count += 1
            return a + b
        return match.group(0)

    return _BROKEN.sub(join, markdown), count


_ESCAPED_BULLET = re.compile(r"^(\s*)\\([-*+]) ", re.M)


# MinerU escapes a list item's marker (`\- item`), and the list reads as a paragraph
def unescape_bullets(markdown: str) -> tuple[str, int]:
    return _ESCAPED_BULLET.subn(r"\1\2 ", markdown)


_INLINE_CODE = re.compile(r"(`[^`\n]*`)")


# Docling and MinerU write `_` outside code as `\_`, and a search for `AT_STATX_SYNC` then misses it
def unescape_underscores(markdown: str) -> tuple[str, int]:
    lines, inside, count = markdown.split("\n"), False, 0
    for n, line in enumerate(lines):
        if _FENCE_LINE.match(line):
            inside = not inside
            continue
        if inside or "\\_" not in line:
            continue
        parts = _INLINE_CODE.split(line)
        for k in range(0, len(parts), 2):
            count += parts[k].count("\\_")
            parts[k] = parts[k].replace("\\_", "_")
        lines[n] = "".join(parts)
    return "\n".join(lines), count


_EDGE_NUMBER = re.compile(r"^\W*\d+\W*|\W*\d+\W*$")
# a line standing first or last on this many pages of the layer is the book's running head, not a section
RUNNING_MIN_PAGES = 3


def bare_line(line: str) -> str:
    return " ".join(_EDGE_NUMBER.sub("", line).lower().split())


def running_heads(layers: list[str]) -> set[str]:
    edges: dict[str, int] = {}
    for layer in layers:
        lines = [line for line in layer.splitlines() if line.strip()]
        for line in {bare_line(line) for line in lines[:1] + lines[-1:]} - {""}:
            edges[line] = edges.get(line, 0) + 1
    # one word is a book's generic heading (Solution, Summary, a chapter label without its number), not its running head
    return {line for line, pages in edges.items() if pages >= RUNNING_MIN_PAGES and len(line.split()) > 1}


# the lines inside code fences blanked to spaces by the fence rule every repair reads, offsets kept for a checker
def unfenced(markdown: str) -> str:
    lines, inside = markdown.split("\n"), False
    for n, line in enumerate(lines):
        fence = bool(_FENCE_LINE.match(line))
        if inside or fence:
            lines[n] = " " * len(line)
        if fence:
            inside = not inside
    return "\n".join(lines)


# a running head Docling made a heading is dropped from its second time on; the first is the chapter's own title
def drop_running_headings(markdown: str, layer: str) -> tuple[str, int]:
    running = running_heads(layer.split("\f"))
    lines, inside, seen, keep, count = markdown.split("\n"), False, set(), [], 0
    for line in lines:
        if _FENCE_LINE.match(line):
            inside = not inside
        found = None if inside else _MD_HEADING.match(line)
        bare = bare_line(found.group(2)) if found else ""
        if bare in running and bare in seen:
            count += 1
            continue
        seen.add(bare)
        keep.append(line)
    return "\n".join(keep), count


# a figure, listing or table caption by its label and number; the repair and the layer checker read the same one
CAPTION = r"(?:Figure|Fig\.|Listing|Table|Рис\.|Рисунок|Листинг|Таблица)\s*\d"
_CAPTION_HEADING = re.compile(rf"^#{{1,6}}[ \t]+({CAPTION}.*)$", re.IGNORECASE)


# a figure, listing or table caption made a heading cuts a section in two for the chunker: it goes back to a line
def demote_caption_headings(markdown: str) -> tuple[str, int]:
    lines, inside, count = markdown.split("\n"), False, 0
    for n, line in enumerate(lines):
        if _FENCE_LINE.match(line):
            inside = not inside
            continue
        found = None if inside else _CAPTION_HEADING.match(line)
        if found:
            lines[n], count = found.group(1), count + 1
    return "\n".join(lines), count


_INLINE_PICTURE = re.compile(r"!\[([^\]]*)\]\(data:[^)]*\)")


# MinerU inlines a picture as base64; it becomes an address over the piece's pages, the page itself unknown
def inline_pictures(markdown: str, pages: tuple[int, int] | None) -> tuple[str, int]:
    where = f"pages{pages[0]}-{pages[1]}" if pages else "pages"
    count = 0

    def address(match):
        nonlocal count
        count += 1
        return f"![{match.group(1)}](picture:{where}-{count})"

    return _INLINE_PICTURE.sub(address, markdown), count


_SPLIT = re.compile(r"\b(\w+) (?=(\w+)\b)")


# a word the converter split with a space (a ligature «fi le», a first letter apart), joined when the layer has it whole
def join_split_words(markdown: str, layer: str | None) -> tuple[str, int]:
    if not layer:
        return markdown, 0
    words = set(re.findall(r"\w+", layer))
    count = 0

    def join(match):
        nonlocal count
        a, b = match.group(1), match.group(2)
        # a half of three letters or less that is no word of the layer is a fragment; two whole words stay apart
        fragment = any(len(x) <= 3 for x in (a, b) if x not in words)
        if a.isalpha() and b.isalpha() and a + b in words and fragment and f"{a} {b}" not in layer:
            count += 1
            return a
        return match.group(0)

    lines, inside = markdown.split("\n"), False
    for n, line in enumerate(lines):
        if _FENCE_LINE.match(line):
            inside = not inside
        elif not inside:
            parts = _INLINE_CODE.split(line)
            parts[::2] = [_SPLIT.sub(join, part) for part in parts[::2]]
            lines[n] = "".join(parts)
    return "\n".join(lines), count


# one underscore after a letter or digit: a dunder name (`__repr__`) ends a word, the next line is not its tail
_WRAPPED_UNDERSCORE = re.compile(r"(?<!\w)(\w*[^\W_]_)\r?\n[ \t]*(\w[\w.]*)")


# an identifier wrapped after its underscore: the converter reads the line end as a space, the layer keeps the wrap
def join_wrapped_identifiers(markdown: str, layer: str | None) -> tuple[str, int]:
    if not layer:
        return markdown, 0
    pairs = {f"{a} {b}": f"{a}{b}" for a, b in _WRAPPED_UNDERSCORE.findall(layer)}
    if not pairs:
        return markdown, 0
    pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, sorted(pairs, key=len, reverse=True))) + r")(?![\w])")
    return pattern.subn(lambda m: pairs[m.group(0)], markdown)


_NAME_SECTION = re.compile(r"^(#{1,5}) NAME[ \t]*\n+([^\n#]+)", re.M)


# a man page's title back over its NAME, from the page's first name: `dup, dup2 - duplicate` heads the page `dup`
def man_page_titles(markdown: str) -> tuple[str, int]:
    def title(found: re.Match) -> str:
        level, line = found.group(1), found.group(2)
        name = re.split(r"[,\s]", line.split(" - ", 1)[0].strip().lstrip("\\"), maxsplit=1)[0]
        # a NAME whose next line is another section's word lost its own line to the scan
        if not name or name.upper() == name and name.isalpha() or name.lower() == "name":
            return found.group(0)
        return f"{level[:-1] or '#'} {name}\n\n{level} NAME\n\n{line}"

    titled = [0]

    def counted(found: re.Match) -> str:
        new = title(found)
        titled[0] += new != found.group(0)
        return new

    return _NAME_SECTION.sub(counted, markdown), titled[0]
