import re
import unicodedata

from use_cases.markup import FENCE_LINE

_HEADING = re.compile(r"^(#{1,6}) ")


# sections an HTML page sets at its title's level (each section's h1) go one level down, where the chunker cuts
def one_title(markdown: str) -> tuple[str, int]:
    lines, fenced, marks = markdown.split("\n"), False, []
    for i, line in enumerate(lines):
        if FENCE_LINE.match(line):
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
    from use_cases.docling_structure import page_of, reading_order

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
        fence = bool(FENCE_LINE.match(line))
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
        if FENCE_LINE.match(line):
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
        if FENCE_LINE.match(line):
            inside = not inside
            continue
        found = None if inside else _CAPTION_HEADING.match(line)
        if found:
            lines[n], count = found.group(1), count + 1
    return "\n".join(lines), count


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
