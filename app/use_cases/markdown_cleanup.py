import re

from use_cases.markup import FENCE_LINE, INLINE_CODE

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


_ESCAPED_BULLET = re.compile(r"^(\s*)\\([-*+]) ", re.M)


# MinerU escapes a list item's marker (`\- item`), and the list reads as a paragraph
def unescape_bullets(markdown: str) -> tuple[str, int]:
    return _ESCAPED_BULLET.subn(r"\1\2 ", markdown)


# Docling and MinerU write `_` outside code as `\_`, and a search for `AT_STATX_SYNC` then misses it
def unescape_underscores(markdown: str) -> tuple[str, int]:
    lines, inside, count = markdown.split("\n"), False, 0
    for n, line in enumerate(lines):
        if FENCE_LINE.match(line):
            inside = not inside
            continue
        if inside or "\\_" not in line:
            continue
        parts = INLINE_CODE.split(line)
        for k in range(0, len(parts), 2):
            count += parts[k].count("\\_")
            parts[k] = parts[k].replace("\\_", "_")
        lines[n] = "".join(parts)
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


_MDX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.S)
_MDX_MODULE_LINE = re.compile(r"^(?:import\s.+?\sfrom\s+['\"][^'\"]+['\"];?|export\s.*)$", re.M)
_COMPONENT_START = re.compile(r"</?[A-Z][\w.]*")


_FENCED = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[^\n]*$", re.M | re.S)


# a JSX tag from its `<` to the `>` closing it, props with braces and quotes skipped over; none past a blank line
def _tag_end(text: str, start: int) -> int | None:
    depth, quote = 0, None
    for i in range(start + 1, len(text)):
        c = text[i]
        if c == "\n" and text.startswith("\n\n", i):
            return None
        if quote:
            quote = None if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == ">" and depth == 0:
            return i + 1
    return None


# a `<Capital` that never closes is the text's own, as `a <B grade`, and stays
def _without_components(text: str) -> str:
    out, at = [], 0
    while found := _COMPONENT_START.search(text, at):
        end = _tag_end(text, found.start())
        out.append(text[at : found.start()] if end else text[at : found.end()])
        at = end or found.end()
    out.append(text[at:])
    return "".join(out)


def _prose(text: str) -> str:
    prose = _MDX_MODULE_LINE.sub("", _MDX_COMMENT.sub("", text))
    # inline code is the page's subject: `Map<String, Object>` keeps its type parameters
    spans = INLINE_CODE.split(prose)
    bare = "".join(s if k % 2 else _without_components(s) for k, s in enumerate(spans))
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"^[ \t]+$", "", bare, flags=re.M))


# MDX is markdown with JSX: its imports, comments and component tags go, the text a component wraps stays
def strip_mdx(text: str) -> str:
    out, at = [], 0
    for fence in _FENCED.finditer(text):
        out += [_prose(text[at : fence.start()]), fence.group()]
        at = fence.end()
    return "".join([*out, _prose(text[at:])])


# a markdown file's text as the stand reads it, the same at intake and at the index
def markdown_of(path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return strip_mdx(text) if path.suffix.lower() == ".mdx" else text


_FRONTMATTER = re.compile(r"\A(---|\+\+\+)[ \t]*\n(.*?)\n\1[ \t]*(?:\n|\Z)", re.S)
_FIRST_HEADING = re.compile(r"^#{1,6}[ \t]+\S.*$", re.M)


def _metadata(fence: str, block: str) -> dict | None:
    import tomllib

    import yaml

    try:
        meta = tomllib.loads(block) if fence == "+++" else yaml.safe_load(block)
    except (yaml.YAMLError, tomllib.TOMLDecodeError):
        return None
    return meta if isinstance(meta, dict) and meta else None


# the frontmatter keys sites render on the page as its lead; the rest is metadata for the site's build
_LEAD_KEYS = ("description", "summary", "excerpt", "intro", "short_description")


# a page's frontmatter is the site's metadata, not its text: its title and lead stay as words, the rest goes
def without_frontmatter(markdown: str) -> str:
    found = _FRONTMATTER.match(markdown)
    meta = _metadata(found.group(1), found.group(2)) if found else None
    if meta is None:
        return markdown
    body = markdown[found.end():].lstrip("\n")
    title = meta.get("title")
    heading = _FIRST_HEADING.search(body)
    if heading is None and isinstance(title, (str, int, float)) and str(title).strip():
        body = f"# {str(title).strip()}\n\n{body}"
        heading = _FIRST_HEADING.search(body)
    lead = list(dict.fromkeys(v.strip() for k in _LEAD_KEYS if isinstance(v := meta.get(k), str) and v.strip()))
    if lead:
        at = heading.end() if heading else 0
        body = f"{body[:at].rstrip()}\n\n{chr(10).join(lead)}\n\n{body[at:].lstrip()}".lstrip()
    return body


_TABLE_ROW = re.compile(r"^\s*\|")
_FENCE_OPEN = re.compile(r"^[ \t]*(```|~~~)")
_PADDING = re.compile(r" {2,}")


# a converter pads each table cell to its column's width: the spaces show nothing and fill a chunk's ceiling
def without_table_padding(markdown: str) -> str:
    lines, fence = markdown.split("\n"), None
    for n, line in enumerate(lines):
        if opened := _FENCE_OPEN.match(line):
            fence = None if fence == opened.group(1)[:3] else fence or opened.group(1)[:3]
        elif fence is None and _TABLE_ROW.match(line):
            spans = INLINE_CODE.split(line)
            lines[n] = "".join(s if k % 2 else _PADDING.sub(" ", s) for k, s in enumerate(spans)).rstrip()
    return "\n".join(lines)


# the text a markdown's chunks are cut from, one function for the index and for the raw report's verdict
def as_indexed(markdown: str) -> str:
    return without_table_padding(without_frontmatter(markdown.lstrip("\ufeff")))
