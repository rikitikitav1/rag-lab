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
