import re
from dataclasses import dataclass
from pathlib import Path

import config
import logging_setup
from langchain_text_splitters import MarkdownHeaderTextSplitter

log = logging_setup.get_logger(__name__)

MAX_CHUNK_SIZE = config.settings.ingestion.chunk_max_size

# one owner for the cut that drops such blocks and the metric that counts them
BOILERPLATE_FILE_SHARE = 0.5
BOILERPLATE_MIN_FILES = 3


# the cut that drops these blocks and the metric that counts them built one spread twice
def wide_bodies(body_and_file, file_count: int) -> set[str]:
    spread: dict[str, set[str]] = {}
    for body, file in body_and_file:
        spread.setdefault(body, set()).add(file)
    return {
        body
        for body, seen in spread.items()
        if len(seen) / file_count >= BOILERPLATE_FILE_SHARE
    }

# the standard parser tracks fenced code, tilde fences and indented headings; ours did not
HEADERS = [("##", "h2"), ("###", "h3")]
PARSER = "langchain_markdown_header"


# only a run that cuts may claim it: read back later it names the wrong build
def parser_version() -> str:
    from importlib.metadata import version

    return f"{PARSER}/{version('langchain-text-splitters')}"
FENCE_LINE = re.compile(r"^\s{0,3}(```|~~~)")
HEADING_LINE = re.compile(r"^(###|##) ")
# the same share the coverage report calls "tiny": one number, declared once
SLIVER_SHARE = 0.1
# two ride on every chunk; over 1001 files 15652 headings, the longest 177, none over 200
HEADING_CAP = 512
# never collapsed, it is the axis variants compare on; live maximum 218 over both variants
SECTION_CAP = 512
# the point already measured spent the ceiling on the body, so that stays the default
BODY, CONTENT = "body", "content"


def _budget(ceiling: int, prefix: str, ceiling_on: str) -> int:
    return ceiling if ceiling_on == BODY else max(1, ceiling - len(prefix))


# from the variant's policy: the constant let a frozen variant declare a ceiling nothing read
def chunk_markdown(content, separator="\n## ", ceiling=None):
    if not content.strip():
        return []

    parts = content.split(separator)

    intro = parts[0]
    h1 = content.splitlines()[0]
    chunks = [intro] + [h1 + separator + part for part in parts[1:]]

    return split_all_by_size(chunks, ceiling)


def split_all_by_size(chunks, ceiling=None):
    result = []
    for chunk in chunks:
        result.extend(split_by_size(chunk, max_size=ceiling))
    return result


def split_by_size(text, separators=("\n\n", "\n", ". ", " "), max_size=None):
    max_size = max_size or MAX_CHUNK_SIZE
    if len(text) <= max_size:
        return [text]

    if not separators:
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    separator, rest = separators[0], separators[1:]
    result, current_chunk = [], ""

    for part in text.split(separator):
        candidate = f"{current_chunk}{separator}{part}" if current_chunk else part
        if len(candidate) <= max_size:
            current_chunk = candidate
        else:
            if current_chunk:
                result.append(current_chunk)
            if len(part) > max_size:
                result.extend(split_by_size(part, rest, max_size))
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        result.append(current_chunk)

    return result


@dataclass
class Cut:
    prefix: str
    body: str
    section: str | None
    # the structure the author wrote, or the counter: known here and nowhere else
    cut_by: str = "section"


# the cap belongs to the rendered text, never to the recorded path
def _one_line(text: str) -> str:
    return " ".join((text or "").split())[:HEADING_CAP]


# only the file's own H1 goes; a deeper leading heading is content
def _without_leading_h1(text: str) -> str:
    head, _, rest = text.partition("\n")
    return rest if head.lstrip().startswith("# ") else text


# which lines sit inside a fence, and where a fence that never closed was opened
def _fence_scan(lines: list[str]) -> tuple[set[int], int | None]:
    token, opened, inside = None, None, set()
    for i, line in enumerate(lines):
        found = FENCE_LINE.match(line)
        if found and token is None:
            token, opened = found.group(1), i
        elif found and found.group(1) == token:
            token, opened = None, None
        elif token is not None:
            inside.add(i)
    return inside, opened


def _heading_marks(content: str, file: str | None = None) -> list[tuple[int, str, str]]:
    lines = content.split("\n")
    inside, opened = _fence_scan(lines)
    if opened is not None:
        log.warning(
            "ingest.unbalanced_fence",
            file=file,
            headings="read without fences",
            opened_at_line=opened + 1,
        )
        return _headings_of(lines)

    # the parser merges repeated headings, so occurrences are counted on the file
    docs = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=True
    ).split_text(content)
    real = {
        (level, _printable(doc.metadata[key]))
        for doc in docs
        for level, key in HEADERS
        if doc.metadata.get(key)
    }
    # a fenced line matching a real heading passes the parser, so both must agree
    return [
        mark
        for mark in _headings_of(lines)
        if mark[0] not in inside and (mark[1], _printable(mark[2])) in real
    ]


def _headings_of(lines: list[str]) -> list[tuple[int, str, str]]:
    return [
        (i, found.group(1), line.lstrip()[len(found.group(1)) :].strip())
        for i, line in enumerate(lines)
        if (found := HEADING_LINE.match(line.lstrip()))
    ]


# the parser drops non-printables, so both sides are compared the same way
def _printable(text: str) -> str:
    return "".join(c for c in text if c.isprintable()).strip()


# heading, whole body, the head before the first subheading, and the subsections
def _sections(content: str, file=None) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    lines = content.split("\n")
    marks = _heading_marks(content, file)
    tops = [(-1, "")] + [(i, h) for i, level, h in marks if level == "##"]
    subs = [(i, h) for i, level, h in marks if level == "###"]

    out = []
    # subs are in file order, so each section takes the next slice: filtering was quadratic
    cursor = 0
    for n, (line_no, heading) in enumerate(tops):
        end = tops[n + 1][0] if n + 1 < len(tops) else len(lines)
        while cursor < len(subs) and subs[cursor][0] <= line_no:
            cursor += 1
        start = cursor
        while cursor < len(subs) and subs[cursor][0] < end:
            cursor += 1
        inside = subs[start:cursor]
        pieces = [
            (h, "\n".join(lines[i + 1 : (inside[k + 1][0] if k + 1 < len(inside) else end)]))
            for k, (i, h) in enumerate(inside)
        ]
        first = inside[0][0] if inside else end
        out.append(
            (
                heading,
                "\n".join(lines[line_no + 1 : end]),
                "\n".join(lines[line_no + 1 : first]),
                pieces,
            )
        )
    return out


# the sections both cutters walk, carrying the prefix and path their chunks will wear
def _bodied_sections(content, root, file):
    if not (content or "").strip():
        return
    for heading, body, head, subs in _sections(content, file):
        if not heading:
            body = _without_leading_h1(body)
        if not _has_text(body):
            continue
        prefix, path = _prefix_and_path((root or "").strip(), heading)
        yield heading, prefix, path, body, head, subs


# `cut_structured` is this plus a branch for sections that do not fit
def _by_size(prefix, body, path, ceiling, ceiling_on) -> list[Cut]:
    budget = _budget(ceiling, prefix, ceiling_on)
    pieces = _absorb_textless(split_by_size(body.strip(), max_size=budget), budget)
    return [
        Cut(prefix, piece, path or None, "size" if len(pieces) > 1 else "section")
        for piece in pieces
    ]


# the prefix repeats whole on every piece and is never cut itself
def cut_with_root(content, root, ceiling=None, ceiling_on=BODY, file=None) -> list[Cut]:
    ceiling = ceiling or MAX_CHUNK_SIZE
    cuts = []
    for _heading, prefix, path, body, _head, _subs in _bodied_sections(content, root, file):
        cuts.extend(_by_size(prefix, body, path, ceiling, ceiling_on))
    return cuts


# a block that is nothing but heading lines is the section that used to be dropped
def _has_text(body: str) -> bool:
    return any(
        line.strip() and not line.lstrip().startswith("#") for line in body.split("\n")
    )


# merged first and cut by size after: under the budget it would stand alone anyway
def _absorb_textless(pieces: list[str], budget: int) -> list[str]:
    merged: list[str] = []
    for piece in pieces:
        if merged and (not _has_text(piece) or not _has_text(merged[-1])):
            merged[-1] = f"{merged[-1]}\n\n{piece}"
            continue
        merged.append(piece)
    return [p for whole in merged for p in split_by_size(whole, max_size=budget)]


def _prefix_and_path(root: str, heading: str) -> tuple[str, str]:
    prefix = f"# {_one_line(root)}\n" if root else ""
    if not heading:
        return prefix, root[:SECTION_CAP]
    prefix += f"## {_one_line(heading)}\n"
    path = f"{root} > {heading}" if root else heading
    return prefix, path[:SECTION_CAP]


# structure first, size last: by subheadings only when it does not fit, by size after
def cut_structured(content, root, ceiling=None, ceiling_on=BODY, file=None) -> list[Cut]:
    ceiling = ceiling or MAX_CHUNK_SIZE
    cuts = []
    for heading, prefix, path, body, head, subs in _bodied_sections(content, root, file):
        if len(body.strip()) <= _budget(ceiling, prefix, ceiling_on) or not subs:
            cuts.extend(_by_size(prefix, body, path, ceiling, ceiling_on))
            continue
        # the intro carries the file's H1 in head too, and only body was stripped
        cuts.extend(
            _by_subsection(
                subs,
                head if heading else _without_leading_h1(head),
                prefix, path, ceiling, ceiling_on,
            )
        )
    return cuts


def _by_subsection(subs, head, prefix, path, ceiling, ceiling_on) -> list[Cut]:
    pieces = []
    head = head.strip()
    if head:
        split = split_by_size(head, max_size=_budget(ceiling, prefix, ceiling_on))
        pieces += [(prefix, "", p, "size" if len(split) > 1 else "subsection") for p in split]
    for sub, text in subs:
        if not text.strip():
            # a subheading with nothing under it is kept, and the merge below folds it in
            pieces.append((prefix, sub, f"### {sub}", "subsection"))
            continue
        deep = f"{prefix}### {_one_line(sub)}\n"
        split = split_by_size(text.strip(), max_size=_budget(ceiling, deep, ceiling_on))
        pieces += [(deep, sub, p, "size" if len(split) > 1 else "subsection") for p in split]
    return _merge_slivers(pieces, path, ceiling, ceiling_on)


# joins a neighbour inside the same section, and its heading goes back into the text
def _merge_slivers(pieces, path, ceiling: int, ceiling_on: str) -> list[Cut]:
    sliver = ceiling * SLIVER_SHARE
    out: list[Cut] = []
    for i, (prefix, heading, body, cut_by) in enumerate(pieces):
        # a piece with no text always tries to join; what it must not do is grow past the budget
        textless = not _has_text(body)
        joins = textless or len(body) < sliver
        if joins and not out and i + 1 < len(pieces):
            nxt = pieces[i + 1]
            carried = f"### {heading}\n{body}" if heading and not body.startswith("### ") else body
            joined = f"{carried}\n\n{nxt[2]}"
            if textless or len(joined) <= _budget(ceiling, nxt[0], ceiling_on):
                pieces[i + 1] = (nxt[0], nxt[1], joined, nxt[3])
                continue
        if joins and out:
            joined = (
                f"{out[-1].body}\n\n### {heading}\n{body}"
                if heading and not body.startswith("### ")
                else f"{out[-1].body}\n\n{body}"
            )
            if textless or len(joined) <= _budget(ceiling, out[-1].prefix, ceiling_on):
                out[-1] = Cut(out[-1].prefix, joined, path or None, out[-1].cut_by)
                continue
        out.append(Cut(prefix, body, path or None, cut_by))
    # cut back here, so the escape above cannot trade headings for a chunk over the ceiling
    return [
        Cut(cut.prefix, piece, cut.section, "size" if len(split) > 1 else cut.cut_by)
        for cut in out
        for split in [split_by_size(cut.body, max_size=_budget(ceiling, cut.prefix, ceiling_on))]
        for piece in split
    ]


# the same rule the backfill migration used, so re-indexing baseline does not lose the axis
def heading_path(chunk: str) -> str | None:
    lines = chunk.split("\n", 2)
    if len(lines) < 2 or not lines[0].startswith("# ") or not lines[1].startswith("## "):
        return None
    return " > ".join(re.sub(r"^#+\s*", "", line) for line in lines[:2])


def path_to_category(rel_path):
    parts = Path(rel_path).with_suffix("").parts
    return ".".join(re.sub(r"[^\w-]", "_", p) for p in parts)
