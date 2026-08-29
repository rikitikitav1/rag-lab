import re
from dataclasses import dataclass
from pathlib import Path

import config
import logging_setup
from langchain_text_splitters import MarkdownHeaderTextSplitter

log = logging_setup.get_logger(__name__)

MAX_CHUNK_SIZE = config.settings.ingestion.chunk_max_size

# what counts as a heading is the standard parser's answer, not ours: it tracks fenced
# code, tilde fences and indented headings, and our regex did none of the three
HEADERS = [("##", "h2"), ("###", "h3")]
PARSER = "langchain_markdown_header"


# which build of the parser is cutting right now. Only a run that cuts may claim it: read
# back from stored rows it would report the process taking the report, not the code that
# drew their boundaries, and baseline would claim a parser that never touched it
def parser_version() -> str:
    from importlib.metadata import version

    return f"{PARSER}/{version('langchain-text-splitters')}"
FENCE_LINE = re.compile(r"^\s{0,3}(```|~~~)")
HEADING_LINE = re.compile(r"^(###|##) ")
# the same share the coverage report calls "tiny": one number, declared once
SLIVER_SHARE = 0.1
# a heading longer than this is not a heading. Two of them ride on every chunk, so
# without a bound one long line in a third-party file multiplies over every piece.
# Measured over all 1001 files: 15652 headings, the longest 177, none over 200, so at 512
# nothing real is touched. At 120 the cap was truncating 49 of them, 16 of those questions
# of the criterion set
HEADING_CAP = 512
# the same bound for the recorded path, and a second name because the rule differs: the
# path is never collapsed, it is the axis variants are compared on and the gold matches
# it as a string. Live maximum is 218
SECTION_CAP = 512
# what the ceiling counts. The point already measured was cut with the ceiling spent on
# the body alone, so that stays the default and a variant asks for the other by name
BODY, CONTENT = "body", "content"


def _budget(ceiling: int, prefix: str, ceiling_on: str) -> int:
    return ceiling if ceiling_on == BODY else max(1, ceiling - len(prefix))


def chunk_markdown(content, separator="\n## "):
    if not content.strip():
        return []

    parts = content.split(separator)

    intro = parts[0]
    h1 = content.splitlines()[0]
    chunks = [intro] + [h1 + separator + part for part in parts[1:]]

    return split_all_by_size(chunks)


def split_all_by_size(chunks):
    result = []
    for chunk in chunks:
        result.extend(split_by_size(chunk))
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
    # what decided this boundary: the structure the author wrote, or the counter. Known
    # here and nowhere else, so it is carried rather than inferred from a length later
    cut_by: str = "section"


# the cap belongs to the text we render, never to the path we record: `section` is the
# axis every variant is compared on, baseline restores it uncut, and the gold matches it
# as a string. Whitespace is collapsed here and deliberately not in the path, so a heading
# with a tab keeps it in `section` while the rendered line does not
def _one_line(text: str) -> str:
    return " ".join((text or "").split())[:HEADING_CAP]


# only the file's own H1 goes: the declared root replaces it. A deeper leading heading
# is content, and on cheatsheets it is the first entry of the file
def _without_leading_h1(text: str) -> str:
    head, _, rest = text.partition("\n")
    return rest if head.lstrip().startswith("# ") else text


# the standard parser says where the headings are; the text between them is sliced from
# the file itself, so nothing the author wrote is rewritten on the way in. Rebuilding the
# body from the parser's own output would swap blank lines for hard breaks
# four files of 1010 open a code fence and never close it. Everything after that point
# reads as code to the parser, and numpy alone loses seven real questions that way. The
# missing bracket cannot be placed back (the text does not say where it belonged), so a
# file whose fences do not balance is read the old fence-blind way and says so out loud
# one scanner of the fence grammar: which lines are inside a fence, and where the fence
# that never closed was opened (None when the walk ended outside one)
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

    # what the parser reports is which texts are headings, not how many times each one
    # occurs: it merges consecutive pieces carrying the same heading into one, so two
    # sections with the same title arrive as one and the second boundary would be lost.
    # Occurrences are counted on the file, membership is the parser's answer
    docs = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=True
    ).split_text(content)
    real = {
        (level, _printable(doc.metadata[key]))
        for doc in docs
        for level, key in HEADERS
        if doc.metadata.get(key)
    }
    # membership is the parser's answer, but a fenced line whose text matches a real
    # heading elsewhere in the same file would pass it, so the fenced regions are taken
    # out as well: both have to agree before a line becomes a boundary
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


# the parser drops non-printable characters from the heading text it reports, and a tab
# is one, so its answer and the line it came from differ on a heading nobody would call
# unusual. Both sides are compared the same way
def _printable(text: str) -> str:
    return "".join(c for c in text if c.isprintable()).strip()


# (heading, body of the whole section, its head before the first subheading, and its
# subsections). The intro of the file comes
# first with an empty heading, and a section keeps its subsection text inside its body:
# the second level is spent only when the first one does not fit
def _sections(content: str, file=None) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    lines = content.split("\n")
    marks = _heading_marks(content, file)
    tops = [(-1, "")] + [(i, h) for i, level, h in marks if level == "##"]
    subs = [(i, h) for i, level, h in marks if level == "###"]

    out = []
    # subs are in file order, so each section takes the next slice of them rather than
    # filtering the whole list again: a file of many headings was quadratic in them
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


# the declared root, not the first line of the file. The prefix repeats whole on every
# piece and is never cut itself, so a variant declares whether the ceiling covers the
# body alone or the prefix with it
def cut_with_root(content, root, ceiling=None, ceiling_on=BODY, file=None) -> list[Cut]:
    if not (content or "").strip():
        return []
    ceiling = ceiling or MAX_CHUNK_SIZE
    root = (root or "").strip()

    cuts = []
    for heading, body, _, _subs in _sections(content, file):
        if not heading:
            body = _without_leading_h1(body)
        if not _has_text(body):
            continue
        prefix, path = _prefix_and_path(root, heading)
        budget = _budget(ceiling, prefix, ceiling_on)
        pieces = _absorb_textless(split_by_size(body.strip(), max_size=budget), budget)
        for piece in pieces:
            cuts.append(
                Cut(
                    prefix=prefix,
                    body=piece,
                    section=path or None,
                    cut_by="size" if len(pieces) > 1 else "section",
                )
            )
    return cuts


# the parser reports a heading only when something is written under it, so a heading with
# an empty section stays in the text as a line. A block that is nothing but such lines is
# the section that used to be dropped, and it still is
def _has_text(body: str) -> bool:
    return any(
        line.strip() and not line.lstrip().startswith("#") for line in body.split("\n")
    )


# a slice that is nothing but heading lines answers nothing, so it joins a neighbour and
# the join is cut by size afterwards: merging under the budget would leave it standing
# alone next to a full piece, and merging without one grows a chunk past the ceiling.
# What survives is a section whose whole text is headings, which has nothing to join
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


# structure first, size last: a section is cut by its subheadings only when it does not
# fit, and by size only when a subheading still does not. The prefix carries the whole
# path down to the subsection, because that is context for the embedder; section stays at
# the question level, because that is the axis every variant is compared on
def cut_structured(content, root, ceiling=None, ceiling_on=BODY, file=None) -> list[Cut]:
    if not (content or "").strip():
        return []
    ceiling = ceiling or MAX_CHUNK_SIZE
    root = (root or "").strip()

    cuts = []
    for heading, body, head, subs in _sections(content, file):
        if not heading:
            body = _without_leading_h1(body)
        if not _has_text(body):
            continue
        prefix, path = _prefix_and_path(root, heading)
        if len(body.strip()) <= _budget(ceiling, prefix, ceiling_on) or not subs:
            budget = _budget(ceiling, prefix, ceiling_on)
            pieces = _absorb_textless(split_by_size(body.strip(), max_size=budget), budget)
            cuts.extend(
                Cut(prefix, piece, path or None, "size" if len(pieces) > 1 else "section")
                for piece in pieces
            )
            continue
        # the intro keeps the file's own H1 in head as well as in body, and only body was
        # stripped, so an over-ceiling intro emitted the declared root and the file title
        # stacked in one chunk
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
            # a subheading with nothing under it still says something: it is kept as a
            # piece of its own and the merge below folds it into the one before it
            pieces.append((prefix, sub, f"### {sub}", "subsection"))
            continue
        deep = f"{prefix}### {_one_line(sub)}\n"
        split = split_by_size(text.strip(), max_size=_budget(ceiling, deep, ceiling_on))
        pieces += [(deep, sub, p, "size" if len(split) > 1 else "subsection") for p in split]
    return _merge_slivers(pieces, path, ceiling, ceiling_on)


# a subsection too short to answer anything joins its neighbour, and only a neighbour
# inside the same section. Its heading goes back into the text so nothing is lost
def _merge_slivers(pieces, path, ceiling: int, ceiling_on: str) -> list[Cut]:
    sliver = ceiling * SLIVER_SHARE
    out: list[Cut] = []
    for i, (prefix, heading, body, cut_by) in enumerate(pieces):
        # a piece with no text of its own always tries to join, whatever its length; what
        # it must not do is grow past the budget while doing it
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
    # a merge that carried headings past the budget is cut back to it here, so the escape
    # above cannot trade a chunk of headings for a chunk over the ceiling
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

    # for part in parts:
    #     if not re.fullmatch(r"[\w-]+", part):
    #         raise ValueError(f"Invalid path part: {part}")

    # return ".".join(parts)


if __name__ == "__main__":
    for file in Path("/notes").rglob("*.md"):
        print(path_to_category(file.relative_to("/notes")))
        print(chunk_markdown(file.read_text(encoding="utf-8")))
