from use_cases.markup import FENCE_LINE


def _open_fence(text: str) -> bool:
    return sum(1 for line in text.splitlines() if FENCE_LINE.match(line)) % 2 == 1


def _cells(line: str) -> int:
    return len(line.strip().strip("|").split("|"))


def _separator(line: str) -> bool:
    return line.strip().startswith("|") and set(line.strip()) <= set("|-: ")


# a code block the page break left open goes on in the next piece's first fence, which is dropped to join them
def _heal_fence(before: str, after: str) -> tuple[str, str, bool]:
    if not _open_fence(before):
        return before, after, False
    lines = after.lstrip("\n").splitlines()
    if lines and FENCE_LINE.match(lines[0]):
        return before.rstrip("\n"), "\n".join(lines[1:]), True
    return before.rstrip("\n") + "\n```", after, True


# a table the break cut in two: the second half's header is either the first half's again or a row read as a header
def _join_table(before: str, after: str, continued_rows: bool = False) -> tuple[str, str, bool, bool]:
    head = before.rstrip("\n").splitlines()
    tail = after.lstrip("\n").splitlines()
    if len(head) < 2 or len(tail) < 2 or not head[-1].strip().startswith("|") or not _separator(tail[1]):
        return before, after, False, False
    top = len(head) - 1
    while top > 0 and head[top - 1].strip().startswith("|"):
        top -= 1
    header = head[top]
    if _cells(tail[0]) != _cells(header):
        return before, after, False, False
    rest = tail[2:] if tail[0].split() == header.split() else [tail[0], *tail[2:]]
    head_text = before.rstrip("\n")
    # a row the page break cut goes on in a row whose first cell is empty: its cells finish the row above
    merged = _continued_row(head[-1], rest[0]) if continued_rows and rest else None
    if merged:
        head_text, rest = "\n".join([*before.rstrip("\n").splitlines()[:-1], merged]), rest[1:]
    return head_text, "\n".join(rest), True, bool(merged)


def _continued_row(above: str, row: str) -> str | None:
    if _separator(above) or not row.strip().startswith("|"):
        return None
    top, low = above.strip().strip("|").split("|"), row.strip().strip("|").split("|")
    if len(top) != len(low) or low[0].strip() or not any(cell.strip() for cell in low):
        return None
    cells = [f"{a.strip()} {b.strip()}".strip() for a, b in zip(top, low, strict=True)]
    return "| " + " | ".join(cells) + " |"


# a file's pieces joined in page order, with what the joins healed
def join(parts: list[str], continued_rows: bool = False) -> tuple[str, dict]:
    healed = {"fences": 0, "tables": 0, "rows_continued": 0}
    if not parts:
        return "", healed
    whole = parts[0]
    for part in parts[1:]:
        whole, part, fence = _heal_fence(whole, part)
        table = row = False
        if not fence:
            whole, part, table, row = _join_table(whole, part, continued_rows)
        healed["fences"] += fence
        healed["tables"] += table
        healed["rows_continued"] += row
        whole = whole + ("\n" if fence or table else "\n\n") + part
    return whole, healed


# a piece's tables as line ranges, fenced code left out
def table_spans(lines: list[str]) -> list[tuple[int, int]]:
    found, fenced, start = [], False, None
    for i, line in enumerate([*lines, ""]):
        if FENCE_LINE.match(line):
            fenced = not fenced
        is_row = not fenced and line.strip().startswith("|")
        if is_row and start is None:
            start = i
        elif not is_row and start is not None:
            found.append((start, i - 1))
            start = None
    return found


# tables the structure says go on over a page break, joined when only blank lines lie between them in the markdown
def join_tables(markdown: str, goes_on: list[bool], continued_rows: bool = False) -> tuple[str, int, int]:
    lines = markdown.split("\n")
    tables = table_spans(lines)
    if len(tables) != len(goes_on):
        return markdown, 0, 0
    joined = rows = 0
    for k in range(len(tables) - 2, -1, -1):
        (start, end), (next_start, next_end) = tables[k], tables[k + 1]
        if not goes_on[k] or any(line.strip() for line in lines[end + 1 : next_start]):
            continue
        first, second = "\n".join(lines[start : end + 1]), "\n".join(lines[next_start : next_end + 1])
        before, after, merged, row = _join_table(first, second, continued_rows)
        if merged:
            lines[start : next_end + 1] = (before + ("\n" + after if after else "")).split("\n")
            joined += 1
            rows += row
    return "\n".join(lines), joined, rows


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
    from use_cases.docling_structure import page_of, reading_order

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


