import re

# a fenced code block whole, its body in the group
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
# a line that opens or closes a fence, ``` or ~~~; a fence closes only on its own token
FENCE_LINE = re.compile(r"^\s*(```|~~~)")
# the chunker reads a fence indented up to three spaces only, as CommonMark does
CHUNKER_FENCE_INDENT = 3


# the one fence rule: which lines open or close a fence, which sit inside one, and where one left open began
def fence_scan(lines: list[str], max_indent: int | None = None) -> tuple[set[int], set[int], int | None]:
    token, opened, fences, inside = None, None, set(), set()
    for i, line in enumerate(lines):
        found = FENCE_LINE.match(line)
        if found and max_indent is not None and len(line) - len(line.lstrip()) > max_indent:
            found = None
        if found and token is None:
            token, opened = found.group(1), i
            fences.add(i)
        elif found and found.group(1) == token:
            token, opened = None, None
            fences.add(i)
        elif token is not None:
            inside.add(i)
    return fences, inside, opened
# an inline code span, kept whole by a split so the prose around it can be rewritten alone
INLINE_CODE = re.compile(r"(`[^`\n]*`)")
# PDFium's mark for a hyphen it took for a word break; the route, the word rules and the code rows all read it
HYPHEN_MARK = "￾"
