import re

from use_cases.markup import FENCE_LINE, HYPHEN_MARK, INLINE_CODE

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


_MID_LINE_MARK = re.compile(rf"(\w+){HYPHEN_MARK}(\w+)")


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
        if FENCE_LINE.match(line):
            inside = not inside
        elif not inside:
            parts = INLINE_CODE.split(line)
            parts[::2] = [_SPLIT.sub(join, part) for part in parts[::2]]
            lines[n] = "".join(parts)
    return "\n".join(lines), count

