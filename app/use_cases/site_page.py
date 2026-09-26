import re

import formats

# DocBook's sectioning vocabulary, from the format's file: the element a DocBook page's own text lives in
DOCBOOK_SECTIONS = "|".join(map(re.escape, formats.format_of("docbook").sections))
_DOCBOOK_SECTION_NAME = re.compile(DOCBOOK_SECTIONS)
_DIV = re.compile(r"<(/?)div\b([^>]*)>", re.I)
_CLASS = re.compile(r'class="([^"]*)"')
_SELECTOR = re.compile(r"^(\w+)(?:([.#])([\w-]+)|~(.+))$")
_NOT_TEXT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_ATTR = r"""\b{}\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))"""
_PRE = re.compile(r"(<pre\b[^>]*>)(.*?)(</pre>)", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


# a DocBook page's own text is its first sectioning element, whatever the site wraps around it
def main_element(page: str) -> str | None:
    depth, start = 0, None
    for m in _DIV.finditer(page):
        if start is None:
            classes = (_CLASS.search(m.group(2)) or [None, ""])[1].split()
            if not m.group(1) and any(_DOCBOOK_SECTION_NAME.fullmatch(c) for c in classes):
                start, depth = m.start(), 1
            continue
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return page[start : m.end()]
    return None


# the element a site's selector names, whole, by counting its own tag open and closed; None when absent
def element(page: str, selector: str) -> str | None:
    if selector == "docbook":
        return main_element(page)
    return next((page[a:b] for a, b in spans(page, selector)), None)


# every element a selector names, as spans of the page, outermost first; tag~text is the tag that holds that text
def spans(page: str, selector: str):
    tag, mark, name, text = _SELECTOR.match(selector).groups()
    attr = re.compile(_ATTR.format("class" if mark == "." else "id"), re.I) if mark else None
    opened = []
    for m in re.finditer(rf"<(/?){tag}\b([^>]*)>", page, re.I):
        if not m.group(1):
            found = attr.search(m.group(2)) if attr else None
            value = next((g for g in found.groups() if g is not None), "") if found else ""
            opened.append((m.start(), text is not None or (name in value.split() if mark == "." else value == name)))
            continue
        if not opened:
            continue
        start, wanted = opened.pop()
        if wanted and (text is None or text in page[start : m.end()]) and not any(w for _, w in opened):
            yield start, m.end()


# the site's furniture and anything that is not text go before a tool or pandoc reads the element
def dropped(fragment: str, drop: list[str]) -> str:
    fragment = _NOT_TEXT.sub("", fragment)
    for selector in drop:
        for a, b in sorted(spans(fragment, selector), reverse=True):
            fragment = fragment[:a] + fragment[b:]
    return fragment


# highlighted code is a span a token; a tool that joins spans with spaces splits identifiers, so they go first
def flat_pre(fragment: str) -> str:
    return _PRE.sub(lambda m: m.group(1) + "<code>" + _TAG.sub("", m.group(2)) + "</code>" + m.group(3), fragment)


# a page as its own text: the site's element, its furniture dropped, its highlighting flat; None when absent
def prepared(page: str, main: str, drop: list[str]) -> str | None:
    found = element(page, main)
    if found is None:
        return None
    body = flat_pre(dropped(found, drop))
    return f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>{body}</body></html>'
