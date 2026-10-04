import re

from use_cases.markup import INLINE_CODE, fence_scan

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
    lines, count = markdown.split("\n"), 0
    fences, inside, _ = fence_scan(lines)
    for n, line in enumerate(lines):
        if n in fences or n in inside or "\\_" not in line:
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
    lines = text.split("\n")
    fences, inside, _ = fence_scan(lines)
    out, prose = [], []
    for n, line in enumerate(lines):
        if n in fences or n in inside:
            out += [_prose("\n".join(prose))] if prose else []
            out.append(line)
            prose = []
        else:
            prose.append(line)
    out += [_prose("\n".join(prose))] if prose else []
    return "\n".join(out)


# a markdown file's text as the stand reads it, the same at intake and at the index
def markdown_of(path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return strip_mdx(text) if path.suffix.lower() == ".mdx" else text


_FRONTMATTER = re.compile(r"\A(---|\+\+\+)[ \t]*\n(.*?)\n\1[ \t]*(?:\n|\Z)", re.S)
_FIRST_HEADING = re.compile(r"^#{1,6}[ \t]+\S.*$", re.M)
_TOP_HEADING = re.compile(r"^#[ \t]+\S.*$", re.M)


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
def _titled(body: str, heading, title: str) -> bool:
    if heading is None:
        return False
    lines = body.split("\n")
    fences, inside, _ = fence_scan(lines)
    # an own top heading outside code is the page's title, even after an import line or a comment
    own_top = any(_TOP_HEADING.match(line) for n, line in enumerate(lines) if n not in fences and n not in inside)
    return own_top or heading.group().lstrip("#").strip() == title


def without_frontmatter(markdown: str) -> str:
    found = _FRONTMATTER.match(markdown)
    meta = _metadata(found.group(1), found.group(2)) if found else None
    if meta is None:
        return markdown
    body = markdown[found.end():].lstrip("\n")
    title = meta.get("title")
    heading = _FIRST_HEADING.search(body)
    # the page's title roots it unless the body has its own top heading outside code or repeats the title
    if isinstance(title, (str, int, float)) and str(title).strip() and not _titled(body, heading, str(title).strip()):
        body = f"# {str(title).strip()}\n\n{body}"
        heading = _FIRST_HEADING.search(body)
    lead = list(dict.fromkeys(v.strip() for k in _LEAD_KEYS if isinstance(v := meta.get(k), str) and v.strip()))
    if lead:
        at = heading.end() if heading else 0
        body = f"{body[:at].rstrip()}\n\n{chr(10).join(lead)}\n\n{body[at:].lstrip()}".lstrip()
    return body


_TABLE_ROW = re.compile(r"^\s*\|")
_PADDING = re.compile(r" {2,}")


# a converter pads each table cell to its column's width: the spaces show nothing and fill a chunk's ceiling
def without_table_padding(markdown: str) -> str:
    lines = markdown.split("\n")
    fences, inside, _ = fence_scan(lines)
    for n, line in enumerate(lines):
        if n not in fences and n not in inside and _TABLE_ROW.match(line):
            spans = INLINE_CODE.split(line)
            lines[n] = "".join(s if k % 2 else _PADDING.sub(" ", s) for k, s in enumerate(spans)).rstrip()
    return "\n".join(lines)


# Hugo shortcodes: `{{< name args >}}` and `{{% name args %}}`, a closing one with a slash before the name
_HUGO_COMMENT = re.compile(r"\{\{([<%])\s*comment\s*[>%]\}\}.*?\{\{[<%]\s*/comment\s*[>%]\}\}", re.S)
_HUGO_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)([\w-]+)((?:[^}]|\}(?!\}))*?)\s*[>%]\}\}")
_HUGO_ARG = re.compile(r'(\w+)="([^"]*)"|"([^"]*)"|(\S+)')
# the section headings a Kubernetes page names by key; another key reads as its words
_HUGO_HEADINGS = {"whatsnext": "What's next", "prerequisites": "Before you begin", "cleanup": "Clean up",
                  "seealso": "See also"}
_HUGO_FENCES = ("highlight", "code")


def _hugo_args(text: str) -> tuple[list[str], dict[str, str]]:
    positional, named = [], {}
    for key, value, quoted, bare in _HUGO_ARG.findall(text):
        if key:
            named[key] = value
        else:
            positional.append(quoted or bare)
    return positional, named


# what the site shows of a shortcode: a tab's label, a term, a heading, a fence; layout and embeds go, their body stays
def _hugo_rendered(match: re.Match, values: dict[str, str]) -> str:
    closing, name, (positional, named) = match.group(1), match.group(2), _hugo_args(match.group(3))
    first = positional[0] if positional else ""
    if name in _HUGO_FENCES:
        return "```" if closing else f"```{first}"
    if closing:
        return ""
    if name == "tab":
        label = named.get("name") or named.get("tabName") or first
        return f"\n{label}\n" if label else ""
    if name == "kbd":
        return first
    if name == "endpoint":
        return f"`{' '.join(positional)}`" if positional else ""
    if name == "glossary_tooltip":
        return named.get("text") or ""
    if name == "heading" and first:
        return f"\n## {_HUGO_HEADINGS.get(first, first.replace('-', ' ').replace('_', ' ').capitalize())}\n"
    if name == "feature-state" and named.get("state"):
        version = named.get("for_k8s_version") or named.get("for_version") or ""
        return f"FEATURE STATE: {f'Kubernetes {version} ' if version else ''}[{named['state']}]"
    if name in ("alert", "details", "collapsible"):
        title = named.get("title") or named.get("summary") or first
        return f"\n**{title}**\n" if title else ""
    # a site parameter the source declares; one it does not keeps its own word, so the sentence does not lose it
    if name in ("param", "skew"):
        return values.get(first, first)
    return ""


def without_hugo_shortcodes(text: str, values: dict[str, str] | None = None) -> str:
    return _HUGO_SHORTCODE.sub(lambda m: _hugo_rendered(m, values or {}), _HUGO_COMMENT.sub("", text))


# MDN's KumaScript macros: `{{name}}` or `{{name("arg", 'arg', 3)}}`, names in any case
_MDN_MACRO = re.compile(r"\{\{\s*([A-Za-z_][\w-]*)\s*(?:\(((?:[^{}]|\{(?!\{))*?)\))?\s*\}\}")
_MDN_ARG = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|(-?\d+(?:\.\d+)?)')
# a reference shows its own words: the label a page gives it, else the name; an API name reads as code
_MDN_CODE_REFS = {"domxref", "cssxref", "jsxref", "svgattr", "httpheader", "httpmethod", "webextapiref", "csp"}
_MDN_TAG_REFS = {"htmlelement", "svgelement", "mathmlelement"}
_MDN_BADGES = {"optional_inline": "(optional)", "readonlyinline": "(read-only)",
               "experimental_inline": "(experimental)", "deprecated_inline": "(deprecated)",
               "non-standard_inline": "(non-standard)", "securecontext_inline": "(secure context)",
               "availableinworkers": "Available in Web Workers."}


def _mdn_rendered(match: re.Match) -> str:
    name = match.group(1).lower()
    args = [a or b or c for a, b, c in _MDN_ARG.findall(match.group(2) or "")]
    shown = args[1] if len(args) > 1 and args[1] else (args[0] if args else "")
    if name in _MDN_BADGES:
        return _MDN_BADGES[name]
    if not shown:
        return ""
    if name in _MDN_CODE_REFS:
        return f"`{shown}`"
    if name in _MDN_TAG_REFS:
        return f"`<{args[0]}>`" if len(args) < 2 or not args[1] else shown
    if name in ("glossary", "httpstatus"):
        return shown
    if name == "rfc":
        return f"RFC {args[0]}"
    return ""


def without_mdn_macros(text: str) -> str:
    return _MDN_MACRO.sub(_mdn_rendered, text)


# a site's own markup rendered before the rules every page goes through, one renderer a name `vocabulary.Markup` allows
MARKUPS = {"hugo": without_hugo_shortcodes, "mdn": lambda text, values=None: without_mdn_macros(text)}


# a page's markup is rendered outside its code fences: a fenced example shows the markup as written
def _outside_fences(text: str, render) -> str:
    lines = text.split("\n")
    fences, inside, _ = fence_scan(lines)
    out, run = [], []
    for n, line in enumerate(lines):
        if n in fences or n in inside:
            if run:
                out.append(render("\n".join(run)))
                run = []
            out.append(line)
        else:
            run.append(line)
    if run:
        out.append(render("\n".join(run)))
    return "\n".join(out)


# the text a markdown's chunks are cut from, one function for the index and for the raw report's verdict
def as_indexed(markdown: str, markup: str | None = None, values: dict[str, str] | None = None) -> str:
    text = without_frontmatter(markdown.lstrip("\ufeff"))
    if markup:
        text = _outside_fences(text, lambda run: MARKUPS[markup](run, values))
    return without_table_padding(text)


_INHERITED = re.compile(
    r"^(?P<indent>[ \t]*)- \*inherited from the\* .*?\*[\w ]+ of\* (?:\[`[^`]*`\]\([^)]*\)|`[^`]*`)"
)


# Sphinx autodoc prints an inherited member's whole docs under every subclass: the pointer to the parent stays
def drop_inherited_members(markdown: str) -> tuple[str, int]:
    out, dropped, inside = [], 0, None
    for line in markdown.split("\n"):
        if inside is not None:
            if line.strip() and len(line) - len(line.lstrip()) > inside:
                continue
            inside = None
        if found := _INHERITED.match(line):
            out.append(found.group())
            inside, dropped = len(found.group("indent")), dropped + 1
            continue
        out.append(line)
    return "\n".join(out), dropped


# a page that prints the same code block again and again (a tutorial's sample data in every example) keeps it once
def drop_repeated_code(markdown: str) -> tuple[str, int]:
    lines = markdown.split("\n")
    fences, _, _ = fence_scan(lines)
    out, seen, dropped, n = [], set(), 0, 0
    opens = sorted(fences)
    closes = dict(zip(opens[::2], opens[1::2], strict=False))
    while n < len(lines):
        if n in closes:
            body = "\n".join(lines[n + 1 : closes[n]]).strip()
            if len(body) >= _REPEATED_CODE_MIN_CHARS and body in seen:
                dropped, n = dropped + 1, closes[n] + 1
                continue
            seen.add(body)
            out += lines[n : closes[n] + 1]
            n = closes[n] + 1
            continue
        out.append(lines[n])
        n += 1
    return "\n".join(out), dropped


# a short block (`npm start`, an import line) repeats by nature, and its repeat is no copy worth dropping
_REPEATED_CODE_MIN_CHARS = 80
