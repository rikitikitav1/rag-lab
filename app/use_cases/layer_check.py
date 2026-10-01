import difflib
import re
import unicodedata
from collections import Counter, defaultdict

from use_cases.heading_rules import CAPTION, bare_line, running_heads, unfenced
from use_cases.route import joined_hyphens, words

VERSION = 8
CHECKS = (
    "glued",
    "split",
    "broken",
    "run_together",
    "wrong_char",
    "unknown",
    "entities",
    "lone_pipes",
    "escapes",
    "private_use",
    "formulas",
    "images",
    "inline_pictures",
    "html_tags",
    "text_lost",
    "caption_headings",
    "running_headings",
)
# too noisy to be read as defects: a word the layer lacks altogether, a picture that carries no words
NOT_DEFECTS = frozenset({"unknown", "images"})
WORD_CHECKS = frozenset({"glued", "split", "broken", "run_together", "wrong_char", "unknown"})
TEXT_LOST_BELOW = 0.9
TEXT_LOST_MIN_WORDS = 20

_TOKEN = re.compile(r"\w+")
_INLINE = re.compile(r"`[^`\n]+`")
_PICTURE = re.compile(r"!\[(?:\\.|[^\]\\])*\]\([^)]*\)")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG = re.compile(r"</?[a-zA-Z]+\b[^>]*>")
_WRAPPER = re.compile(
    r"<(?:span|small|sup|sub|br)\b[^>]*>|</(?:span|small|sup|sub)>|<[a-z]+\s[^>]*\b(?:class|style)=[^>]*>"
)
_ENTITY = re.compile(r"&(?:lt|gt|amp|quot|#\d+);")
_ESCAPE = re.compile(r"\\[!-/:-@\[-`{-~]")
_DASHED = re.compile(r"(\w+)\s*[-\u2013\u2014]\s*(\w+)")
_BROKEN = re.compile(r"\b(\w+)(?: -|- | - )(\w+)\b")
_LINE_END_HYPHEN = re.compile(r"(\w+)[-\u2010\u00ad]\r?\n\s*(\w+)")
_LONE_PIPE = re.compile(r"^[ \t]*\|[ \t]*$", re.M)
_TABLE_ROW = re.compile(r"^[ \t]*\|.*$", re.M)
_LETTER_WORDS = frozenset("ивскуоая")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+)$", re.M)
_CAPTION = re.compile(CAPTION, re.I)



def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().replace("ё", "е")


def _tokens(text: str) -> list[str]:
    return [_norm(w) for w in _TOKEN.findall(text)]


# each page's piece of the markdown, found by aligning the markdown's words to the layer's words page by page
def page_slices(markdown: str, layers: list[str], first: int) -> list[dict]:
    md = [(m.start(), _norm(m.group())) for m in _TOKEN.finditer(markdown)]
    lay, page_of = [], []
    for n, text in enumerate(layers):
        found = _tokens(text)
        lay += found
        page_of += [first + n] * len(found)
    owner: list[int | None] = [None] * len(md)
    for a, b, size in difflib.SequenceMatcher(None, [w for _, w in md], lay).get_matching_blocks():
        for k in range(size):
            owner[a + k] = page_of[b + k]
    starts, last = {}, first
    for i, page in enumerate(owner):
        if page is None or page < last or page in starts:
            continue
        starts[page], last = md[i][0], page
    pages = [first + n for n in range(len(layers))]
    bounds = [0 if n == 0 else starts.get(page) for n, page in enumerate(pages)]
    out = []
    for n, page in enumerate(pages):
        if bounds[n] is None:
            out.append({"page": page, "start": None, "end": None, "layer": layers[n]})
            continue
        end = next((b for b in bounds[n + 1 :] if b is not None), len(markdown))
        out.append({"page": page, "start": bounds[n], "end": end, "layer": layers[n]})
    return out


def _vocabulary(layer: str) -> dict:
    lay = _norm(layer)
    vocab = {w.replace("ё", "е") for w in words(layer)} | {a + b for a, b in _LINE_END_HYPHEN.findall(lay)}
    near = defaultdict(set)
    for w in vocab:
        for i in range(len(w)):
            near[w[:i] + "?" + w[i + 1 :]].add(w)
    glued = {a + b for a, b in _DASHED.findall(lay) if a + b not in vocab and len(a) > 1 and len(b) > 2}
    return {"vocab": vocab, "near": near, "glued": glued, "lay": lay}


def _masked(markdown: str) -> str:
    def blank(match):
        return re.sub(r"[^\n]", " ", match.group(0))

    return _INLINE.sub(blank, unfenced(markdown))


# the fewest layer words a token splits into, by prefix; a long glued token stays linear, not exponential
def _splits_into(token: str, vocab: set) -> bool:
    fewest = [0] + [None] * len(token)
    for end in range(1, len(token) + 1):
        for start in range(end):
            piece = token[start:end]
            if fewest[start] is not None and piece in vocab and (len(piece) > 1 or piece in _LETTER_WORDS):
                count = fewest[start] + 1
                fewest[end] = count if fewest[end] is None else min(fewest[end], count)
    return fewest[-1] is not None and 2 <= fewest[-1] <= 12


def _word_checks(prose: str, ctx: dict, found: Counter) -> None:
    vocab, lay = ctx["vocab"], ctx["lay"]
    text = _norm(re.sub(r"\\(?=[!-/:-@\[-`{-~])", "", prose))
    tokens = [t for t in _TOKEN.findall(text) if any(c.isalpha() for c in t)]
    pairs = zip(tokens, tokens[1:], strict=False)
    found["split"] += sum(1 for a, b in pairs if a + b in vocab and (a not in vocab or b not in vocab))
    found["broken"] += sum(1 for a, b in _BROKEN.findall(text) if a + b in vocab and f"{a}-{b}" not in lay)
    for t in tokens:
        if t in vocab:
            continue
        if t in ctx["glued"]:
            found["glued"] += 1
        elif len(t) >= 4 and any(t[:i] + "?" + t[i + 1 :] in ctx["near"] for i in range(len(t))):
            found["wrong_char"] += 1
        elif len(t) >= 10 and _splits_into(t, vocab):
            found["run_together"] += 1
        elif len(t) > 2:
            found["unknown"] += 1


def _page_checks(raw: str, prose: str, ctx: dict | None) -> Counter:
    found = Counter()
    found["inline_pictures"] = len(re.findall(r"!\[[^\]]*\]\(data:", raw))
    found["formulas"] = raw.count("formula-not-decoded")
    found["images"] = raw.count("<!-- image -->")
    found["private_use"] = sum(1 for c in raw if "\ue000" <= c <= "\uf8ff" or c in "\ufffe\uffff")
    prose = _COMMENT.sub(" ", _PICTURE.sub(" ", prose))
    found["html_tags"] = len(_WRAPPER.findall(prose))
    prose = _TAG.sub(" ", prose)
    found["entities"] = len(_ENTITY.findall(prose))
    # a pipe inside a table cell is written `\|` or the row gains a column: that escape is the table's own
    found["escapes"] = len(_ESCAPE.findall(_TABLE_ROW.sub(lambda m: m.group(0).replace("\\|", " "), prose)))
    if ctx is not None:
        _word_checks(prose, ctx, found)
    return found


# a caption made a heading, or a running head made one again after its first time, the chapter's own title
def heading_defects(markdown: str, layers: list[str]) -> list[tuple[int, str]]:
    running = running_heads(layers)
    seen, found = set(), []
    # a heading line inside a code block is code, by the fence rule the repairs read
    for m in _HEADING.finditer(unfenced(markdown)):
        text, bare = m.group(1), bare_line(m.group(1))
        if _CAPTION.match(text.strip()):
            found.append((m.start(), "caption_headings"))
        elif bare in running and bare in seen:
            found.append((m.start(), "running_headings"))
        seen.add(bare)
    return found


def _lost(page_layer: str, near: str) -> bool:
    layer = Counter(_tokens(joined_hyphens(page_layer)))
    total = sum(layer.values())
    return total >= TEXT_LOST_MIN_WORDS and sum((layer & Counter(_tokens(near))).values()) / total < TEXT_LOST_BELOW


# a document's defects page by page against its own text layer; `trust_words` false skips the word checks
def check_document(markdown: str, layers: list[str], first: int, trust_words: bool = True) -> list[tuple[int, Counter]]:
    ctx = _vocabulary(joined_hyphens("\f".join(layers))) if trust_words else None
    prose = _masked(markdown)
    pipes = [m.start() for m in _LONE_PIPE.finditer(markdown)]
    suspects = heading_defects(markdown, layers)
    got = page_slices(markdown, layers, first)
    rows = []
    for n, page in enumerate(got):
        start, end = page["start"], page["end"]
        found = Counter() if start is None else _page_checks(markdown[start:end], prose[start:end], ctx)
        if start is not None:
            found["lone_pipes"] = sum(1 for at in pipes if start <= at < end)
            found.update(kind for at, kind in suspects if start <= at < end)
        near = "".join(markdown[g["start"] : g["end"]] for g in got[max(0, n - 1) : n + 2] if g["start"] is not None)
        if _lost(page["layer"], near):
            found["text_lost"] += 1
        rows.append((page["page"], found))
    return rows


def defects(found: Counter) -> int:
    return sum(n for name, n in found.items() if name not in NOT_DEFECTS)
