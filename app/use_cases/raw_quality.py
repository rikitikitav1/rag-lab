import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

import config
from corpus_keys import SECTION_SEP, chapter_of
from sources.base import cuts_of, first_heading, hygienic
from tool_names import Tool
from use_cases import ingest_quality as quality
from use_cases.code_lines import FENCE
from use_cases.route import mixed_share, words

# an HTML tag by its name, on one line; a bare `<` in code (`a < b`, `<%= %>`, JSX) is text and stays
_MARKUP = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>\n]*)?/?>|!\[[^\]\n]*\]\([^)\n]*\)|\[([^\]\n]*)\]\([^)\n]*\)")
_LINE_MARKS = re.compile(r"^[#>*\-+|\s]+|[*_`|]+", re.M)
# a converter escapes markdown's own characters; the backslash is its mark, never the page's text
_ESCAPES = re.compile(r"\\+")
# the minus sign is a math symbol by category, yet a page sets it where the source has a hyphen
_MINUS = "\u2212"


# a signal the report carries, which way is better, and what it is read from
@dataclass(frozen=True)
class Signal:
    better: str
    source: str


# every name a raw source's row may carry; a loop reads these and nothing it would have to invent
SIGNALS = {
    "chunks": Signal("none", "chunker"),
    "section_coverage": Signal("higher", "chunker"),
    **{
        name: Signal("lower", "chunker")
        for name in (
            "prefix_dominates",
            "dup_in_file",
            "dup_in_source",
            "tiny",
            "boilerplate",
            "orphans",
            "size_cut",
            "soup",
            "code_only",
        )
    },
    "layer_words": Signal("none", "layer"),
    "output_words": Signal("none", "output"),
    "output_share": Signal("band", "layer"),
    "layer_f1": Signal("higher", "layer"),
    "mixed_script": Signal("lower", "output"),
    "seconds": Signal("lower", "engine"),
}


# typography is the page's, not the tool's: every quote one form and every dash a hyphen, by Unicode category
def typography(text: str) -> str:
    def one(c):
        category = unicodedata.category(c)
        if category in ("Pi", "Pf") or c in "'\"":
            return '"'
        return "-" if category == "Pd" or c == _MINUS else c

    return "".join(one(c) if not c.isalnum() and not c.isspace() else c for c in text)


# markdown as the words a reader sees; a character-level score folds typography, a word count needs not
def plain(markdown: str, fold_typography: bool = False, keep_escapes: bool = False) -> str:
    text = unicodedata.normalize("NFKC", markdown)
    text = FENCE.sub(lambda m: m.group(1), typography(text) if fold_typography else text)
    text = _MARKUP.sub(r"\1", text)
    text = _LINE_MARKS.sub(" ", text if keep_escapes else _ESCAPES.sub(" ", text))
    return " ".join(text.split())


def _policy() -> dict:
    return config.settings.corpus.policy(config.settings.corpus.variant)


# a file's markdown cut whole, as the index will cut it: a piece cut alone takes its own first heading for the root
def _samples(markdown: str, file: str, policy: dict) -> list:
    text = markdown.lstrip("\ufeff")
    root = first_heading(text) if hygienic(policy) else None
    return [
        quality.Sample(file=file, content=content, chunk_index=i, body=body, section=section, root=root, cut_by=cut_by)
        for i, (content, body, section, root, cut_by) in enumerate(cuts_of(text, root, policy, file))
    ]


def _gates(samples: list, policy: dict) -> dict:
    metrics = quality.measure(samples, ceiling=policy["max_chunk_size"], records_sections=hygienic(policy))
    hard, soft, _, verdict = quality.gates_of(metrics, config.settings.ingest_quality)
    return {"verdict": verdict, "hard": hard, "soft": soft, "metrics": asdict(metrics)}


# the stand's own cut and gates over a whole markdown, as the corpus would meet it
def chunker_gates(markdown: str, file: str) -> dict:
    policy = _policy()
    return _gates(_samples(markdown, file, policy), policy)


# the chunker's gates a chapter of a file, a chapter the second step of the section path; no chapters, one row
def section_rows(markdown: str, file: str) -> list[dict]:
    policy = _policy()
    chapters: dict[str | None, list] = {}
    for sample in _samples(markdown, file, policy):
        chapters.setdefault(chapter_of(sample.section), []).append(sample)
    rows = []
    for chapter, samples in chapters.items():
        gates = _gates(samples, policy)
        # a chapter that is the file's root alone has no heading below it by shape, so coverage measures nothing there
        root_only = chapter is None or SECTION_SEP not in chapter
        hard = [g for g in gates["hard"] if not (root_only and g.startswith("section_coverage."))]
        verdict = gates["verdict"] if hard == gates["hard"] else quality.verdict(hard, gates["soft"], judged=True)
        rows.append(
            {
                "file": file,
                "section": chapter,
                "words": len(words(" ".join(s.body if s.body is not None else s.content for s in samples))),
                **gates["metrics"],
                "chunker_verdict": verdict,
                "breached": hard + gates["soft"],
                "coverage_by_shape": root_only,
            }
        )
    return rows


# what the output says against the text the file carried with it; no layer, no layer signals
def conversion_signals(markdown: str, layer_text: str | None) -> dict:
    out = words(plain(markdown))
    signals = {
        "output_words": len(out),
        "mixed_script": mixed_share(out),
        "layer_words": None,
        "output_share": None,
        "layer_f1": None,
    }
    if layer_text is None:
        return signals
    layer = words(layer_text)
    signals["layer_words"] = len(layer)
    if not layer:
        return signals
    common = sum((Counter(out) & Counter(layer)).values())
    precision = common / len(out) if out else 0.0
    recall = common / len(layer)
    signals["output_share"] = round(len(out) / len(layer), 4)
    signals["layer_f1"] = round(2 * precision * recall / (precision + recall), 4) if common else 0.0
    return signals


# the conversion's own gates, named like the chunker's so a row carries one list of what it breached
def conversion_breaches(signals: dict, engine: str | None = Tool.docling) -> list[str]:
    rule = config.settings.intake.quality
    breached = []
    if signals["output_words"] == 0:
        breached.append("output_words.empty")
    # the layer band was read on Docling's outputs only; another engine's layer signals are kept and not judged
    judged = engine in rule.layer_band_engines
    share = signals["output_share"]
    if judged and share is not None and not rule.output_share_min <= share <= rule.output_share_max:
        breached.append("output_share.band")
    if judged and signals["layer_f1"] is not None and signals["layer_f1"] < rule.layer_f1_min:
        breached.append("layer_f1.min")
    if signals["mixed_script"] is not None and signals["mixed_script"] > rule.mixed_script_max:
        breached.append("mixed_script.max")
    return breached


# one piece of a raw source's conversion: the unit, the arm that made it, the conversion's signals, what it breached
def unit_row(markdown: str, layer_text: str | None, unit: dict, arm: dict, seconds: float | None) -> dict:
    signals = conversion_signals(markdown, layer_text)
    return {**unit, **arm, **signals, "seconds": seconds, "breached": conversion_breaches(signals, arm.get("engine"))}


_CELL = re.compile(r"<t[dh][^>]*>\s*[^<\s]")


# filled cells of the markdown's tables, pipe and HTML alike; a reading that merges columns has fewer of them
def table_cells(markdown: str) -> int:
    pipe = 0
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("|") and not set(line) <= set("|-: "):
            pipe += sum(1 for cell in line.strip("|").split("|") if cell.strip())
    return pipe + len(_CELL.findall(markdown))


# a second reading replaces the first when it agrees better with the layer and keeps every filled table cell
def better_reading(first: dict, second: dict, cells_slack: float = 0.0) -> bool:
    if first["layer_f1"] is None or second["layer_f1"] is None:
        return False
    return second["layer_f1"] > first["layer_f1"] and second["table_cells"] >= first["table_cells"] * (1 - cells_slack)


_FENCE_LINE = re.compile(r"^\s*```", re.M)
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.M)


def _title(text: str) -> str:
    return " ".join(words(text))


# an outline title is found when a heading holds it, or it holds a heading of two words or more («Chapter 2 Foo»)
def _found(title: str, headings: list[str]) -> bool:
    return any(title in h or (h.count(" ") and h in title) for h in headings)


# what a file's markdown says about itself with no gold: its outline found as headings, fences, one-line code
def self_check(markdown: str, outline: list[str]) -> dict:
    headings = [t for t in map(_title, _HEADING.findall(markdown)) if t]
    titles = [t for t in map(_title, outline) if t]
    blocks = [b.strip() for b in FENCE.findall(markdown) if b.strip()]
    return {
        "outline": len(titles),
        "outline_found": sum(_found(t, headings) for t in titles),
        "fences_unbalanced": len(_FENCE_LINE.findall(markdown)) % 2,
        "code_blocks": len(blocks),
        "code_one_line": sum("\n" not in b for b in blocks),
    }
