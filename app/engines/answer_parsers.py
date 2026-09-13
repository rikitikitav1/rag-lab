import re
from dataclasses import dataclass

import token_fields


# what a broker left in an answer's text, cut before the judge, the guest or the agent reads it
@dataclass(frozen=True)
class Parsed:
    text: str | None
    reasoning_chars: int = 0
    # a thinking block with no end: cut by the length limit, or followed by a call the broker took out
    reasoning_unclosed: bool = False
    # the model wrote its call into the text too; the broker's `tool_calls` stay the only call read
    call_markup_in_content: bool = False
    leftover_markers: tuple[str, ...] = ()
    # open because the length limit ended the answer, not because a call followed the thinking
    reasoning_cut_by_length: bool = False


_BAR, _SEP = "\uff5c\uff5c", "\u2581"
_DEEPSEEK_BEGIN = f"<{_BAR}tool{_SEP}calls{_SEP}beginning{_BAR}>"
_DEEPSEEK_END = f"<{_BAR}tool{_SEP}calls{_SEP}end{_BAR}>"
_DEEPSEEK_CALLS = re.compile(f">?{re.escape(_DEEPSEEK_BEGIN)}.*?(?:{re.escape(_DEEPSEEK_END)}|\\Z)", re.S)
_MINIMAX_CALLS = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.S)
# what no clean answer carries; any of it after the cut means the row names the wrong parser
MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>", _DEEPSEEK_BEGIN, _DEEPSEEK_END)


def _calls(pattern):
    def cut(text: str, seen: dict) -> str:
        left, found = pattern.subn("", text)
        seen["call"] = seen["call"] or found > 0
        return left
    return cut


def _think_tags(text: str, seen: dict) -> str:
    start = text.find("<think>")
    if start < 0:
        return text
    end = text.find("</think>", start)
    if end < 0:
        seen["reasoning"] += len(text) - start - len("<think>")
        seen["unclosed"] = True
        return text[:start]
    seen["reasoning"] += end - start - len("<think>")
    return text[:start] + text[end + len("</think>"):]


# each has a version, and the stamp carries it: a changed cut is a changed reading of the answer
NONE = "none"
PARSERS = {
    NONE: (1, lambda text, seen: text),
    "deepseek_tools": (1, _calls(_DEEPSEEK_CALLS)),
    "minimax_tools": (1, _calls(_MINIMAX_CALLS)),
    "think_tags": (1, _think_tags),
}
# the call markup comes out first, or an open `<think>` would swallow the call with it
_ORDER = ("deepseek_tools", "minimax_tools", "think_tags", NONE)


def names(spec: str) -> tuple[str, ...]:
    parts = {p.strip() for p in (spec or "").split("+") if p.strip()}
    unknown = sorted(parts - set(PARSERS))
    if unknown or not parts:
        raise ValueError(
            f"unknown answer parser {', '.join(unknown) or repr(spec)}; known: {', '.join(sorted(PARSERS))}"
        )
    return tuple(sorted(parts, key=_ORDER.index))


def refuse_unknown(spec: str) -> None:
    names(spec)


def label(spec: str) -> str:
    return "+".join(f"{name}@{PARSERS[name][0]}" for name in names(spec))


NO_PARSER = label(NONE)


def parse(spec: str, content: str | None, reasoning_content: str | None = None,
          finish_reason: str | None = None) -> Parsed:
    seen = {"reasoning": len(reasoning_content or ""), "unclosed": False, "call": False}
    if content is None:
        return Parsed(None, seen["reasoning"])
    text = content
    for name in names(spec):
        text = PARSERS[name][1](text, seen)
    # trimmed only where something was cut: an answer nothing touched stays byte for byte
    if text != content:
        text = text.strip()
    return Parsed(
        text=text,
        reasoning_chars=seen["reasoning"],
        reasoning_unclosed=seen["unclosed"],
        call_markup_in_content=seen["call"],
        leftover_markers=tuple(m for m in MARKERS if m in text),
        reasoning_cut_by_length=seen["unclosed"] and token_fields.cut(finish_reason),
    )


# what a row keeps of the cut; nothing when nothing was cut, so a local answer's row stays as it was
def record(parsed: Parsed | None) -> dict | None:
    if parsed is None or not (parsed.reasoning_chars or parsed.reasoning_unclosed
                              or parsed.call_markup_in_content or parsed.leftover_markers):
        return None
    return {
        "reasoning_chars": parsed.reasoning_chars,
        "reasoning_unclosed": parsed.reasoning_unclosed,
        "reasoning_cut_by_length": parsed.reasoning_cut_by_length,
        "call_markup_in_content": parsed.call_markup_in_content,
        "leftover_markers": list(parsed.leftover_markers),
    }


# an agent answers over several hops, and its row keeps the cut of all of them together
def summarize(hops) -> dict | None:
    seen = [p for p in hops if p is not None]
    if not seen:
        return None
    return record(Parsed(
        text=None,
        reasoning_chars=sum(p.reasoning_chars for p in seen),
        reasoning_unclosed=any(p.reasoning_unclosed for p in seen),
        call_markup_in_content=any(p.call_markup_in_content for p in seen),
        leftover_markers=tuple(sorted({m for p in seen for m in p.leftover_markers})),
        reasoning_cut_by_length=any(p.reasoning_cut_by_length for p in seen),
    ))
