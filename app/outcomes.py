import re
from enum import StrEnum

NO_RESULTS = "No relevant documents found."

REFUSAL_MARKERS = (
    "cannot answer", "can't answer", "cannot provide", "unable to find", "could not find",
    "couldn't find", "no relevant", "not able to answer", "cannot be answered",
    "не могу ответить", "не удалось найти", "не нашёл", "не нашел", "нет информации",
    "не располагаю",
)

# these refuse only when the answer blames the sources: "not found" alone is prose
MISSING_MARKERS = (
    "no information", "not found", "don't have", "do not have", "не удалось", "не найдено",
    "нет данных", "не содержится", "отсутствует",
)
SOURCE_MARKERS = (
    "source", "available", "corpus", "context", "документ", "источник", "материал", "корпус",
)
# "does not contain" is prose about anything: it refuses only when the sources lack the answer itself
SOURCES_LACK = re.compile(
    r"(контекст\w*|источник\w*|материал\w*|документ\w*|фрагмент\w*) не содерж\w* (информаци|сведени|ответ|данны|упоминани)"
    r"|(context|sources?|documents?|materials?|passages?) (does not|doesn't|do not|don't) contain"
    r" (any |enough |the |specific |relevant )?(information|details|answer|data|mention)"
)


class Outcome(StrEnum):
    # a stored `answered` is the pre-judge reading: only a recomputed one can be ungrounded
    answered = "answered"
    # sources came back and nothing in the answer was grounded in them
    answered_ungrounded = "answered_ungrounded"
    refused = "refused"
    unsupported_answer = "unsupported_answer"
    narrated_call = "narrated_call"
    exhausted = "exhausted"
    error = "error"


def looks_like_raw_call(text: str) -> bool:
    head = text.lstrip()[:400]
    return '"name"' in head and ('"parameters"' in head or '"arguments"' in head)


def narrated_tool_call(text: str | None, names=(), prefixes=()) -> bool:
    if not text:
        return False
    head = text[:400]
    if looks_like_raw_call(text) or any(
        f"{name}(" in head or f'"{name}"' in head for name in names
    ):
        return True
    return any(
        re.search(rf'{re.escape(p)}\w+\s*\(|"{re.escape(p)}\w+"', head) for p in prefixes
    )


REFUSAL_MAX_CHARS = 400

# outcomes are read from the text on every read, so a cited refusal share carries the rule it was read with
RULE = 3


def refusal(text: str) -> bool:
    stripped = text.strip()
    if stripped == NO_RESULTS:
        return True
    if len(stripped) > REFUSAL_MAX_CHARS:
        return False
    lowered = stripped.lower()
    if any(m in lowered for m in REFUSAL_MARKERS) or SOURCES_LACK.search(lowered):
        return True
    return any(m in lowered for m in MISSING_MARKERS) and any(
        s in lowered for s in SOURCE_MARKERS
    )


# the half of a classification that never moves: it reads the text and the tool names, never a judge
def reads_as_refusal(text: str | None, names=(), prefixes=()) -> bool:
    return bool(text) and not narrated_tool_call(text, names, prefixes) and refusal(text)


# `grounded` is None where nothing judged the row, and such a row stays `answered`


def classify(
    text: str | None, has_sources: bool, names=(), prefixes=(), exhausted=False, grounded=None
) -> str:
    if not text:
        return Outcome.exhausted if exhausted else Outcome.error
    if narrated_tool_call(text, names, prefixes):
        return Outcome.narrated_call
    if refusal(text):
        return Outcome.refused
    if has_sources:
        return Outcome.answered if grounded is not False else Outcome.answered_ungrounded
    return Outcome.unsupported_answer
