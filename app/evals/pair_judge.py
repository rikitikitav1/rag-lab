import re

# the passage the judge reads: the evidence with this many words on each side, not the whole section
AROUND_WORDS = 150
# the evidence is found by its first words, whatever markup or spacing the section puts between them
_LEADING_WORDS = 8
_WORD = re.compile(r"\w+")

SPLIT = "the judge answered one half of the pair and not the other"
NOT_ANSWERED = "the judge found the evidence does not answer the question"
UNREAD = "the judge's reply was not YES or NO"
NOT_FOUND = "the evidence was not found in its section"
REFUSED_CALL = "the judge's engine refused the request"


# where a quote sits in a text: the occurrence of its first words that holds the most of its words after it
def locate(text: str, evidence: str) -> int | None:
    words = _WORD.findall(evidence)[:_LEADING_WORDS]
    if not words:
        return None
    starts = list(re.finditer(r"[\W_]+".join(map(re.escape, words)), text, re.IGNORECASE))
    if not starts:
        return None
    # a page built from a template repeats a quote's first words; the right place holds the rest of it too
    wanted = [w.casefold() for w in _WORD.findall(evidence)]
    span = len(evidence) * 2

    def held(match) -> int:
        near = {w.casefold() for w in _WORD.findall(text[match.start(): match.start() + span])}
        return sum(1 for w in wanted if w in near)

    return max(starts, key=held).start()


def around(text: str, evidence: str, at: int | None = None) -> str | None:
    start = locate(text, evidence) if at is None else at
    if start is None:
        return None
    before = text[:start].split()[-AROUND_WORDS:]
    after = text[start:].split()[: AROUND_WORDS + len(evidence.split())]
    return " ".join(before + after)


def user_turn(question: str, evidence: str, passage: str) -> str:
    return f"Question: {question}\n\nEvidence: {evidence}\n\nPassage:\n{passage}"


def verdict(reply: str | None) -> bool | None:
    word = (reply or "").strip().strip(".!\"'`*").split()
    head = word[0].upper() if word else ""
    return True if head == "YES" else False if head == "NO" else None


# accepted when both halves are answered, refused when neither is: the same rule the sieve keeps
def pair_outcome(said: list[bool | None]) -> tuple[str, str | None]:
    if any(v is None for v in said):
        return "undecided", UNREAD
    if all(said):
        return "accepted", None
    if not any(said):
        return "refused", NOT_ANSWERED
    return "undecided", SPLIT
