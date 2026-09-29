import re

NOT_IN_SECTION = "NOT IN SECTION"
# the share of the evidence's words the reader's quote must hold for the two to name one place
EVIDENCE_HELD = 0.5
# the reader loses a fact past about 2900 words of a chapter; 1500 answered most of 800, 1500 and 2500 on think-python
WINDOW_WORDS = 1500
WINDOW_OVERLAP = 200
# a window of links or code with no spaces between is long in tokens though short in words; it is cut again here
WINDOW_CHARS = 12000
_WORD = re.compile(r"\w+")
# a word this short is a particle or an article in both languages and names no place
_SHORTEST = 3

NO_ANSWER = "the reader found no answer in the section"
ELSEWHERE = "the reader quoted another place of the section"
UNREAD = "the reader's reply was empty or cut"
TOO_LONG = "the section is longer than the reader's window"
SECTION_GONE = "its section is no longer exported"
REFUSED_CALL = "the reader's engine refused the request"


# a number is kept at any length: for a fact about a limit the number is the evidence
def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.casefold()) if len(w) >= _SHORTEST or w.isdigit()]


# how much of the evidence the quote holds, and the F1 of the two word bags beside it
def overlap(evidence: str, quote: str) -> dict:
    said, told = _words(evidence), _words(quote)
    common = sum(min(said.count(w), told.count(w)) for w in set(said))
    held = common / len(said) if said else 0.0
    precision = common / len(told) if told else 0.0
    f1 = 2 * held * precision / (held + precision) if held + precision else 0.0
    return {"held": round(held, 3), "f1": round(f1, 3)}


# a long section read in overlapping windows, in order; the reader is never told where the evidence lies
def windows(text: str) -> list[str]:
    words = text.split()
    if len(words) <= WINDOW_WORDS:
        by_words = [text]
    else:
        step = WINDOW_WORDS - WINDOW_OVERLAP
        by_words = [" ".join(words[i:i + WINDOW_WORDS]) for i in range(0, len(words) - WINDOW_OVERLAP, step)]
    return [piece for window in by_words for piece in _by_chars(window)]


def _by_chars(window: str) -> list[str]:
    # measured as the words it holds: a table's line breaks and padding are no reason to cut it
    if len(" ".join(window.split())) <= WINDOW_CHARS:
        return [window]
    pieces, held, size = [], [], 0
    for word in window.split():
        if held and size + len(word) + 1 > WINDOW_CHARS:
            pieces.append(" ".join(held))
            held, size = [], 0
        held.append(word)
        size += len(word) + 1
    return [*pieces, " ".join(held)] if held else pieces


def user_turn(question: str, section: str, text: str) -> str:
    return f"Question: {question}\n\nSection path: {section}\n\nSection text:\n{text}"


# one question's word: answered and where, from the reader's reply and the generator's evidence
def verdict(reply: str | None, finish_reason: str | None, evidence: str) -> dict:
    said = (reply or "").strip()
    if not said or finish_reason == "length":
        return {"answerable": None, "why": UNREAD, "held": None, "f1": None}
    # the whole reply, not a substring: a quote about config sections may hold those words
    if said.strip(" .!\"'`").casefold() == NOT_IN_SECTION.casefold():
        return {"answerable": False, "why": NO_ANSWER, "held": None, "f1": None}
    measured = overlap(evidence, said)
    return {"answerable": True, "why": None if measured["held"] >= EVIDENCE_HELD else ELSEWHERE, **measured}


# refused only when both halves find no answer: one half alone says more about the reader's language than the question
def pair_outcome(verdicts: list[dict]) -> tuple[str, str | None]:
    if all(v["answerable"] is False for v in verdicts):
        return "refused", NO_ANSWER
    if all(v["answerable"] and v["why"] is None for v in verdicts):
        return "accepted", None
    return "undecided", next(v["why"] for v in verdicts if v["why"])
