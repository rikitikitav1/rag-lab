"""One grader for every path that filters chunks: the graph node, the direct path and the bench."""

import json
import time

import llm
import logging_setup
import prompt_repo
from errors import StandFault
from models.registry import Purpose, Role

log = logging_setup.get_logger(__name__)

# the shape the canon asks for: one binary score, so an engine that honours a schema cannot ramble
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["relevant"],
}


# read once before a run, not per chunk: a missing prompt used to kill every row it reached
def system_prompt(version: int | None = None) -> str:
    try:
        if version is not None:
            return prompt_repo.template_of(Purpose.grade_chunk, version)
        return prompt_repo.active_template(Purpose.grade_chunk)
    except RuntimeError as e:
        raise StandFault(
            f"this run grades chunks and the stand has no active `grade.chunk` prompt: {e}"
        ) from e


# yes, no, or neither: a cut or garbled verdict keeps the chunk, because a grader is not a judge
def read_verdict(text: str) -> str | None:
    said = (text or "").strip()
    if said.startswith("{"):
        try:
            said = str(json.loads(said).get("relevant", ""))
        except (ValueError, AttributeError):
            return None
    said = said.strip().strip('"').lower()
    return said if said in ("yes", "no") else None


# a chunk is addressed by its file and its place in it; a bare number addresses nothing across hops
def address_of(chunk, nth: int) -> tuple[str, bool]:
    if chunk and chunk.get("source"):
        return f"{chunk['source']}#{chunk.get('chunk_index')}", True
    return str(nth), False


# the probability of the word the grader chose: with it the filter has a dial, without it a switch
def confidence(completion, verdict: str | None) -> float | None:
    for token in getattr(completion, "logprobs", None) or ():
        said = _word(token["token"])
        if said in ("yes", "no"):
            return token["top"].get(token["token"]) or token["top"].get(said)
    return None


def ask_door(asks: list):
    def ask(stage: str, key: str, system: str, user: str, schema=None):
        completion = llm.ask(
            system=system, user=user, role=str(Role.grading), schema=schema, logprobs=True
        )
        text = (completion.text or "").strip()
        asks.append({"stage": stage, "key": key, "text": text,
                     **_probability(completion, text)})
        return text

    return ask


# recorded beside the verdict, never instead of it: a missing probability is a fact, not a zero
def _probability(completion, text: str) -> dict:
    said = confidence(completion, read_verdict(text))
    out = {"p": said} if said is not None else {}
    both = both_words(completion)
    return out | ({"top": both} if both else {})


# a grammar masks everything but the two words, so `1 - p` of a `no` is not the mass of `yes`
def both_words(completion) -> dict | None:
    for token in getattr(completion, "logprobs", None) or ():
        if _word(token["token"]) not in ("yes", "no"):
            continue
        got: dict = {}
        for alternative, mass in (token.get("top") or {}).items():
            word = _word(alternative)
            if word in ("yes", "no"):
                got[word] = max(got.get(word, 0.0), mass)
        return got or None
    return None


def _word(token: str) -> str:
    return token.strip().strip('"').lower()


# the same digest the freeze wrote, so a moved chunk is caught before a verdict is spent on it
def digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def piece_verdict(question: str, piece: str, system: str, ask, key: str) -> str | None:
    user = f"Question: {question}\n\nPassage:\n{piece}"
    try:
        said = ask("grade", key, system, user, schema=VERDICT_SCHEMA)
    except (RuntimeError, llm.InputOverWindow) as e:
        log.error("grading.failed", chunk=key, error=str(e))
        return None
    return read_verdict(said)


# the one decision every path shares: which of these pieces the generator is allowed to read
def grade_pieces(question: str, pieces: list, chunks: list, system: str, ask, memo=None) -> dict:
    memo = memo if memo is not None else {}
    kept, unreadable, asked, order, started = [], 0, 0, [], time.perf_counter()
    for nth, piece in enumerate(pieces):
        address, stable = address_of(chunks[nth] if nth < len(chunks) else None, nth)
        order.append(address)
        if stable and address in memo:
            if memo[address] != "no":
                kept.append(nth)
            continue
        verdict = piece_verdict(question, piece, system, ask, address)
        asked += 1
        unreadable += verdict is None
        if stable:
            memo[address] = verdict or "unreadable"
        if verdict != "no":
            kept.append(nth)
    return {
        "kept": kept,
        "unreadable": unreadable,
        "asked": asked,
        "order": order,
        "dropped": [a for n, a in enumerate(order) if n not in set(kept)],
        "seconds": round(time.perf_counter() - started, 3),
    }
