import hashlib
import json
import re

import config
from corpus_keys import anchors, language_by_alphabet, leaf_of, short_hash, spaceless_key, unlinked
from models.eval import CANDIDATE, text_hash

# a section this short is a stub: one sentence gave a dozen questions on the same fact in the probe of the generators
MIN_WORDS = 60
# the generator quotes its evidence; a quote longer than this is a paragraph copied, not a pointer
MAX_EVIDENCE_WORDS = 40
# pairs one call is asked for: the generator thinks about 3000 tokens a pair, and ten in one reply were cut
PAIRS_PER_CALL = 2
REFUSED_CALL = "the generator refused the call"
PAST_THE_ASK = "pairs past the number asked"
_JSON = re.compile(r"\{.*\}", re.S)
_WORD = re.compile(r"\w+")


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


# a model quoting without its thinking drops the emphasis and the backticks, and the words are still the section's
_MARKUP = re.compile(r"[*_`]")


def _unmarked(text: str) -> str:
    return _fold(_MARKUP.sub("", unlinked(text)))


# a quote is the section's words in their order, as the strata read an identifier: one key for both
_key = spaceless_key


# a chapter's pairs over its sections by words, highest averages, a stub section left out
def section_pairs(rows: list[dict], chapter_pairs: dict) -> dict[tuple, int]:
    by_chapter: dict = {}
    for row in rows:
        if row["words"] >= MIN_WORDS:
            by_chapter.setdefault(row["chapter"], []).append(row)
    out = {}
    for chapter, members in by_chapter.items():
        given = {section_key(r): 0 for r in members}
        words = {section_key(r): r["words"] for r in members}
        for _ in range(chapter_pairs.get(chapter, 0)):
            best = max(given, key=lambda k: words[k] / (given[k] + 1))
            given[best] += 1
        out |= {k: n for k, n in given.items() if n}
    return out


# the languages a set asks in when its job names none: the stand's own, in its order
def set_languages(given=None) -> tuple[str, ...]:
    return tuple(given or config.settings.evals.question_set.languages)


# the system text the generator reads, the section apart in the user turn
_NAMES = {"en": "English", "ru": "Russian"}


def prompt(template: str, source: str, pairs: int, languages=None) -> str:
    languages = set_languages(languages)
    named = " and ".join(f"{_NAMES[code]} ({code})" for code in languages)
    keys = ", ".join(f'"{code}": "..."' for code in languages)
    return template.format(source=source, pairs=pairs, languages=named, keys=keys)


# a section's pairs over its blocks by length, the longest first to get the remainder; a block may get none
def pairs_by_block(pairs: int, blocks: list[str]) -> list[int]:
    total = sum(len(b) for b in blocks) or 1
    share = [pairs * len(b) // total for b in blocks]
    for i in sorted(range(len(blocks)), key=lambda i: -len(blocks[i]))[: pairs - sum(share)]:
        share[i] += 1
    return share


def asks(pairs: int) -> list[int]:
    return [min(PAIRS_PER_CALL, pairs - done) for done in range(0, pairs, PAIRS_PER_CALL)]


# a later call of one section is told what the earlier ones kept, so it writes about other facts
def user_turn(row: dict, already: list[str] = ()) -> str:
    turn = f"Section path: {row['section']}\n\nSection text:\n{row['text']}"
    if already:
        asked = "\n".join(f"- {q}" for q in already)
        turn += f"\n\nAlready asked about this section, write about other facts:\n{asked}"
    return turn


# a question that is the heading, or carries the whole path, asks what the title says and tests no search
def _echoes_heading(question: str, section: str) -> bool:
    leaf = leaf_of(section)
    return _fold(question).strip(" ?.") == _fold(leaf).strip(" ?.") or _fold(section) in _fold(question)


# the pairs a reply holds that the section can stand behind, and each one refused with its reason
def parse(
    reply: str, row: dict, wanted: int, taken: list[str] = (), languages=None
) -> tuple[list[dict], list[dict]]:
    languages = set_languages(languages)
    found = _JSON.search(reply or "")
    try:
        pairs = json.loads(found.group(0)).get("pairs") if found else None
    except (json.JSONDecodeError, AttributeError):
        pairs = None
    if not isinstance(pairs, list):
        return [], [{"why": "the reply holds no pairs as JSON", "count": wanted}]
    kept, refused, seen = [], [], set()
    quoted = {_key(e) for e in taken}
    text = _key(row["text"])
    for pair in pairs[:wanted]:
        why = _refusal(pair, row, text, seen, languages)
        # without its thinking the generator asks a kept fact again in other words, on the same quote
        if not why and _key(pair["evidence"]) in quoted:
            why = "the evidence repeats another pair of the section"
        if why:
            refused.append({"pair": pair, "why": why})
            continue
        seen |= {text_hash(pair[code]) for code in languages}
        quoted.add(_key(pair["evidence"]))
        kept.append({k: pair[k].strip() for k in (*languages, "answer", "evidence")})
    # the reasons are fixed words, so the set's report can count them
    if len(pairs) < wanted:
        refused.append({"why": "fewer pairs than asked", "count": wanted - len(pairs)})
    if len(pairs) > wanted:
        refused.append({"why": PAST_THE_ASK, "count": len(pairs) - wanted})
    return kept, refused


# a question that leans on the page it was written from: a reader who never saw it cannot tell what is meant
_LEANS_ON_THE_PAGE = re.compile(
    r"\b(the )?(provided|mentioned|shown|given|above|following|presented|listed) "
    r"(code|example|examples|dataset|data|list|table|script|function|model|tools?|snippet|logic|output)\b"
    r"|\bin (this|the) (material|chapter|section|text|book|example|page|guide|tutorial)\b"
    r"|\b(covered|described|discussed) (so far|here|above)\b"
    r"|\b(is|are|was|were) mentioned (as|in)\b"
    r"|\b(listing|figure|table) \d+(\.\d+)*\b|\bлистинг\w* \d|\bрисун\w* \d"
    r"|представленн|упомянут|приведённ|приведенн|показанн"
    r"|в (этом|данном) (материале|разделе|примере|тексте|руководстве)",
    re.IGNORECASE,
)


def _refusal(pair, row: dict, text: str, seen: set, languages: tuple[str, ...]) -> str | None:
    fields = (*languages, "answer", "evidence")
    if not isinstance(pair, dict) or not all(isinstance(pair.get(k), str) and pair[k].strip() for k in fields):
        return f"a field of {', '.join(fields)} is missing or empty"
    # the language is the stand's one rule, not the generator's word
    if any(language_by_alphabet(pair[code]) != code for code in languages):
        return "a question is not in the language of its place"
    if len(_WORD.findall(_unmarked(pair["evidence"]))) > MAX_EVIDENCE_WORDS:
        return "the evidence runs past its word cap"
    if _key(pair["evidence"]) not in text:
        return "the evidence is not the section's own words"
    if any(_echoes_heading(pair[k], row["section"]) for k in languages):
        return "a question repeats the heading"
    if any(_LEANS_ON_THE_PAGE.search(pair[k]) for k in languages):
        return "a question leans on the page it was written from"
    if any(text_hash(pair[k]) in seen for k in languages):
        return "a question repeats another of the section"
    return None


# a kept pair as one question row per language, sharing their pair id and the section's exact gold
def question_rows(pair: dict, row: dict, set_name: str, anchors_of=None, evidence_at=None,
                  languages=None) -> list[dict]:
    languages = set_languages(languages)
    gold = {"file": row["file"], "section": row["section"], "version": section_key(row)[2]}
    pair_id = short_hash("#".join([row["file"], row["section"], *(pair[code] for code in languages)]))
    return [
        {
            "original_text": pair[language],
            "text_hash": text_hash(pair[language]),
            "language": language,
            "set_name": set_name,
            "kind": "in_corpus",
            "status": CANDIDATE,
            "gold": gold,
            "reference_answer": pair["answer"],
            "evidence": pair["evidence"],
            "pair_id": pair_id,
            "anchors": anchors_of(pair[language]) if anchors_of else None,
            "evidence_at": evidence_at(pair) if evidence_at else None,
        }
        for language in languages
    ]


# the anchors of a source's questions read against all its sections at once: the keys are built one time
def anchors_for(sections: list[dict]):
    keys = {section_key(r): spaceless_key(r["text"]) for r in sections}
    every = list(keys.values())
    return lambda row: (lambda question: anchors(question, keys[section_key(row)], every))


# a section as the export keys it: a versioned source holds one path once per version stream
def section_key(row: dict) -> tuple:
    return row["file"], row["section"], row.get("stream")


# the section a question's gold names, spelled as `section_key` spells an exported row
def gold_key(gold: dict | None) -> tuple:
    gold = gold or {}
    return gold.get("file"), gold.get("section"), gold.get("version")


# a block's fingerprint: the generator stamps it, and a later reader opens the block only while it still matches
def block_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# where a kept pair's evidence sits in the block the generator read, so a judge opens that place and not a look-alike
def placed_in(block: int, text: str):
    from evals.pair_judge import locate

    sha = block_sha(text)
    return lambda pair: {"block": block, "block_sha": sha, "char": locate(text, pair["evidence"])}
