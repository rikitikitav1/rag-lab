"""How often the query-language rules disagree: the numbers under `query_lang` in the config.

Compares every rule against the configured default. Bucketed by question length rather than by set,
because the set grouping was a judgement and the length is what decides whether langdetect copes.
"""


import config
from orm.sync_db import engine
from sqlalchemy import text

import db

BUCKETS = ((0, 40), (40, 80), (80, 10_000))
RULES = ("langdetect", "cyrillic_ratio", "function_words")
QUESTIONS = "SELECT set_name, original_text FROM questions WHERE original_text <> ''"


def bucket(length: int) -> tuple:
    return next(b for b in BUCKETS if b[0] <= length < b[1])


with engine.connect() as conn:
    by_set: dict[str, list[int]] = {}
    by_length: dict[tuple, list[int]] = {b: [0, 0] for b in BUCKETS}
    lengths: dict[bool, list[int]] = {True: [], False: []}
    against_default: dict[str, list[int]] = {rule: [0, 0] for rule in RULES}

    for set_name, question in conn.execute(text(QUESTIONS)):
        default = db.detect_language(question)
        verdicts = {rule: db.detect_language(question, rule) for rule in RULES}
        for rule, verdict in verdicts.items():
            against_default[rule][0] += verdict != default
            against_default[rule][1] += 1
        differs = verdicts["langdetect"] != verdicts["cyrillic_ratio"]
        seen = by_set.setdefault(set_name or "(none)", [0, 0])
        seen[0] += differs
        seen[1] += 1
        cell = by_length[bucket(len(question))]
        cell[0] += differs
        cell[1] += 1
        lengths[bool(differs)].append(len(question))

    print(f"each rule against the configured default ({config.settings.retrieval.query_lang})")
    for rule, (differ, total) in against_default.items():
        print(f"  {rule:16} {differ:5} of {total:5} ({differ / total:.1%})")

    print("\nlangdetect against the alphabet rule, by set (20 questions or more)")
    for name, (differ, total) in sorted(by_set.items(), key=lambda kv: -kv[1][1]):
        if total >= 20:
            print(f"  {name:24} {differ:5} of {total:5} ({differ / total:.1%})")

    print("\nby question length, which is what actually decides it")
    for (low, high), (differ, total) in by_length.items():
        span = f"{low}-{high} chars" if high < 10_000 else f"{low}+ chars"
        share = f"{differ / total:.1%}" if total else "n/a"
        print(f"  {span:16} {differ:5} of {total:5} ({share})")

    # the buckets above are a choice; these lines are not
    longest = max(lengths[True], default=0)
    above = sum(1 for n in lengths[True] + lengths[False] if n > longest)
    print(
        f"\nlongest question the rules disagree on: {longest} chars"
        f"\nshortest they agree on:                 {min(lengths[False], default=0)} chars"
        f"\nquestions longer than that, all agreeing: {above}"
    )
