"""Does our own faithfulness prompt lose a point on Russian, or does the loss live upstream.

The guest judge does not penalise a Russian restatement of the same context (1.0 on nine of nine).
Three links stand between that and the drop measured in 3.2: retrieval differs on a Russian query,
the generator writes a different answer, and only then does the judge read it. This holds the first
two still and moves only the language of the answer.
"""

import statistics

import llm
from evals.guest_probes import RESTATE, sentence_of
from use_cases.judge import faithful_verdict

# 2 the restated subject is the row's own answer; 3 the rows may be named instead of counted
SCHEMA = 3

# history of this instrument on the 3.2 sets: 96.4% and 95.6% at least 7, and it drifts on a reload
CONTROL_FLOOR = 0.90


# the answer, not a line of the context: judging a restated context line is judging a non-answer
def pairs_for(ql) -> list[tuple[str, str]]:
    source = (ql.answer or "").strip() or sentence_of(ql.contexts)
    if not source:
        return []
    english = llm.ask(RESTATE, source, role="generation").text or ""
    russian = llm.ask(RESTATE + " Answer in Russian.", source, role="generation").text or ""
    return [("en", english), ("ru", russian)]


# the reason as well as the score: a fifth of pass 1 scored zero and nothing said which rule fired
def score(ql, answer: str) -> tuple[int | None, str]:
    context = "\n\n".join(ql.contexts or [])
    verdict = faithful_verdict(ql.question_text or "", answer, context)
    return (verdict.score, verdict.reason) if verdict else (None, "")


# the loop lived in the script, so a job would have been a second copy of it
def measure(run_name: str, rows: int, note=None, stop=None, log_ids=None) -> dict:
    from evals.loaders import load_logs
    from evals.pools import kind

    # the corpus pool alone: the control threshold comes from it, and a mixed pool fails it rightly
    pool = [
        q for q in load_logs(run_name)
        if q.answered and q.contexts and q.answer and kind(q) == "in_corpus"
    ]
    # a declared group is named, not counted off the top: a cut is not the first rows of a run
    pool = [q for q in pool if q.id in set(log_ids)] if log_ids else pool[:rows]
    scored, originals = [], []
    for ql in pool:
        # two model calls a row: a cancelled probe that runs to the end is not cancelled
        if stop and stop():
            break
        # free, and it is the control: our judge already scored this row's own answer
        originals.append(ql.faithfulness)
        for lang, answer in pairs_for(ql):
            got, why = score(ql, answer)
            scored.append({"row": ql.id, "lang": lang, "score": got,
                           "chars": len(answer), "reason": why})
            if note:
                note(f"{lang} {ql.id}: {scored[-1]['score']}")
    population = "the named rows" if log_ids else f"the first {rows} of the corpus pool"
    return (report(scored) | control(originals)
            | {"run_name": run_name, "population": population, "n_asked": len(pool),
               "rows": scored})


# pass 1 scored a grounded restatement zero in a fifth of pairs and nothing said the regime was off
def control(originals: list) -> dict:
    from evals.stats import score_of

    got = [v for v in (score_of(x) for x in originals) if v is not None]
    share = round(sum(1 for v in got if v >= 7) / len(got), 3) if got else None
    return {
        "control": {
            "of": "our judge on the row's own answer, at least 7",
            "n": len(got),
            "share": share,
            "threshold": CONTROL_FLOOR,
            "in_regime": share is not None and share >= CONTROL_FLOOR,
        }
    }


def report(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["lang"], []).append(r["score"])
    paired = {}
    for r in rows:
        paired.setdefault(r["row"], {})[r["lang"]] = r["score"]
    deltas = [p["en"] - p["ru"] for p in paired.values()
              if p.get("en") is not None and p.get("ru") is not None]
    from use_cases.retrieval_compare import bootstrap_ci

    return {
        "schema": SCHEMA,
        "n_rows": len(paired),
        "means": {lang: round(statistics.fmean([s for s in v if s is not None]), 3)
                  for lang, v in by.items() if any(s is not None for s in v)},
        "paired": {
            "of": "en minus ru, our own judge",
            "n": len(deltas),
            "mean": round(statistics.fmean(deltas), 3) if deltas else None,
            "ci95": [round(v, 3) for v in bootstrap_ci(deltas)] if len(deltas) > 1 else None,
            "moved": sum(1 for d in deltas if d != 0),
        },
    }
