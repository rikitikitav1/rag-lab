"""Does our own faithfulness prompt lose a point on Russian, or does the loss live upstream."""

import json
import statistics

import llm
from evals.guest_probes import RESTATE, sentence_of
from evals.measurements import FOLDER
from evals.stats import bootstrap_ci
from use_cases.judge import faithful_verdict

# the report's shape, raised with every key it gains
SCHEMA = 8

# a floor that catches only the gross: the judge's history on these rows sat at 96.4% and 95.6% at least 7
CONTROL_FLOOR = 0.90

# fixed English rows: the regime is a property of the instrument, so it is not read on the run it measures
PANEL = FOLDER / "judge_language_panel_ids.txt"

# the panel's first reading under today's ruler, rewritten by hand when the ruler changes, never by the probe
REFERENCE = FOLDER / "judge_language_panel_reference.json"

# a regime holds while at most this share of the panel crosses 7 against the reference, on either part
MOVES_ALLOWED = 0.10


def panel_ids() -> list[int]:
    lines = PANEL.read_text(encoding="utf-8").splitlines()
    return [int(x) for line in lines for x in line.split("#")[0].split()]


def panel_reference() -> dict | None:
    return json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() else None


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
def measure(run_name: str, rows: int, note=None, stop=None, log_ids=None, stamp=None) -> dict:
    from evals.loaders import load_logs
    from evals.pools import kind

    # the corpus pool alone: the control threshold comes from it, and a mixed pool fails it rightly
    pool = [
        q for q in load_logs(run_name)
        if q.answered and q.contexts and q.answer and kind(q) == "in_corpus"
    ]
    # a declared group is named, not counted off the top: a cut is not the first rows of a run
    pool = [q for q in pool if q.id in set(log_ids)] if log_ids else pool[:rows]
    panel_rows = judge_panel(stop)
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
    return (report(scored) | regime(panel_rows, panel_reference()) | run_answers(originals)
            | {"run_name": run_name, "population": population, "n_asked": len(pool),
               # a control out of regime is unreadable without knowing what judged it
               "instrument": stamp or {}, "rows": scored, "panel_rows": panel_rows})


# the panel's own answer gives the judge's regime, its English restatement the restating path's
def judge_panel(stop=None) -> list[dict]:
    from evals.loaders import load_logs

    rows = []
    for ql in load_logs(ids=panel_ids()):
        if stop and stop():
            break
        # one row that fails is one lost verdict, read as moved, not a panel paid for again on retry
        try:
            english = llm.ask(RESTATE, ql.answer, role="generation").text or ""
        except RuntimeError as e:
            rows += [
                {"row": ql.id, "part": part, "score": None, "reason": f"failed: {e}"}
                for part in ("own", "restated")
            ]
            continue
        for part, answer in (("own", ql.answer), ("restated", english)):
            try:
                got, why = score(ql, answer)
            except RuntimeError as e:
                got, why = None, f"failed: {e}"
            rows.append({"row": ql.id, "part": part, "score": got, "reason": why})
    return rows


def _at_least_7(scores: list) -> dict:
    from evals.stats import score_of

    got = [v for v in (score_of(x) for x in scores) if v is not None]
    return {"n": len(got), "share": round(sum(1 for v in got if v >= 7) / len(got), 3) if got else None}


def _crosses_7(score) -> bool | None:
    from evals.stats import score_of

    got = score_of(score)
    return None if got is None else got >= 7


# a fixed share measured the panel's answers as much as the judge; against a reference it reads drift
def regime(panel_rows: list[dict], reference: dict | None) -> dict:
    parts = {p: _at_least_7([r["score"] for r in panel_rows if r["part"] == p]) for p in ("own", "restated")}
    # a floor read on the verdicts that came back passed on two thirds of the panel
    whole = all(part["n"] >= len(panel_ids()) for part in parts.values())
    control = {
        "of": "our judge at least 7 on the fixed panel: its own answers and their English restatement",
        "panel": PANEL.name,
        "whole_panel": whole,
        **parts,
        "floor": CONTROL_FLOOR,
        "above_floor": (
            all(p["share"] is not None and p["share"] >= CONTROL_FLOOR for p in parts.values())
            if whole else None
        ),
    }
    if reference is None:
        return {"control": control | {
            "reference": None, "moved_vs_reference": None, "in_regime": None,
            "why": "no reference reading yet: the first reading under a ruler is kept by hand, never by the probe",
        }}
    # a verdict lost, a row the reference never read, or one this reading never reached counts as moved
    was = {(r["row"], r["part"]): _crosses_7(r["score"]) for r in reference["rows"]}
    seen = {(r["row"], r["part"]) for r in panel_rows}
    moved = {p: sum(1 for r in panel_rows if r["part"] == p
                    and was.get((r["row"], p), "unread") != _crosses_7(r["score"]))
                + sum(1 for row, part in was if part == p and (row, part) not in seen)
             for p in ("own", "restated")}
    allowed = int(len(panel_ids()) * MOVES_ALLOWED)
    return {"control": control | {
        "reference": {"file": REFERENCE.name, "taken": reference.get("taken")},
        "moved_vs_reference": moved,
        "moves_allowed": allowed,
        "in_regime": all(n <= allowed for n in moved.values()),
    }}


# a run's own rows mix the judge's state with the run's quality, so they are the run's, not a regime
def run_answers(originals: list) -> dict:
    return {"run_answers": {"of": "our judge on the run's own answers, at least 7", **_at_least_7(originals)}}


def report(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["lang"], []).append(r["score"])
    paired = {}
    for r in rows:
        paired.setdefault(r["row"], {})[r["lang"]] = r["score"]
    deltas = [p["en"] - p["ru"] for p in paired.values()
              if p.get("en") is not None and p.get("ru") is not None]
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
