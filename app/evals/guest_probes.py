"""What the guest faithfulness axis reads, probed by holding one thing still and moving the other.

Three arms, all on rows of a real run. `paraphrase` keeps the meaning and destroys the overlap;
`negated` keeps the overlap and breaks one claim; `code` strips the fenced code and leaves the prose.
Every report carries the instrument's stamp and a bootstrap interval, because a point estimate on
ten rows is what let a difference be called refuted once already.
"""

import statistics

import llm
from errors import StandFault
from evals.guest_llm import stamp
from redaction import redact
from use_cases.ingest_quality import FENCE

# 1 the first shape of this report: three arms, a stamp and an interval each
SCHEMA = 1

RESTATE = (
    "Restate the passage in your own words. Keep every claim, add nothing, drop nothing. "
    "Answer with the restatement only, no preamble."
)
NEGATE = (
    "Repeat the passage word for word, changing exactly one factual claim into its opposite. "
    "Keep every other word identical. Answer with the altered passage only, no preamble."
)


def sentence_of(contexts: list[str]) -> str | None:
    for line in "\n".join(contexts or []).splitlines():
        if len(line) > 90 and not line.startswith(("#", "```", "//", "  ", "|")):
            return line.strip()
    return None


def without_code(text: str) -> str:
    return "\n\n".join(p for p in FENCE.sub("", text or "").split("\n\n") if p.strip()).strip()


def arms_for(arm: str, ql) -> list[tuple[str, str]]:
    if arm == "code":
        return [("code_kept", ql.answer), ("code_stripped", without_code(ql.answer))]
    source = sentence_of(ql.contexts)
    if not source:
        return []
    if arm == "negated":
        return [("copy", source), ("negated", llm.ask(NEGATE, source, role="generation").text or "")]
    return [
        ("copy", source),
        ("paraphrase_en", llm.ask(RESTATE, source, role="generation").text or ""),
    ]


def score(metric, ql, answer) -> tuple[float | None, int, str | None]:
    import asyncio

    row = {"user_input": ql.question_text or "", "response": answer,
           "retrieved_contexts": list(ql.contexts or [])}

    async def run():
        statements = (await metric._create_statements(row, None)).statements
        if not statements:
            return None, 0
        verdicts = await metric._create_verdicts(row, statements, None)
        return float(metric._compute_score(verdicts)), len(statements)

    try:
        value, n = asyncio.run(run())
        return value, n, None
    except StandFault:
        raise
    except Exception as e:
        return None, 0, f"{type(e).__name__}: {redact(str(e))}"[:120]


# the stand already has one bootstrap, and it holds its own generator instead of seeding everyone's
def interval(sample: list[float]) -> list[float] | None:
    from use_cases.retrieval_compare import bootstrap_ci

    if len(sample) < 2:
        return None
    return [round(v, 4) for v in bootstrap_ci(list(sample))]


def report(arm: str, done: list[dict]) -> dict:
    by = {}
    for r in done:
        by.setdefault(r["arm"], []).append(r)
    pairs = {}
    for r in done:
        pairs.setdefault(r["row"], {})[r["arm"]] = r["score"]
    names = sorted(by)
    left, right = (names[0], names[1]) if len(names) > 1 else (names[0], names[0])
    deltas = [p[left] - p[right] for p in pairs.values()
              if p.get(left) is not None and p.get(right) is not None]
    return {
        "schema": SCHEMA,
        "arm": arm,
        "instrument": stamp(),
        "n_rows": len({r["row"] for r in done}),
        "failures": sum(1 for r in done if r["error"]),
        "means": {
            name: {
                "score": round(statistics.fmean(
                    [r["score"] for r in rows if r["score"] is not None]), 4),
                "overlap": round(statistics.fmean(r["overlap"] for r in rows), 4),
                "n": len(rows),
            }
            for name, rows in by.items() if any(r["score"] is not None for r in rows)
        },
        "paired": {
            "of": f"{left} minus {right}",
            "n": len(deltas),
            "mean": round(statistics.fmean(deltas), 4) if deltas else None,
            "ci95": interval(deltas),
            "moved": sum(1 for d in deltas if d != 0),
        },
        "rows": done,
    }
