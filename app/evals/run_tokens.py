import token_fields
from evals.guest_axes import PREFIX
from job_queue import merged_tokens
from models import Job, JobStatus
from models.eval import QuestionLog
from orm.sync_db import Session
from sqlalchemy import select
from use_cases.rejudge import AXES

SCHEMA = 2
READS = (
    "spent is the sum over the run's jobs per role, engine and model, retries and failed attempts included, "
    "and is what a broker's quota sees; per_question is the price of one row per role, read from the rows, with "
    "the rows that carry no count named rather than read as zero; neither is derived from the other; "
    "spent_by_engine sums spent per engine; jobs counts the jobs read and those from before the count; "
    "debited is each broker's balance before less after every job, in the broker's unit: the balance is the key's, "
    "so a chat, another session or the worker's second lane on the same key lands in it too"
)
FINISHED = (JobStatus.done, JobStatus.error, JobStatus.cancelled)


def of(run_name: str) -> dict:
    with Session() as session:
        jobs = session.execute(
            select(Job.tokens, Job.status, Job.balances).where(Job.options["run_name"].astext == run_name)
        ).all()
        rows = session.execute(
            select(QuestionLog.prompt_tokens, QuestionLog.completion_tokens, QuestionLog.metrics)
            .where(QuestionLog.run_name == run_name)
        ).all()
    return summarize(
        [(tokens, status) for tokens, status, _ in jobs], [tuple(r) for r in rows], [b for _, _, b in jobs]
    )


def summarize(
    jobs: list[tuple[dict | None, str]], rows: list[tuple[int | None, int | None, dict | None]],
    balances: list[dict | None] = (),
) -> dict:
    spent = None
    for tokens, _ in jobs:
        if tokens is not None:
            spent = merged_tokens(spent, tokens)
    by_engine: dict[str, dict] = {}
    for entries in (spent or {}).values():
        for entry in entries:
            held = by_engine.setdefault(entry["engine"], {"prompt": 0, "completion": 0, "calls": 0})
            for key in held:
                held[key] += entry.get(key, 0)
    return {
        "schema": SCHEMA,
        "spent": spent or None,
        "spent_by_engine": by_engine or None,
        "debited": _debited(balances),
        "jobs": {
            "counted": sum(1 for tokens, _ in jobs if tokens is not None),
            # null is a job from before the count; one that spent nothing wrote {}
            "before_the_count": sum(1 for tokens, status in jobs if tokens is None and status in FINISHED),
            "not_finished": sum(1 for tokens, status in jobs if tokens is None and status not in FINISHED),
        },
        "per_question": {
            "generation": _price([(p, c) for p, c, _ in rows]),
            "judging": _price([_judged(metrics) for _, _, metrics in rows if _judged(metrics) is not _UNJUDGED]),
            "ragas": _price([_guested(metrics) for _, _, metrics in rows if _guested(metrics) is not _UNJUDGED]),
            "ragas_embedding": _price([
                _guested(metrics, embedder=True) for _, _, metrics in rows if _guested(metrics) is not _UNJUDGED
            ]),
        },
        "reads": READS,
    }


_UNJUDGED = object()


# a job whose balance was not read on both sides is counted apart, never as a zero debit
def _debited(balances: list[dict | None]) -> dict | None:
    out: dict[str, dict] = {}
    for job in balances:
        for engine, seen in (job or {}).items():
            held = out.setdefault(engine, {"debited": 0.0, "unit": seen.get("unit"), "jobs_read": 0, "jobs_unread": 0})
            try:
                held["debited"] += float(seen["before"]) - float(seen["after"])
                held["jobs_read"] += 1
            except (KeyError, TypeError, ValueError):
                held["jobs_unread"] += 1
    for held in out.values():
        held["debited"] = round(held["debited"], 8)
    return out or None


# a verdict from before the output was counted carries the input alone, and the row is a named gap
def _judged(metrics: dict | None):
    stamps = [(metrics or {}).get(axis) for axis in AXES]
    stamps = [s for s in stamps if isinstance(s, dict) and token_fields.JUDGE_PROMPT in s]
    if not stamps:
        return _UNJUDGED
    if any(s.get(token_fields.JUDGE_PROMPT) is None or s.get(token_fields.JUDGE_COMPLETION) is None for s in stamps):
        return None, None
    return sum(s[token_fields.JUDGE_PROMPT] for s in stamps), sum(s[token_fields.JUDGE_COMPLETION] for s in stamps)


_EMBEDDERS = ("ragas_embedding", "embedding")


def _guested(metrics: dict | None, embedder: bool = False):
    stamps = [v for k, v in (metrics or {}).items() if k.startswith(PREFIX) and isinstance(v, dict) and "score" in v]
    if not stamps:
        return _UNJUDGED
    if any("tokens" not in s for s in stamps):
        return None, None
    # the guest's embedder is its own role, and folded in it made this disagree with `spent`
    entries = [e for s in stamps for name, got in (s["tokens"] or {}).items() if (name in _EMBEDDERS) == embedder for e in got]
    return sum(e.get("prompt", 0) for e in entries), sum(e.get("completion", 0) for e in entries)


def _price(pairs: list[tuple[int | None, int | None]]) -> dict:
    counted = [(p, c) for p, c in pairs if p is not None and c is not None]
    n = len(counted)
    return {
        "rows_counted": n,
        "rows_missing": len(pairs) - n,
        "prompt": sum(p for p, _ in counted),
        "completion": sum(c for _, c in counted),
        "prompt_per_row": round(sum(p for p, _ in counted) / n, 1) if n else None,
        "completion_per_row": round(sum(c for _, c in counted) / n, 1) if n else None,
    }
