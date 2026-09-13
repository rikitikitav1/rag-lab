from evals.guest_axes import PREFIX
from job_queue import merged_tokens
from models import Job, JobStatus
from models.eval import QuestionLog
from orm.sync_db import Session
from sqlalchemy import select
from use_cases.rejudge import AXES

SCHEMA = 1
READS = (
    "spent is what the run's jobs paid, retries and failed attempts included, and is what a broker's "
    "quota sees; per_question is the price of one row, read from the rows; neither is derived from the other"
)
FINISHED = (JobStatus.done, JobStatus.error, JobStatus.cancelled)


def of(run_name: str) -> dict:
    with Session() as session:
        jobs = session.execute(
            select(Job.tokens, Job.status).where(Job.options["run_name"].astext == run_name)
        ).all()
        rows = session.execute(
            select(QuestionLog.prompt_tokens, QuestionLog.completion_tokens, QuestionLog.metrics)
            .where(QuestionLog.run_name == run_name)
        ).all()
    return summarize([(tokens, status) for tokens, status in jobs], [tuple(r) for r in rows])


def summarize(jobs: list[tuple[dict | None, str]], rows: list[tuple[int | None, int | None, dict | None]]) -> dict:
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
        },
        "reads": READS,
    }


_UNJUDGED = object()


# a verdict from before the output was counted carries the input alone, and the row is a named gap
def _judged(metrics: dict | None):
    stamps = [(metrics or {}).get(axis) for axis in AXES]
    stamps = [s for s in stamps if isinstance(s, dict) and "judge_prompt_tokens" in s]
    if not stamps:
        return _UNJUDGED
    if any(s.get("judge_prompt_tokens") is None or s.get("judge_completion_tokens") is None for s in stamps):
        return None, None
    return sum(s["judge_prompt_tokens"] for s in stamps), sum(s["judge_completion_tokens"] for s in stamps)


def _guested(metrics: dict | None):
    stamps = [v for k, v in (metrics or {}).items() if k.startswith(PREFIX) and isinstance(v, dict) and "score" in v]
    if not stamps:
        return _UNJUDGED
    if any("tokens" not in s for s in stamps):
        return None, None
    entries = [e for s in stamps for role in (s["tokens"] or {}).values() for e in role]
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
