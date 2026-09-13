"""The standard's judged axes as guests on our own row.

They ride the same judging pass as our three: same client, same seed, same row. What they are not
is a second arm, and nothing here may decide whether the row counts as judged by us.
"""

import asyncio
import math
import time
from dataclasses import dataclass

# one prefix tells a guest key from ours in `metrics`, in a copy and in a report
PREFIX = "ragas_"


@dataclass(frozen=True)
class Guest:
    metric: str
    # what the row must carry, named once: the sweep's query and the row's check read this
    needs: tuple[str, ...]
    # what one row costs, read by nobody yet: a smoke that prices a set is still to be written
    calls_per_row: str
    # the only one that measures with vectors, so it borrows our embedder as well as our judge
    embeds: bool = False
    # what the metric must be told about us, rather than discover and silently work around
    options: tuple[tuple[str, object], ...] = ()


# every axis is handed `user_input`, so `question_text` is material to all three
AXES = {
    f"{PREFIX}faithfulness": Guest(
        "Faithfulness", ("question_text", "answer", "contexts"), "2"
    ),
    f"{PREFIX}context_precision": Guest(
        "LLMContextPrecisionWithReference",
        ("question_text", "contexts", "reference"), "one per context",
    ),
    f"{PREFIX}context_recall": Guest(
        "LLMContextRecall", ("question_text", "contexts", "reference"), "1"
    ),
    # strictness 1, not the default 3: our judge is seeded, and measured at one chat, two embeddings
    f"{PREFIX}answer_relevancy": Guest(
        "ResponseRelevancy", ("question_text", "answer"), "1 plus 2 embeddings",
        embeds=True, options=(("strictness", 1),),
    ),
}

NAMES = tuple(AXES)

HAS = {
    "question_text": lambda ql: bool(ql.question_text),
    "answer": lambda ql: bool(ql.answer),
    "contexts": lambda ql: bool(ql.contexts),
    "reference": lambda ql: bool(ql.question and ql.question.reference_answer),
}


def material(axis: str, ql) -> bool:
    return all(HAS[name](ql) for name in AXES[axis].needs)


# an axis that abstained holds no score and is still answered: `nan` is the standard's verdict
def owed(ql, metrics: dict) -> tuple[str, ...]:
    return tuple(
        axis
        for axis in AXES
        if "abstained" not in (metrics.get(axis) or {}) and material(axis, ql)
    )


# the four fields a guest reads, off the session: scoring held a connection idle for minutes
def carried(ql):
    from types import SimpleNamespace

    return SimpleNamespace(
        question_text=ql.question_text,
        answer=ql.answer,
        contexts=list(ql.contexts or []),
        question=SimpleNamespace(
            reference_answer=ql.question.reference_answer if ql.question else None
        ),
    )


def _sample(ql):
    from ragas.dataset_schema import SingleTurnSample

    return SingleTurnSample(
        user_input=ql.question_text or "",
        response=ql.answer or "",
        retrieved_contexts=list(ql.contexts or []),
        reference=ql.question.reference_answer if ql.question else None,
    )


# how the guest's prompt reaches the model; the second is the ruler every older guest number was taken with
MESSAGE_FORMS = ("user_only", "empty_system")

_METRICS: dict = {}


def _metric(axis: str, messages: str = MESSAGE_FORMS[0]):
    if (axis, messages) not in _METRICS:
        import ragas.metrics as guest_metrics
        from evals.guest_llm import OurClient, OurEmbeddings

        guest = AXES[axis]
        extra = {"embeddings": OurEmbeddings()} if guest.embeds else {}
        _METRICS[(axis, messages)] = getattr(guest_metrics, guest.metric)(
            llm=OurClient(messages=messages), **extra, **dict(guest.options)
        )
    return _METRICS[(axis, messages)]


# nan is how the standard abstains, and JSONB has no place to put it
def _finite(score) -> float | None:
    value = float(score)
    return None if math.isnan(value) else round(value, 4)


def score(axis: str, ql, messages: str = MESSAGE_FORMS[0]) -> dict:
    import llm
    from evals.guest_llm import stamp

    start = time.perf_counter()
    # ragas asks the model several times for one row and keeps only the text; this scope keeps the sum
    with llm.accounting() as row:
        value = asyncio.run(_metric(axis, messages).single_turn_ascore(_sample(ql)))
    finite = _finite(value)
    return {
        "score": finite,
        "abstained": finite is None,
        "elapsed": round(time.perf_counter() - start, 3),
        "tokens": row.record(),
        **stamp(messages),
    }
