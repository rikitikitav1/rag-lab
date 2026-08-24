import sys
import time
from operator import itemgetter

import config
import job_queue
import llm
import logging_setup
import rerank
from models.eval import Question
from models.registry import Pipeline
from orm.sync_db import Session
from sqlalchemy import select
from use_cases import agent, chat

import db

log = logging_setup.get_logger(__name__)


def _target_texts(set_name: str | None, question_ids: list[int] | None) -> list[str]:
    with Session() as session:
        stmt = select(Question.original_text)
        if question_ids:
            stmt = stmt.where(Question.id.in_(question_ids))
        elif set_name:
            stmt = stmt.where(Question.set_name == set_name)
        return list(session.scalars(stmt))


def _answer_one(
    text: str,
    run_name: str,
    use_rerank: bool | None,
    pipeline: Pipeline,
    language: str | None,
    k: int | None,
    max_hops: int | None,
    model: str | None,
) -> None:
    if pipeline == Pipeline.agent:
        agent.run(
            text,
            run_name=run_name,
            language=language,
            k=k,
            max_hops=max_hops,
            use_rerank=use_rerank,
            model=model,
        )
    elif pipeline == Pipeline.single_shot:
        chat.answer(
            text,
            add_context=True,
            run_name=run_name,
            use_rerank=use_rerank,
            language=language,
            k=k,
            model=model,
        )
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")


def _run_sequential(
    texts: list[str],
    run_name: str,
    use_rerank: bool | None,
    pipeline: Pipeline,
    language: str | None,
    k: int | None,
    max_hops: int | None,
    model: str | None,
    job_id: int | None,
) -> tuple[int, bool]:
    answered = 0
    for text in texts:
        if job_id is not None and job_queue.is_cancelled(job_id):
            return answered, True
        try:
            _answer_one(text, run_name, use_rerank, pipeline, language, k, max_hops, model)
            answered += 1
        except Exception as e:
            log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
    return answered, False


def _phase_retrieve(texts: list[str], k: int, use_rerank: bool) -> list:
    limit = config.settings.rerank.candidates if use_rerank else k
    vectors = llm.request_embeddings_batch(texts)
    return [
        (text, db.hybrid_search(text, vector, None, limit=limit))
        for text, vector in zip(texts, vectors, strict=True)
    ]


def _phase_rerank(retrieved: list, k: int) -> list:
    scores = rerank.score_pairs([(text, row[0]) for text, rows in retrieved for row in rows])

    ranked, offset = [], 0
    for text, rows in retrieved:
        window = scores[offset : offset + len(rows)]
        offset += len(rows)
        best = sorted(zip(rows, window, strict=True), key=itemgetter(1), reverse=True)
        ranked.append((text, [row for row, _ in best[:k]]))
    return ranked


def _phase_generate(
    retrieved: list,
    run_name: str,
    use_rerank: bool,
    language: str | None,
    k: int,
    model: str | None,
    job_id: int | None,
) -> tuple[int, bool]:
    answered = 0
    for text, rows in retrieved:
        if job_id is not None and job_queue.is_cancelled(job_id):
            return answered, True
        try:
            chat.answer_from_rows(
                text,
                rows,
                add_context=True,
                run_name=run_name,
                use_rerank=use_rerank,
                language=language,
                k=k,
                model=model,
                phased=True,
            )
            answered += 1
        except Exception as e:
            log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
    return answered, False


def run_phased(
    run_name: str,
    texts: list[str],
    use_rerank: bool,
    language: str | None,
    k: int | None,
    model: str | None,
    job_id: int | None,
) -> tuple[int, bool]:
    k = k or config.settings.retrieval.results_limit

    started = time.perf_counter()
    retrieved = _phase_retrieve(texts, k, use_rerank)
    log.info("eval_run.phase", name="retrieve", n=len(retrieved),
             elapsed=round(time.perf_counter() - started, 1))

    if use_rerank:
        llm.unload("embedding")
        llm.unload("generation")
        started = time.perf_counter()
        retrieved = _phase_rerank(retrieved, k)
        log.info("eval_run.phase", name="rerank", n=len(retrieved),
                 elapsed=round(time.perf_counter() - started, 1))
        rerank.unload()

    started = time.perf_counter()
    answered, cancelled = _phase_generate(
        retrieved, run_name, use_rerank, language, k, model, job_id
    )
    log.info("eval_run.phase", name="generate", n=answered,
             elapsed=round(time.perf_counter() - started, 1))
    return answered, cancelled


def run(
    run_name: str,
    set_name: str | None = None,
    question_ids: list[int] | None = None,
    use_rerank: bool | None = None,
    pipeline: str = "single_shot",
    language: str | None = None,
    k: int | None = None,
    max_hops: int | None = None,
    model: str | None = None,
    job_id: int | None = None,
    phased: bool | None = None,
) -> int:
    pipeline = Pipeline(pipeline)
    texts = _target_texts(set_name, question_ids)
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    if phased is None:
        phased = pipeline == Pipeline.single_shot

    if phased and pipeline == Pipeline.single_shot:
        answered, cancelled = run_phased(
            run_name, texts, use_rerank, language, k, model, job_id
        )
    else:
        answered, cancelled = _run_sequential(
            texts, run_name, use_rerank, pipeline, language, k, max_hops, model, job_id
        )
    if not cancelled:
        job_queue.enqueue("judge_answers", {"run_name": run_name})
    log.info(
        "eval_run.answered",
        run_name=run_name,
        answered=answered,
        total=len(texts),
        cancelled=cancelled,
    )
    return answered


if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else "curated"
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"{set_name}_{int(time.time())}"
    n = run(run_name, set_name=set_name)
    print(f"run: {run_name} | set: {set_name} | answered: {n} | judging enqueued")
