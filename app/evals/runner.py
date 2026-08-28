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
    fallback_policy: str | None,
    gate_signal: str | None,
    weak_distance: float | None,
    topic_threshold: float | None,
    orchestrator: str | None,
    variant: str,
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
            fallback_policy=fallback_policy,
            gate_signal=gate_signal,
            weak_distance=weak_distance,
            topic_threshold=topic_threshold,
            orchestrator=orchestrator,
            variant=variant,
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
            variant=variant,
        )
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")


def _refuse_a_cpu_run(allow_cpu: bool) -> None:
    # ollama drops the card and keeps answering: same numbers, four times the hours, and
    # nothing in the run says so. Both paths ask after the first answer, when the
    # generator is loaded and a spill is finally visible
    llm.warn_if_models_do_not_fit()
    off_card = llm.models_off_the_card()
    if off_card and not allow_cpu:
        raise RuntimeError(
            f"models are not on the GPU: {', '.join(off_card)}."
            " Pass allow_cpu if this run is meant to measure the CPU"
        )
    if off_card:
        log.warning("eval_run.cpu_allowed", models=off_card)


def _run_sequential(
    texts: list[str],
    run_name: str,
    use_rerank: bool | None,
    pipeline: Pipeline,
    language: str | None,
    k: int | None,
    max_hops: int | None,
    model: str | None,
    fallback_policy: str | None,
    gate_signal: str | None,
    weak_distance: float | None,
    topic_threshold: float | None,
    orchestrator: str | None,
    job_id: int | None,
    allow_cpu: bool,
    variant: str,
) -> tuple[int, bool]:
    answered = 0
    # the agent path is this one, and a sweep over `model` runs several of them back to
    # back: without giving the card back, arm two loads its generator beside arm one's
    try:
        for text in texts:
            if job_id is not None and job_queue.is_cancelled(job_id):
                return answered, True
            # after the first answer the generator is loaded, so a spill is finally visible
            if answered == 1:
                _refuse_a_cpu_run(allow_cpu)
            try:
                _answer_one(
                    text, run_name, use_rerank, pipeline, language, k, max_hops, model,
                    fallback_policy, gate_signal, weak_distance, topic_threshold,
                    orchestrator, variant,
                )
                answered += 1
            except Exception as e:
                log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
        return answered, False
    finally:
        _free_the_card(model)


def _embed_in_batches(texts: list[str]) -> list:
    size = config.settings.ingestion.batch_size
    vectors: list = []
    for start in range(0, len(texts), size):
        chunk = texts[start : start + size]
        try:
            vectors.extend(llm.request_embeddings_batch(chunk))
        except Exception as e:
            log.error("eval_run.embed_failed", start=start, n=len(chunk), error=str(e))
            vectors.extend([None] * len(chunk))
    return vectors


def _phase_retrieve(texts: list[str], k: int, use_rerank: bool, variant: str) -> tuple[list, int]:
    from use_cases import search_depth

    limit = config.settings.rerank.candidates if use_rerank else k
    # resolved once for the phase and carried into every snapshot: a phased run used to
    # record `ef_search: null`, and phased is the default for single-shot, so the depth of
    # the runs this branch produced was not in their own records
    depth = search_depth.resolve(variant)
    retrieved = []
    for text, vector in zip(texts, _embed_in_batches(texts), strict=True):
        if vector is None:
            continue
        try:
            retrieved.append(
                (
                    text,
                    db.hybrid_search(text, vector, None, limit=limit, variant=variant,
                                     ef_search=depth),
                    None,
                )
            )
        except Exception as e:
            log.error("eval_run.search_failed", run_text=text[:80], error=str(e))
    return retrieved, depth


def _phase_rerank(retrieved: list, k: int) -> list:
    scores = rerank.score_pairs(
        [(text, row[0]) for text, rows, _ in retrieved for row in rows]
    )

    ranked, offset = [], 0
    for text, rows, _ in retrieved:
        window = scores[offset : offset + len(rows)]
        offset += len(rows)
        best = sorted(zip(rows, window, strict=True), key=itemgetter(1), reverse=True)
        top = best[:k]
        ranked.append((text, [row for row, _ in top], [float(s) for _, s in top]))
    return ranked


def _phase_generate(
    retrieved: list,
    run_name: str,
    use_rerank: bool,
    language: str | None,
    k: int,
    model: str | None,
    job_id: int | None,
    variant: str,
    rerank_device: str | None = None,
    ef_search: int | None = None,
    allow_cpu: bool = False,
) -> tuple[int, bool]:
    answered = 0
    for text, rows, rerank_scores in retrieved:
        if job_id is not None and job_queue.is_cancelled(job_id):
            return answered, True
        try:
            chat.answer_from_rows(
                text,
                rows,
                rerank_scores=rerank_scores,
                add_context=True,
                run_name=run_name,
                use_rerank=use_rerank,
                language=language,
                k=k,
                model=model,
                phased=True,
                rerank_device=rerank_device,
                variant=variant,
                ef_search=ef_search,
            )
            answered += 1
        except Exception as e:
            log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
        # outside the try: a guard whose refusal the loop swallows is not a guard
        if answered == 1:
            _refuse_a_cpu_run(allow_cpu)
    return answered, False


def run_phased(
    run_name: str,
    texts: list[str],
    use_rerank: bool,
    language: str | None,
    k: int | None,
    model: str | None,
    job_id: int | None,
    variant: str,
    allow_cpu: bool = False,
) -> tuple[int, bool]:
    k = k or config.settings.retrieval.results_limit
    try:
        return _phased(
            run_name, texts, use_rerank, language, k, model, job_id, variant, allow_cpu
        )
    finally:
        # every exit, not only the last line: the refusal below raises, a cancel returns
        # early, and a rerank that throws skips both unloads. A run that leaves its own
        # generator on the card makes the retry refuse too, and the loop never breaks
        _free_the_card(model)


def _free_the_card(model: str | None) -> None:
    rerank.unload()
    llm.unload("embedding")
    llm.unload("generation", model=model)


def _phased(
    run_name: str,
    texts: list[str],
    use_rerank: bool,
    language: str | None,
    k: int,
    model: str | None,
    job_id: int | None,
    variant: str,
    allow_cpu: bool,
) -> tuple[int, bool]:
    rerank_device = None
    started = time.perf_counter()
    retrieved, ef_search = _phase_retrieve(texts, k, use_rerank, variant)
    log.info("eval_run.phase", name="retrieve", n=len(retrieved),
             elapsed=round(time.perf_counter() - started, 1))

    # asked here and not only after the first answer: the embedder is loaded and a lost
    # card is already visible, so a run that would answer off the card stops two minutes
    # in rather than after the generator has been loaded onto the processor
    _refuse_a_cpu_run(allow_cpu)

    # retrieval is over, and its model is 1.2 GiB the generator is about to want on a
    # card that holds 8
    llm.unload("embedding")

    if job_id is not None and job_queue.is_cancelled(job_id):
        return 0, True

    if use_rerank:
        llm.unload("generation", model=model)
        started = time.perf_counter()
        retrieved = _phase_rerank(retrieved, k)
        log.info("eval_run.phase", name="rerank", n=len(retrieved),
                 elapsed=round(time.perf_counter() - started, 1))
        rerank_device = rerank.device()
        rerank.unload()

        if job_id is not None and job_queue.is_cancelled(job_id):
            return 0, True

    started = time.perf_counter()
    answered, cancelled = _phase_generate(
        retrieved, run_name, use_rerank, language, k, model, job_id, variant,
        rerank_device, ef_search, allow_cpu,
    )
    log.info("eval_run.phase", name="generate", n=answered,
             elapsed=round(time.perf_counter() - started, 1))
    return answered, cancelled


# the runner asks the same resolver every other caller does: a default is only one
# default if one place decides it
from use_cases.chat import resolve_rerank  # noqa: E402


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
    fallback_policy: str | None = None,
    gate_signal: str | None = None,
    weak_distance: float | None = None,
    topic_threshold: float | None = None,
    orchestrator: str | None = None,
    job_id: int | None = None,
    phased: bool | None = None,
    allow_cpu: bool = False,
    variant: str | None = None,
) -> int:
    pipeline = Pipeline(pipeline)
    variant = variant or config.settings.corpus.variant
    # both checks belong at the start: an hour of answers is a poor way to learn the variant is wrong
    config.settings.corpus.policy(variant)
    known = db.corpus_variants()
    if db.is_empty(variant=variant):
        raise RuntimeError(
            f"corpus variant '{variant}' is empty; known variants: "
            f"{[v['variant'] for v in known]}"
        )
    log.info("eval_run.corpus", variant=variant, known=known)
    texts = _target_texts(set_name, question_ids)
    use_rerank = resolve_rerank(use_rerank)
    if phased is None:
        phased = pipeline == Pipeline.single_shot

    if phased and pipeline == Pipeline.single_shot:
        answered, cancelled = run_phased(
            run_name, texts, use_rerank, language, k, model, job_id, variant, allow_cpu
        )
    else:
        answered, cancelled = _run_sequential(
            texts, run_name, use_rerank, pipeline, language, k, max_hops, model,
            fallback_policy, gate_signal, weak_distance, topic_threshold, orchestrator,
            job_id, allow_cpu, variant,
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
