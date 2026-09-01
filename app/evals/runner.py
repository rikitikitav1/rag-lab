import sys
import time
from dataclasses import dataclass, replace
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
from use_cases import agent, chat, search_depth

# the same resolver every other caller asks: a default is one default only if one decides
from use_cases.chat import resolve_rerank

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


# the knobs a run answers with, carried whole: fourteen positional arguments, then sixteen
@dataclass(frozen=True)
class RunSpec:
    variant: str
    pipeline: Pipeline = Pipeline.single_shot
    use_rerank: bool | None = None
    language: str | None = None
    k: int | None = None
    max_hops: int | None = None
    model: str | None = None
    fallback_policy: str | None = None
    gate_signal: str | None = None
    weak_distance: float | None = None
    topic_threshold: float | None = None
    orchestrator: str | None = None


def _answer_one(text: str, run_name: str, spec: RunSpec) -> None:
    if spec.pipeline == Pipeline.agent:
        agent.run(
            text,
            run_name=run_name,
            language=spec.language,
            k=spec.k,
            max_hops=spec.max_hops,
            use_rerank=spec.use_rerank,
            model=spec.model,
            fallback_policy=spec.fallback_policy,
            gate_signal=spec.gate_signal,
            weak_distance=spec.weak_distance,
            topic_threshold=spec.topic_threshold,
            orchestrator=spec.orchestrator,
            variant=spec.variant,
        )
    elif spec.pipeline == Pipeline.single_shot:
        chat.answer(
            text,
            add_context=True,
            run_name=run_name,
            use_rerank=spec.use_rerank,
            language=spec.language,
            k=spec.k,
            model=spec.model,
            variant=spec.variant,
        )
    else:
        raise ValueError(f"unknown pipeline: {spec.pipeline}")


def _refuse_a_cpu_run(allow_cpu: bool, use_rerank: bool = False) -> None:
    # ollama drops the card and keeps answering: same numbers, four times the hours
    llm.warn_if_models_do_not_fit()
    off_card = llm.models_off_the_card()
    # ollama cannot see the cross-encoder: it is torch in this process and is loaded here
    if use_rerank:
        rerank.warm()
    spilled = rerank.off_the_card()
    if spilled:
        off_card = [*off_card, spilled]
    if off_card and not allow_cpu:
        raise RuntimeError(
            f"models are not on the GPU: {', '.join(off_card)}."
            " Pass allow_cpu if this run is meant to measure the CPU"
        )
    if off_card:
        log.warning("eval_run.cpu_allowed", models=off_card)


def _run_sequential(
    texts: list[str], run_name: str, spec: RunSpec, *, job_id: int | None, allow_cpu: bool
) -> tuple[int, bool]:
    answered = 0
    # without giving the card back, arm two loads its generator beside arm one's
    try:
        for text in texts:
            if job_id is not None and job_queue.is_cancelled(job_id):
                return answered, True
            # after the first answer the generator is loaded, so a spill is finally visible
            if answered == 1:
                _refuse_a_cpu_run(allow_cpu, spec.use_rerank)
            try:
                _answer_one(text, run_name, spec)
                answered += 1
            except Exception as e:
                log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
        return answered, False
    finally:
        _free_the_card(spec.model)


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


def _phase_retrieve(texts: list[str], spec: RunSpec) -> tuple[list, int]:
    limit = config.settings.rerank.candidates if spec.use_rerank else spec.k
    # resolved once and carried: a phased run recorded `ef_search: null`, and phased is default
    depth = search_depth.resolve(spec.variant)
    retrieved = []
    for text, vector in zip(texts, _embed_in_batches(texts), strict=True):
        if vector is None:
            continue
        try:
            retrieved.append(
                (
                    text,
                    db.hybrid_search(text, vector, None, limit=limit, variant=spec.variant,
                                     ef_search=depth),
                    None,
                )
            )
        except Exception as e:
            log.error("eval_run.search_failed", run_text=text[:80], error=str(e))
    return retrieved, depth


def _phase_rerank(retrieved: list, k: int) -> list:
    scores = rerank.score_pairs(
        [(text, hit.content) for text, rows, _ in retrieved for hit in rows]
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
    spec: RunSpec,
    *,
    job_id: int | None = None,
    allow_cpu: bool = False,
    rerank_device: str | None = None,
    ef_search: int | None = None,
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
                use_rerank=spec.use_rerank,
                language=spec.language,
                k=spec.k,
                model=spec.model,
                phased=True,
                rerank_device=rerank_device,
                variant=spec.variant,
                ef_search=ef_search,
            )
            answered += 1
        except Exception as e:
            log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
        # outside the try: a guard whose refusal the loop swallows is not a guard
        if answered == 1:
            _refuse_a_cpu_run(allow_cpu, spec.use_rerank)
    return answered, False


def run_phased(
    texts: list[str], run_name: str, spec: RunSpec, *,
    job_id: int | None = None, allow_cpu: bool = False,
) -> tuple[int, bool]:
    spec = replace(spec, k=spec.k or config.settings.retrieval.results_limit)
    try:
        return _phased(texts, run_name, spec, job_id=job_id, allow_cpu=allow_cpu)
    finally:
        # every exit: a run that leaves its generator on the card makes the retry refuse too
        _free_the_card(spec.model)


def _free_the_card(model: str | None) -> None:
    rerank.unload()
    llm.unload("embedding")
    llm.unload("generation", model=model)


def _phased(
    texts: list[str], run_name: str, spec: RunSpec, *, job_id: int | None, allow_cpu: bool
) -> tuple[int, bool]:
    rerank_device = None
    started = time.perf_counter()
    retrieved, ef_search = _phase_retrieve(texts, spec)
    log.info("eval_run.phase", name="retrieve", n=len(retrieved),
             elapsed=round(time.perf_counter() - started, 1))

    # here, so a run that would answer off the card stops two minutes in
    _refuse_a_cpu_run(allow_cpu, spec.use_rerank)

    # retrieval is over, and its model is 1.2 GiB the generator wants on a card that holds 8
    llm.unload("embedding")

    if job_id is not None and job_queue.is_cancelled(job_id):
        return 0, True

    if spec.use_rerank:
        llm.unload("generation", model=spec.model)
        started = time.perf_counter()
        retrieved = _phase_rerank(retrieved, spec.k)
        log.info("eval_run.phase", name="rerank", n=len(retrieved),
                 elapsed=round(time.perf_counter() - started, 1))
        rerank_device = rerank.device()
        rerank.unload()

        if job_id is not None and job_queue.is_cancelled(job_id):
            return 0, True

    started = time.perf_counter()
    answered, cancelled = _phase_generate(
        retrieved, run_name, spec, job_id=job_id, allow_cpu=allow_cpu,
        rerank_device=rerank_device, ef_search=ef_search,
    )
    log.info("eval_run.phase", name="generate", n=answered,
             elapsed=round(time.perf_counter() - started, 1))
    return answered, cancelled


def _walks_the_index(variant: str, depth: int) -> bool:
    with db.engine.connect() as conn:
        return search_depth.uses_index(conn, variant, depth)


def run(
    run_name: str,
    set_name: str | None = None,
    question_ids: list[int] | None = None,
    use_rerank: bool | None = None,
    pipeline: str = Pipeline.single_shot,
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
    # the preflight answers from before the queue moved, and the crossover can shift
    depth = search_depth.resolve(variant)
    if not _walks_the_index(variant, depth):
        raise RuntimeError(
            f"variant '{variant}' at ef_search {depth} no longer walks its index: the plan"
            " sorts, so this run would measure exact search and record hnsw"
        )
    log.info("eval_run.corpus", variant=variant, known=known, ef_search=depth)
    texts = _target_texts(set_name, question_ids)
    use_rerank = resolve_rerank(use_rerank)
    if phased is None:
        phased = pipeline == Pipeline.single_shot
    spec = RunSpec(
        variant=variant,
        pipeline=pipeline,
        use_rerank=use_rerank,
        language=language,
        k=k,
        max_hops=max_hops,
        model=model,
        fallback_policy=fallback_policy,
        gate_signal=gate_signal,
        weak_distance=weak_distance,
        topic_threshold=topic_threshold,
        orchestrator=orchestrator,
    )

    if phased and pipeline == Pipeline.single_shot:
        answered, cancelled = run_phased(
            texts, run_name, spec, job_id=job_id, allow_cpu=allow_cpu
        )
    else:
        answered, cancelled = _run_sequential(
            texts, run_name, spec, job_id=job_id, allow_cpu=allow_cpu
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
