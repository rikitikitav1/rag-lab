import sys
import time
from dataclasses import dataclass, replace
from operator import itemgetter

import config
import job_queue
import llm
import logging_setup
import passes
import rerank
from engines import card
from errors import StandFault
from models.eval import Question, QuestionLog
from models.registry import Pipeline, Role
from orm.sync_db import Session
from outcomes import Outcome
from passes import Pass, Seat
from sqlalchemy import delete, func, select
from use_cases import agent, chat, run_snapshot, search_depth

# the same resolver every other caller asks: a default is one default only if one decides
from use_cases.chat import resolve_rerank

import db

log = logging_setup.get_logger(__name__)


# an unnamed target is every question there is: `__noop__` swept the corpus through a typed door
def _target_texts(set_name: str | None, question_ids: list[int] | None) -> list[str]:
    if not question_ids and not set_name:
        raise ValueError("a run needs a target: name a set or the question ids, not neither")
    with Session() as session:
        if question_ids:
            found = session.execute(
                select(Question.id, Question.original_text).where(Question.id.in_(question_ids))
            ).all()
            _refuse_missing(question_ids, {qid for qid, _ in found})
            return [text for _, text in found]
        return list(session.scalars(select(Question.original_text).where(Question.set_name == set_name)))


class MissingQuestions(StandFault):
    pass


# a floor measured on 50 questions that quietly became 48 is read as a floor on 50
def _refuse_missing(asked: list[int], found: set[int]) -> None:
    missing = sorted(set(asked) - found)
    if missing:
        raise MissingQuestions(
            f"{len(missing)} of {len(set(asked))} question ids are not in the stand: {missing[:20]}"
        )


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
    restate_tools: bool = False
    grade_chunks: bool = False
    weak_distance: float | None = None
    topic_threshold: float | None = None
    orchestrator: str | None = None
    # carried onto every row: whether the sweep may ever judge what this run wrote
    judge_wanted: bool = True


# answered is a row that answered: the agent writes a row for a hop that failed and returns normally
def _answer_one(text: str, run_name: str, spec: RunSpec) -> bool:
    if spec.pipeline == Pipeline.agent:
        result = agent.run(
            text,
            run_name=run_name,
            language=spec.language,
            k=spec.k,
            max_hops=spec.max_hops,
            use_rerank=spec.use_rerank,
            model=spec.model,
            fallback_policy=spec.fallback_policy,
            gate_signal=spec.gate_signal,
            restate_tools=spec.restate_tools,
            grade_chunks=spec.grade_chunks,
            weak_distance=spec.weak_distance,
            topic_threshold=spec.topic_threshold,
            orchestrator=spec.orchestrator,
            variant=spec.variant,
            judge_wanted=spec.judge_wanted,
        )
        return not result.failed and result.outcome != Outcome.error
    if spec.pipeline == Pipeline.single_shot:
        chat.answer(
            text,
            add_context=True,
            run_name=run_name,
            use_rerank=spec.use_rerank,
            language=spec.language,
            k=spec.k,
            model=spec.model,
            variant=spec.variant,
            grade_chunks=spec.grade_chunks,
            judge_wanted=spec.judge_wanted,
        )
        return True
    raise ValueError(f"unknown pipeline: {spec.pipeline}")


# the role names a model and the model names its engine: releasing the card asks that engine
def _release(role: str, model: str | None = None) -> None:
    try:
        picked = llm.resolve_for(role, model)
    except Exception as e:
        log.warning("eval_run.release_skipped", role=role, error=str(e))
        return
    card.release_model(picked.engine, picked.name)


# the pass's own spill rule, asked where a phase ends: a batch of vectors has no rows to stop between
def _refuse_a_cpu_run(roles: tuple[Role, ...], allow_cpu: bool, model: str | None) -> None:
    passes.refuse_spill([Seat(role, model if role == Role.generation else None) for role in roles], allow_cpu)


def _run_sequential(
    texts: list[str], run_name: str, spec: RunSpec, *, job_id: int | None, allow_cpu: bool
) -> tuple[int, bool]:
    answered = 0
    # the generator took the card at the door; the embedder loads on the first search, so a row later
    walk = Pass(job_id, (Seat(Role.generation, spec.model), Seat(Role.embedding, spill_from=1)),
                allow_spill=allow_cpu)
    # without giving the card back, arm two loads its generator beside arm one's
    try:
        for text in texts:
            if not walk.before_row():
                return answered, True
            try:
                with llm.placements():
                    answered += _answer_one(text, run_name, spec)
                    walk.after_row(llm.placed_in_calls())
            except StandFault:
                raise
            except Exception as e:
                log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
        walk.close(owed=len(texts), done=answered, nothing=NoAnswers)
        return answered, False
    finally:
        _free_the_card(spec.model)


# each vector with the label of the embedder that made it: a role moved mid-run mislabels none
def _embed_in_batches(texts: list[str]) -> list[tuple[str | None, list | None]]:
    size = config.settings.ingestion.batch_size
    embedded: list = []
    for start in range(0, len(texts), size):
        chunk = texts[start : start + size]
        try:
            label, vectors = llm.embed_labelled(chunk)
            embedded.extend((label, vector) for vector in vectors)
        except StandFault:
            raise
        except Exception as e:
            log.error("eval_run.embed_failed", start=start, n=len(chunk), error=str(e))
            embedded.extend([(None, None)] * len(chunk))
    return embedded


def _phase_retrieve(texts: list[str], spec: RunSpec) -> tuple[list, int]:
    limit = config.settings.rerank.candidates if spec.use_rerank else spec.k
    # resolved once and carried: a phased run recorded `ef_search: null`, and phased is default
    depth = search_depth.resolve(spec.variant)
    retrieved = []
    for text, (label, vector) in zip(texts, _embed_in_batches(texts), strict=True):
        if vector is None:
            continue
        try:
            retrieved.append(
                (
                    text,
                    db.hybrid_search(text, vector, None, limit=limit, variant=spec.variant,
                                     ef_search=depth, embedded_by=label),
                    None,
                )
            )
        except StandFault:
            raise
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
    walk: Pass,
    rerank_device: str | None = None,
    ef_search: int | None = None,
    placed_during: dict | None = None,
) -> tuple[int, bool]:
    answered = 0
    for text, rows, rerank_scores in retrieved:
        if not walk.before_row():
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
                placed_during=placed_during,
                grade_chunks=spec.grade_chunks,
                judge_wanted=spec.judge_wanted,
            )
            answered += 1
        except StandFault:
            raise
        except Exception as e:
            log.error("eval_run.answer_failed", run_name=run_name, error=str(e))
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
    _release("embedding")
    _release("generation", model)


def _phased(
    texts: list[str], run_name: str, spec: RunSpec, *, job_id: int | None, allow_cpu: bool
) -> tuple[int, bool]:
    rerank_device = None
    # the embedder was given back after retrieval, and the generator loads on its first answer
    walk = Pass(job_id, (Seat(Role.generation, spec.model, spill_from=1),), allow_spill=allow_cpu)
    started = time.perf_counter()
    retrieved, ef_search = _phase_retrieve(texts, spec)
    log.info("eval_run.phase", name="retrieve", n=len(retrieved),
             elapsed=round(time.perf_counter() - started, 1))

    # here, so a run whose embedder spilled stops two minutes in, before the generator is paid for
    _refuse_a_cpu_run((Role.embedding,), allow_cpu, spec.model)

    # read before the release: every row is written after it, and read then the embedder is gone
    placed_during = {"embedding": run_snapshot.placed("embedding")}
    # retrieval is over, and its model is 1.2 GiB the generator wants on a card that holds 8
    _release("embedding")

    if walk.cancelled():
        return 0, True

    if spec.use_rerank:
        _release("generation", spec.model)
        started = time.perf_counter()
        retrieved = _phase_rerank(retrieved, spec.k)
        log.info("eval_run.phase", name="rerank", n=len(retrieved),
                 elapsed=round(time.perf_counter() - started, 1))
        rerank_device = rerank.device()
        placed_during["reranking"] = run_snapshot.placed("reranking")

        if walk.cancelled():
            return 0, True

    started = time.perf_counter()
    answered, cancelled = _phase_generate(
        retrieved, run_name, spec, walk=walk,
        rerank_device=rerank_device, ef_search=ef_search, placed_during=placed_during,
    )
    if not cancelled:
        walk.close(owed=len(texts), done=answered, nothing=NoAnswers)
    log.info("eval_run.phase", name="generate", n=answered,
             elapsed=round(time.perf_counter() - started, 1))
    return answered, cancelled


def _walks_the_index(variant: str, depth: int) -> bool:
    with db.engine.connect() as conn:
        return search_depth.uses_index(conn, variant, depth)


# the runner's exit guard is the pass's own, named for what a run owes
class NoAnswers(passes.NothingDone):
    pass


# a row that is not an error answered its question; an error row is replaced, and the new one says so
def _split_answered(texts: list[str], rows: list[tuple]) -> tuple[list[str], dict[str, dict]]:
    answered = _answered(rows)
    return [t for t in texts if t not in answered], _errors_of(rows, {text for _, text, _ in rows} - answered)


def _answered(rows: list[tuple]) -> set[str]:
    return {text for _, text, metrics in rows if (metrics or {}).get("outcome") != "error"}


def _errors_of(rows: list[tuple], texts: set[str]) -> dict[str, dict]:
    held: dict[str, dict] = {}
    for log_id, text, metrics in rows:
        if text in texts and (metrics or {}).get("outcome") == "error":
            one = held.setdefault(text, {"log_ids": [], "outcome": "error", "failed": None})
            one["log_ids"].append(log_id)
            one["failed"] = (metrics or {}).get("failed") or one["failed"]
    return held


# a worker that died mid-resume left the question answered and its error rows in place
def _settle_left_behind(session, run_name: str, left: dict[str, dict]) -> None:
    rows = session.execute(
        select(QuestionLog, Question.original_text)
        .join(Question, Question.id == QuestionLog.question_id)
        .where(QuestionLog.run_name == run_name, Question.original_text.in_(list(left)))
    ).all()
    for ql, text in rows:
        if (ql.metrics or {}).get("outcome") != "error":
            ql.metrics = {**(ql.metrics or {}), "resumed": True, "resumed_from": left[text]}
    session.execute(delete(QuestionLog).where(QuestionLog.id.in_([i for one in left.values() for i in one["log_ids"]])))
    session.commit()
    log.warning("eval_run.settled_left_behind", run_name=run_name, questions=len(left))


# one row per question stays true: `requeued_stale` duplicates are caught by that count
def _still_to_answer(run_name: str, texts: list[str]):
    with Session() as session:
        since = session.scalar(select(func.now()))
        rows = session.execute(
            select(QuestionLog.id, Question.original_text, QuestionLog.metrics)
            .join(Question, Question.id == QuestionLog.question_id)
            .where(QuestionLog.run_name == run_name)
        ).all()
        rows = [tuple(r) for r in rows]
        todo, replaced = _split_answered(texts, rows)
        left = _errors_of(rows, _answered(rows))
        if left:
            _settle_left_behind(session, run_name, left)
    log.info("eval_run.resumed", run_name=run_name, still_to_answer=len(todo), of=len(texts), to_replace=len(replaced))
    return todo, replaced, since


# only a question that got a new row is marked and loses its error rows: a resume stopped again keeps the rest
def _mark_resumed(run_name: str, texts: list[str], *, replaced: dict[str, dict], since) -> None:
    if not texts:
        return
    with Session() as session:
        rows = session.execute(
            select(QuestionLog, Question.original_text)
            .join(Question, Question.id == QuestionLog.question_id)
            .where(
                QuestionLog.run_name == run_name,
                QuestionLog.created_at >= since,
                Question.original_text.in_(texts),
            )
        ).all()
        for ql, text in rows:
            ql.metrics = {
                **(ql.metrics or {}), "resumed": True,
                **({"resumed_from": replaced[text]} if text in replaced else {}),
            }
        gone = [log_id for text in {text for _, text in rows} & set(replaced) for log_id in replaced[text]["log_ids"]]
        if gone:
            session.execute(delete(QuestionLog).where(QuestionLog.id.in_(gone)))
        session.commit()


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
    restate_tools: bool = False,
    grade_chunks: bool = False,
    weak_distance: float | None = None,
    topic_threshold: float | None = None,
    orchestrator: str | None = None,
    job_id: int | None = None,
    phased: bool | None = None,
    allow_cpu: bool = False,
    variant: str | None = None,
    resume: bool = False,
    generation_sampler: dict | None = None,
    judge: bool = True,
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
    replaced, since = {}, None
    if resume:
        texts, replaced, since = _still_to_answer(run_name, texts)
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
        restate_tools=restate_tools,
        grade_chunks=grade_chunks,
        weak_distance=weak_distance,
        topic_threshold=topic_threshold,
        orchestrator=orchestrator,
        judge_wanted=judge,
    )

    try:
        with llm.run_sampler(generation_sampler):
            if phased and pipeline == Pipeline.single_shot:
                answered, cancelled = run_phased(
                    texts, run_name, spec, job_id=job_id, allow_cpu=allow_cpu
                )
            else:
                answered, cancelled = _run_sequential(
                    texts, run_name, spec, job_id=job_id, allow_cpu=allow_cpu
                )
    except BaseException:
        # a resume is stopped again for the reasons it is resumed, and what it wrote by then still says so
        if resume:
            try:
                _mark_resumed(run_name, texts, replaced=replaced, since=since)
            except StandFault:
                raise
            except Exception as e:
                # the stop is what the caller must see, not a mark that failed on its way out
                log.error("eval_run.mark_failed", run_name=run_name, error=str(e))
        raise
    if resume:
        _mark_resumed(run_name, texts, replaced=replaced, since=since)
    # a run whose number is not a judged score pays for the judge's residency and its noise for nothing
    if not cancelled and judge:
        job_queue.enqueue("judge_answers", {"run_name": run_name})
    log.info(
        "eval_run.answered",
        run_name=run_name,
        answered=answered,
        total=len(texts),
        cancelled=cancelled,
        judge=judge,
    )
    return answered


if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else "curated"
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"{set_name}_{int(time.time())}"
    n = run(run_name, set_name=set_name)
    print(f"run: {run_name} | set: {set_name} | answered: {n} | judging enqueued")
