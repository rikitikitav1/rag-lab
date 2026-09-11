import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
import job_queue
import llm
import logging_setup
import outcomes
import prompt_repo
import sources.base
from models.eval import Question, QuestionLog, text_hash
from models.registry import Purpose
from orm.sync_db import Session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from timing_wrappers import measure_elapsed
from use_cases import run_snapshot, search_depth

import db

log = logging_setup.get_logger(__name__)


NO_RESULTS = outcomes.NO_RESULTS


@dataclass
class Source:
    source: str
    vector_rank: float | None
    keyword_rank: float | None
    vector_distance: float | None
    score: float
    rerank_score: float | None = None
    # which hop returned it: None is single_shot and every row written before this
    hop: int | None = None

    def __str__(self) -> str:
        return (
            f"{self.source} ({self.score}: vector_rank={self.vector_rank}, "
            f"keyword_rank={self.keyword_rank}, vector_distance={self.vector_distance})"
        )


@dataclass
class Retrieval:
    elapsed: float = 0.0
    sources: list[Source] = field(default_factory=list)

    def __str__(self) -> str:
        sources = "\n".join(str(s) for s in self.sources)
        return f"elapsed: {self.elapsed}s\nSources:\n{sources}"


@dataclass
class AnswerMetric:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    distance_threshold: float = field(
        default_factory=lambda: round(config.settings.retrieval.distance_threshold, 3)
    )
    model: str = field(default_factory=lambda: llm.resolve_name("generation"))

    def __str__(self) -> str:
        return (
            f"Model: {self.model}, distance threshold: {self.distance_threshold}, "
            f"prompt_tokens: {self.prompt_tokens}, completion_tokens: {self.completion_tokens}"
        )


@dataclass
class Answer:
    text: str
    elapsed: float = 0.0
    context: str | None = None
    success: bool = False
    sources: list[Source] = field(default_factory=list)
    metrics: AnswerMetric = field(default_factory=AnswerMetric)

    def __str__(self) -> str:
        sources = "\n".join(str(s) for s in self.sources)
        return (
            f"{self.text}\n\n"
            f"Success: {self.success}, elapsed: {self.elapsed}s, {self.metrics}\n"
            f"Sources:\n{sources}"
        )


def _source_from_row(
    src, vector_rank, keyword_rank, vector_distance, score, rerank_score=None
) -> Source:
    return Source(
        source=src,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        vector_distance=round(vector_distance, 3)
        if vector_distance is not None
        else None,
        score=round(float(score), 3),
        rerank_score=round(float(rerank_score), 3) if rerank_score is not None else None,
    )


# one place decides whether reranking happens: the number of sites is what makes a default
def resolve_rerank(use_rerank: bool | None) -> bool:
    if use_rerank is None:
        return config.settings.rerank.enabled
    return use_rerank


# stamped where the sources are collected, or every hop looks like the first
def stamped(sources: list, hop: int) -> list:
    for source in sources:
        source.hop = hop
    return sources


def take_sources(rows, rerank_scores=None, variant: str | None = None) -> list[Source]:
    variant = variant or config.settings.corpus.variant
    scores = rerank_scores or [None] * len(rows)
    kept: dict[str, Source] = {}
    for hit, rerank_score in zip(rows, scores, strict=True):
        if _hidden_by_cut(hit.source, variant):
            continue
        if hit.source in kept:
            # a duplicated path must not hide the best cross-encoder score from the gate
            best = kept[hit.source].rerank_score
            if rerank_score is not None and (best is None or rerank_score > best):
                kept[hit.source].rerank_score = round(float(rerank_score), 3)
            continue
        kept[hit.source] = _source_from_row(
            hit.source, hit.vector_rank, hit.keyword_rank, hit.distance, hit.score,
            rerank_score,
        )
    return list(kept.values())


# baseline only: it filters after the search took k, so an answer can come back with fewer
LEGACY_SKIP_NAMES = frozenset({"index.md"})


def _hidden_by_cut(source: str, variant: str) -> bool:
    # raising here would do it once per retrieved row in the middle of an answer
    policy = config.settings.corpus.policy_or_none(variant)
    if policy is not None and sources.base.hygienic(policy):
        return False
    return Path(source).name in LEGACY_SKIP_NAMES


# resolving a second time is a second answer, so the depth comes back with the rows
def _retrieve_rows(question: str, category, k: int, rerank_enabled: bool, variant: str,
                   ef_search: int | None = None):
    depth = search_depth.resolve(variant, ef_search)
    if not rerank_enabled:
        return (
            db.hybrid_search(
                question, llm.embed(question), category, limit=k, variant=variant,
                ef_search=depth, embedded_by=llm.embedder_label(),
            ),
            None,
            depth,
        )

    import rerank

    candidates = db.hybrid_search(
        question,
        llm.embed(question),
        category,
        limit=config.settings.rerank.candidates,
        variant=variant,
        ef_search=depth,
        embedded_by=llm.embedder_label(),
    )
    ranked = rerank.rerank(question, candidates, top=k)
    return [row for row, _ in ranked], [score for _, score in ranked], depth


# one filter, one order, one pass: the text and its address cannot come out different lengths
def kept_chunks(rows, variant: str | None = None) -> tuple[list[str], list[dict]]:
    variant = variant or config.settings.corpus.variant
    kept = [hit for hit in rows if not _hidden_by_cut(hit.source, variant)]
    texts = [f"[{hit.source}]\n{hit.content}" for hit in kept]
    chunks = [
        {"source": hit.source, "section": hit.section, "chunk_index": hit.chunk_index}
        for hit in kept
    ]
    return texts, chunks


# the join is built from these same elements, so the two cannot drift
def chunk_texts(rows, variant: str | None = None) -> list[str]:
    return kept_chunks(rows, variant)[0]


def format_chunks(rows, variant: str | None = None) -> str:
    return "\n\n".join(chunk_texts(rows, variant))


def _gate_scores(query: str, rows, top: int) -> list:
    import rerank

    head = rows[:top]
    scores = rerank.score_pairs([(query, hit.content) for hit in head])
    return [float(s) for s in scores] + [None] * (len(rows) - len(head))


def search_chunks(
    query: str,
    category: str | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
    gate_top: int | None = None,
    *,
    variant: str,
) -> tuple[str, list[str], list[Source], int, list[dict]]:
    k = k or config.settings.retrieval.results_limit
    use_rerank = resolve_rerank(use_rerank)
    rows, rerank_scores, depth = _retrieve_rows(query, category, k, use_rerank, variant)
    if not rows:
        return NO_RESULTS, [], [], depth, []
    if rerank_scores is None and gate_top:
        rerank_scores = _gate_scores(query, rows, gate_top)
    texts, chunks = kept_chunks(rows, variant)
    return (
        "\n\n".join(texts) or NO_RESULTS,
        texts,
        take_sources(rows, rerank_scores, variant),
        depth,
        chunks,
    )


@measure_elapsed
def retrieve(
    question: str,
    category: str | None = None,
    k: int | None = None,
    *,
    variant: str,
    ef_search: int | None = None,
) -> Retrieval:
    k = k or config.settings.retrieval.results_limit
    rows, rerank_scores, _depth = _retrieve_rows(
        question, category, k, resolve_rerank(None), variant, ef_search
    )
    return Retrieval(sources=take_sources(rows, rerank_scores, variant))


def answer(
    question: str,
    category: str | None = None,
    k: int | None = None,
    add_context=False,
    run_name: str | None = None,
    use_rerank: bool | None = None,
    language: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    ef_search: int | None = None,
) -> Answer:
    start = time.perf_counter()
    use_rerank = resolve_rerank(use_rerank)
    k = k or config.settings.retrieval.results_limit
    variant = variant or config.settings.corpus.variant
    rows, rerank_scores, depth = _retrieve_rows(
        question, category, k, use_rerank, variant, ef_search
    )
    return answer_from_rows(
        question,
        rows,
        rerank_scores=rerank_scores,
        add_context=add_context,
        run_name=run_name,
        use_rerank=use_rerank,
        language=language,
        model=model,
        k=k,
        started_at=start,
        variant=variant,
        ef_search=depth,
    )


def answer_from_rows(
    question: str,
    rows,
    rerank_scores=None,
    add_context=False,
    run_name: str | None = None,
    use_rerank: bool | None = None,
    language: str | None = None,
    model: str | None = None,
    k: int | None = None,
    started_at: float | None = None,
    phased: bool = False,
    rerank_device: str | None = None,
    ef_search: int | None = None,
    *,
    variant: str,
    placed_during: dict | None = None,
) -> Answer:
    start = started_at if started_at is not None else time.perf_counter()
    lang = resolve_language(question, language)
    use_rerank = resolve_rerank(use_rerank)
    k = k or config.settings.retrieval.results_limit

    texts, chunks = kept_chunks(rows, variant) if rows else ([], [])
    context = "\n\n".join(texts) or None
    if not context:
        ans = Answer(text=NO_RESULTS)
    else:
        user = f"{context}\n\nQuestion: {question}"
        # always, not only when a run forced one: without it the model follows whatever it last read
        user = told_to_answer_in(user, lang)
        response = llm.ask(
            system=prompt_repo.active_template(Purpose.generate_answer),
            user=user,
            model=model,
        )
        metrics = AnswerMetric(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        if model:
            metrics.model = model
        ans = Answer(
            text=response.text,
            success=True,
            sources=take_sources(rows, rerank_scores, variant),
            metrics=metrics,
        )
        if add_context:
            ans.context = context

    ans.elapsed = round(time.perf_counter() - start, 3)

    try:
        _log_answer(
            question, ans, lang, context, run_name, use_rerank, k, phased, rerank_device,
            _retrieval_snapshot(rows, ans.sources), variant=variant, ef_search=ef_search,
            contexts=texts or None, chunks=chunks or None, placed_during=placed_during,
        )
    except SQLAlchemyError as e:
        log.error("question_log.insert_failed", reason=str(e))

    return ans


_LANG_NAMES = {"ru": "Russian", "en": "English"}


# re-exported: three callers above this layer already say `chat.resolve_language`
resolve_language = db.resolve_language


# an unknown code is not a language name, and `replay` reads this out of a snapshot past the doors
def language_directive(language: str) -> str:
    said = _LANG_NAMES.get(language)
    return f"Respond in {said}." if said else ""


# the append was written three times with its own empty guard, and replay has to match all of them
def told_to_answer_in(text: str, language: str) -> str:
    said = language_directive(language)
    return f"{text}\n\n{said}" if said else text


def _retrieval_snapshot(rows, sources) -> dict:
    distances = [hit.distance for hit in rows if hit.distance is not None]
    rerank_scores = [s.rerank_score for s in sources if s.rerank_score is not None]
    return run_snapshot.of_retrieval(
        results_count=len(rows),
        min_distance=min(distances) if distances else None,
        top_rerank_score=max(rerank_scores) if rerank_scores else None,
    )


def _config_snapshot(use_rerank, k, phased, distance_threshold, rerank_device, variant: str,
                     ef_search: int | None = None, model: str | None = None,
                     language: str | None = None, placed_during: dict | None = None) -> dict:
    return run_snapshot.of_run(
        language=language,
        variant=variant,
        use_rerank=use_rerank,
        k=k,
        ef_search=ef_search,
        distance_threshold=distance_threshold,
        model=model,
        rerank_device=rerank_device,
        placed_during=placed_during,
        # the agent has no phase and single_shot has no hops: each records None for the other
        phased=phased,
    )


def _log_answer(
    original_text: str, ans: Answer, lang: str, context=None, run_name=None,
    use_rerank=False, k=None, phased=False, rerank_device=None, retrieval=None,
    *, variant: str, ef_search: int | None = None, contexts=None, chunks=None,
    placed_during: dict | None = None,
) -> None:
    with Session() as session:
        question = _find_or_create_question(session, original_text, lang)
        log_row = QuestionLog(
            run_name=run_name,
            question_id=question.id,
            question_text=question.original_text,
            reference_answer=question.reference_answer,
            answered=ans.success,
            answer=ans.text,
            context=context,
            contexts=contexts,
            chunks=chunks,
            sources=[asdict(s) for s in ans.sources],
            models={
                "generation": ans.metrics.model,
                "embedding": llm.resolve_name("embedding"),
                # the cross-encoder left the config for a role, and a run that reranked names its model
                **({"reranking": llm.resolve_name("reranking")} if use_rerank else {}),
            },
            prompts={
                "generate_answer": prompt_repo.active_version(Purpose.generate_answer)
            },
            metrics={
                "config": _config_snapshot(
                    use_rerank, k, phased, ans.metrics.distance_threshold,
                    rerank_device, variant, ef_search, ans.metrics.model, lang,
                    placed_during=placed_during,
                ),
                "retrieval": retrieval,
                # what the ceiling grid is gated on, as a number rather than arithmetic done by hand
                "context_chars": len(context) if context else 0,
                # what this path can know now; groundedness waits for the judge, see `settled_outcome`
                "outcome": outcomes.classify(ans.text, bool(ans.sources)),
                # the one fact both the judge and the report may read: neither re-derives it
                "refusal": outcomes.reads_as_refusal(ans.text),
            },
            prompt_tokens=ans.metrics.prompt_tokens,
            completion_tokens=ans.metrics.completion_tokens,
            elapsed=ans.elapsed,
        )
        session.add(log_row)
        session.commit()
        log_id = log_row.id

    _judge_later(ans, run_name, log_id)


# a live answer joins the waiting batch; a run's answers are judged by the run's own job
def _judge_later(ans: Answer, run_name: str | None, log_id: int) -> None:
    if ans.success and run_name is None:
        job_queue.judge_live(log_id)


def _find_or_create_question(session, original_text, lang, set_name="live"):
    stmt = (
        insert(Question)
        .values(
            original_text=original_text,
            set_name=set_name,
            language=lang,
            text_hash=text_hash(original_text),
        )
        .on_conflict_do_update(
            index_elements=["text_hash"],
            set_={"original_text": Question.original_text},
        )
        .returning(Question)
    )
    return session.scalar(select(Question).from_statement(stmt))
