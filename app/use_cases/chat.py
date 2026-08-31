import hashlib
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
from models.eval import Question, QuestionLog
from models.registry import Purpose
from orm.sync_db import Session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from timing_wrappers import measure_elapsed
from use_cases import search_depth

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


# one place decides whether reranking happens, and every caller asks it rather than
# reading the key: the number of decision sites is what makes a default true
def resolve_rerank(use_rerank: bool | None) -> bool:
    if use_rerank is None:
        return config.settings.rerank.enabled
    return use_rerank


def take_sources(rows, rerank_scores=None, variant: str | None = None) -> list[Source]:
    variant = variant or config.settings.corpus.variant
    scores = rerank_scores or [None] * len(rows)
    kept: dict[str, Source] = {}
    for row, rerank_score in zip(rows, scores, strict=True):
        _, src, _, _, vector_rank, keyword_rank, vector_distance, score, *_ = row
        if _hidden_by_cut(src, variant):
            continue
        if src in kept:
            # a duplicated path must not hide the best cross-encoder score from the gate
            best = kept[src].rerank_score
            if rerank_score is not None and (best is None or rerank_score > best):
                kept[src].rerank_score = round(float(rerank_score), 3)
            continue
        kept[src] = _source_from_row(
            src, vector_rank, keyword_rank, vector_distance, score, rerank_score
        )
    return list(kept.values())


# a shim for serving baseline, not a design. baseline was cut before index.md was taken
# out at ingest and it is never re-cut, so the rule is applied where that cut is read.
# It dies the day the default variant stops being a legacy one; nothing else should grow
# here. Note it filters after the search took its k rows, so an answer over baseline can
# come back with fewer than k: true before this branch too, but it was silent
LEGACY_SKIP_NAMES = frozenset({"index.md"})


def _hidden_by_cut(source: str, variant: str) -> bool:
    # a variant present in data_chunks and absent from the config is possible: the two
    # sets are independent. Raising here would do it once per retrieved row, in the middle
    # of an answer, so an unknown cut is treated as the legacy one, which hides more
    policy = config.settings.corpus.policy_or_none(variant)
    if policy is not None and sources.base.hygienic(policy):
        return False
    return Path(source).name in LEGACY_SKIP_NAMES


# returns the depth it searched at along with the rows: the snapshot has to record what
# the search used, and resolving a second time is a second answer, not the same one
def _retrieve_rows(question: str, category, k: int, rerank_enabled: bool, variant: str,
                   ef_search: int | None = None):
    depth = search_depth.resolve(variant, ef_search)
    if not rerank_enabled:
        return (
            db.hybrid_search(
                question, llm.embed(question), category, limit=k, variant=variant,
                ef_search=depth,
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
    )
    ranked = rerank.rerank(question, candidates, top=k)
    return [row for row, _ in ranked], [score for _, score in ranked], depth


def format_chunks(rows, variant: str | None = None) -> str:
    variant = variant or config.settings.corpus.variant
    return "\n\n".join(
        f"[{src}]\n{content}"
        for content, src, *_ in rows
        if not _hidden_by_cut(src, variant)
    )


def _gate_scores(query: str, rows, top: int) -> list:
    import rerank

    head = rows[:top]
    scores = rerank.score_pairs([(query, row[0]) for row in head])
    return [float(s) for s in scores] + [None] * (len(rows) - len(head))


def search_chunks(
    query: str,
    category: str | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
    gate_top: int | None = None,
    *,
    variant: str,
) -> tuple[str, list[Source], int]:
    k = k or config.settings.retrieval.results_limit
    use_rerank = resolve_rerank(use_rerank)
    rows, rerank_scores, depth = _retrieve_rows(query, category, k, use_rerank, variant)
    if not rows:
        return NO_RESULTS, [], depth
    if rerank_scores is None and gate_top:
        rerank_scores = _gate_scores(query, rows, gate_top)
    return (
        format_chunks(rows, variant) or NO_RESULTS,
        take_sources(rows, rerank_scores, variant),
        depth,
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
) -> Answer:
    start = started_at if started_at is not None else time.perf_counter()
    lang = _resolve_language(question, language)
    use_rerank = resolve_rerank(use_rerank)
    k = k or config.settings.retrieval.results_limit

    context = format_chunks(rows, variant) if rows else None
    if not context:
        ans = Answer(text=NO_RESULTS)
    else:
        user = f"{context}\n\nQuestion: {question}"
        if language:
            user += f"\n\n{_language_directive(language)}"
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
        )
    except SQLAlchemyError as e:
        log.error("question_log.insert_failed", reason=str(e))

    return ans


_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _detect_language(text) -> str:
    # same rule as the search config: a wrong guess here answers a Russian question in English
    return db.detect_language(text)


def _resolve_language(question: str, language: str | None) -> str:
    return language or _detect_language(question)


def _language_directive(language: str) -> str:
    return f"Respond in {_LANG_NAMES.get(language, language)}."


def _retrieval_snapshot(rows, sources) -> dict:
    distances = [row[6] for row in rows if row[6] is not None]
    rerank_scores = [s.rerank_score for s in sources if s.rerank_score is not None]
    return {
        "results_count": len(rows),
        "min_distance": round(min(distances), 3) if distances else None,
        "top_rerank_score": max(rerank_scores) if rerank_scores else None,
    }


def _config_snapshot(use_rerank, k, phased, distance_threshold, rerank_device, variant: str,
                     ef_search: int | None = None) -> dict:
    return {
        "rerank": use_rerank,
        "rerank_device": (rerank_device or _rerank_device()) if use_rerank else None,
        "distance_threshold": distance_threshold,
        "k": k,
        "phased": phased,
        "variant": variant,
        "keyword": {
            "query": config.settings.retrieval.keyword_query,
            "rank": config.settings.retrieval.keyword_rank,
            "norm": config.settings.retrieval.keyword_norm,
            "query_lang": config.settings.retrieval.query_lang,
        },
        "ef_search": ef_search,
        # the same tolerance `_hidden_by_cut` was given: a variant present in the table
        # and absent from the config is possible, and raising here kills an answer the
        # generator was already paid for
        "variant_policy": config.settings.corpus.policy_or_none(variant),
        "corpus_fingerprint": db.fingerprint_or_none(variant=variant),
    }


def _rerank_device() -> str | None:
    try:
        import rerank

        return rerank.device()
    except Exception:
        return None


def _log_answer(
    original_text: str, ans: Answer, lang: str, context=None, run_name=None,
    use_rerank=False, k=None, phased=False, rerank_device=None, retrieval=None,
    *, variant: str, ef_search: int | None = None,
) -> None:
    with Session() as session:
        question = _find_or_create_question(session, original_text, lang)
        log_row = QuestionLog(
            run_name=run_name,
            question_id=question.id,
            answered=ans.success,
            answer=ans.text,
            context=context,
            sources=[asdict(s) for s in ans.sources],
            models={
                "generation": ans.metrics.model,
                "embedding": llm.resolve_name("embedding"),
            },
            prompts={
                "generate_answer": prompt_repo.active_version(Purpose.generate_answer)
            },
            metrics={
                "config": _config_snapshot(
                    use_rerank, k, phased, ans.metrics.distance_threshold,
                    rerank_device, variant, ef_search,
                ),
                "retrieval": retrieval,
                # what the ceiling grid is actually gated on, as a number in the row
                # rather than an arithmetic done by hand afterwards: the context is what
                # `k` chunks came to, and the window it has to fit in is fixed
                "context_chars": len(context) if context else 0,
            },
            prompt_tokens=ans.metrics.prompt_tokens,
            completion_tokens=ans.metrics.completion_tokens,
            elapsed=ans.elapsed,
        )
        session.add(log_row)
        session.commit()
        log_id = log_row.id

    if ans.success and run_name is None:
        job_queue.enqueue("judge_answers", {"log_ids": [log_id]})


def _find_or_create_question(session, original_text, lang, set_name="live"):
    stmt = (
        insert(Question)
        .values(
            original_text=original_text,
            set_name=set_name,
            language=lang,
            text_hash=hashlib.sha256(original_text.encode("utf-8")).hexdigest(),
        )
        .on_conflict_do_update(
            index_elements=["text_hash"],
            set_={"original_text": Question.original_text},
        )
        .returning(Question)
    )
    return session.scalar(select(Question).from_statement(stmt))
