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
from langdetect import LangDetectException, detect
from models.eval import Question, QuestionLog
from models.registry import Purpose
from orm.sync_db import Session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from timing_wrappers import measure_elapsed

import db

log = logging_setup.get_logger(__name__)

IGNORED_SOURCES = config.settings.ignored_sources

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


def is_ignored_source(source) -> bool:
    return Path(source).name in IGNORED_SOURCES


def take_sources(rows, rerank_scores=None) -> list[Source]:
    scores = rerank_scores or [None] * len(rows)
    kept: dict[str, Source] = {}
    for row, rerank_score in zip(rows, scores, strict=True):
        _, src, _, _, vector_rank, keyword_rank, vector_distance, score = row
        if is_ignored_source(src):
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


def _retrieve_rows(question: str, category, k: int, rerank_enabled: bool):
    if not rerank_enabled:
        return db.hybrid_search(question, llm.embed(question), category, limit=k), None

    import rerank

    candidates = db.hybrid_search(
        question, llm.embed(question), category, limit=config.settings.rerank.candidates
    )
    ranked = rerank.rerank(question, candidates, top=k)
    return [row for row, _ in ranked], [score for _, score in ranked]


def format_chunks(rows) -> str:
    return "\n\n".join(
        f"[{src}]\n{content}" for content, src, *_ in rows if not is_ignored_source(src)
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
) -> tuple[str, list[Source]]:
    k = k or config.settings.retrieval.results_limit
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    rows, rerank_scores = _retrieve_rows(query, category, k, use_rerank)
    if not rows:
        return NO_RESULTS, []
    if rerank_scores is None and gate_top:
        rerank_scores = _gate_scores(query, rows, gate_top)
    return format_chunks(rows) or NO_RESULTS, take_sources(rows, rerank_scores)


@measure_elapsed
def retrieve(
    question: str,
    category: str | None = None,
    k: int | None = None,
) -> Retrieval:
    k = k or config.settings.retrieval.results_limit
    rows, rerank_scores = _retrieve_rows(question, category, k, config.settings.rerank.enabled)
    return Retrieval(sources=take_sources(rows, rerank_scores))


def answer(
    question: str,
    category: str | None = None,
    k: int | None = None,
    add_context=False,
    run_name: str | None = None,
    use_rerank: bool | None = None,
    language: str | None = None,
    model: str | None = None,
) -> Answer:
    start = time.perf_counter()
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    k = k or config.settings.retrieval.results_limit
    rows, rerank_scores = _retrieve_rows(question, category, k, use_rerank)
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
) -> Answer:
    start = started_at if started_at is not None else time.perf_counter()
    lang = _resolve_language(question, language)
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    k = k or config.settings.retrieval.results_limit

    context = format_chunks(rows) if rows else None
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
            sources=take_sources(rows, rerank_scores),
            metrics=metrics,
        )
        if add_context:
            ans.context = context

    ans.elapsed = round(time.perf_counter() - start, 3)

    try:
        _log_answer(
            question, ans, lang, context, run_name, use_rerank, k, phased, rerank_device,
            _retrieval_snapshot(rows, ans.sources),
        )
    except SQLAlchemyError as e:
        log.error("question_log.insert_failed", reason=str(e))

    return ans


_LANG_NAMES = {"ru": "Russian", "en": "English"}


def _detect_language(text) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


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


def _config_snapshot(use_rerank, k, phased, distance_threshold, rerank_device=None) -> dict:
    return {
        "rerank": use_rerank,
        "rerank_device": (rerank_device or _rerank_device()) if use_rerank else None,
        "distance_threshold": distance_threshold,
        "k": k,
        "phased": phased,
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
                    use_rerank, k, phased, ans.metrics.distance_threshold, rerank_device
                ),
                "retrieval": retrieval,
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
