import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
import job_queue
import llm
import logging_setup
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

NO_RESULTS = "No relevant documents found."


@dataclass
class Source:
    source: str
    vector_rank: float | None
    keyword_rank: float | None
    vector_distance: float | None
    score: float

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


def _source_from_row(src, vector_rank, keyword_rank, vector_distance, score) -> Source:
    return Source(
        source=src,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        vector_distance=round(vector_distance, 3)
        if vector_distance is not None
        else None,
        score=round(float(score), 3),
    )


def is_ignored_source(source) -> bool:
    return Path(source).name in IGNORED_SOURCES


def take_sources(rows) -> list[Source]:
    seen: set[str] = set()
    sources: list[Source] = []
    for _, src, _, _, vector_rank, keyword_rank, vector_distance, score in rows:
        if src in seen or is_ignored_source(src):
            continue
        seen.add(src)
        sources.append(
            _source_from_row(src, vector_rank, keyword_rank, vector_distance, score)
        )
    return sources


def _retrieve_rows(question: str, category, k: int, rerank_enabled: bool):
    if not rerank_enabled:
        return db.hybrid_search(question, llm.embed(question), category, limit=k)

    import rerank

    candidates = db.hybrid_search(
        question, llm.embed(question), category, limit=config.settings.rerank.candidates
    )
    return rerank.rerank(question, candidates, top=k)


def format_chunks(rows) -> str:
    return "\n\n".join(
        f"[{src}]\n{content}" for content, src, *_ in rows if not is_ignored_source(src)
    )


def search_chunks(
    query: str,
    category: str | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
) -> tuple[str, list[Source]]:
    k = k or config.settings.retrieval.results_limit
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    rows = _retrieve_rows(query, category, k, use_rerank)
    if not rows:
        return NO_RESULTS, []
    return format_chunks(rows) or NO_RESULTS, take_sources(rows)


@measure_elapsed
def retrieve(
    question: str,
    category: str | None = None,
    k: int | None = None,
) -> Retrieval:
    k = k or config.settings.retrieval.results_limit
    rows = _retrieve_rows(question, category, k, config.settings.rerank.enabled)
    return Retrieval(sources=take_sources(rows))


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
    rows = _retrieve_rows(question, category, k, use_rerank)
    return answer_from_rows(
        question,
        rows,
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
    add_context=False,
    run_name: str | None = None,
    use_rerank: bool | None = None,
    language: str | None = None,
    model: str | None = None,
    k: int | None = None,
    started_at: float | None = None,
    phased: bool = False,
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
            sources=take_sources(rows),
            metrics=metrics,
        )
        if add_context:
            ans.context = context

    ans.elapsed = round(time.perf_counter() - start, 3)

    try:
        _log_answer(question, ans, lang, context, run_name, use_rerank, k, phased)
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


def _log_answer(
    original_text: str, ans: Answer, lang: str, context=None, run_name=None,
    use_rerank=False, k=None, phased=False,
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
                "config": {
                    "rerank": use_rerank,
                    "distance_threshold": ans.metrics.distance_threshold,
                    "k": k,
                    "phased": phased,
                }
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
