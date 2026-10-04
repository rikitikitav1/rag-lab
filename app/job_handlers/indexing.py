import config
import job_queue
import llm
import logging_setup
from models.eval import Question
from orm.sync_db import Session
from sqlalchemy import select
from use_cases.index import question_needs_embedding

from .base import Final, register, require_embedder_ready
from .card import clear_the_engine_for

log = logging_setup.get_logger(__name__)


@register("index_data")
def index_data(options: dict) -> dict:
    import sources.factory
    import use_cases.index

    require_embedder_ready()
    clear_the_engine_for("embedding")
    # a job for one source builds that one alone: the rest are not cloned, read or cut
    wanted = options.get("source") or "all"
    built = list(sources.factory.sources(None if wanted == "all" else [wanted]))
    if wanted != "all" and not built:
        raise Final(f"no accepted source named {wanted!r}; only accepted rows are indexed")
    # resolved once: the call below took it bare and requeued itself with an unmatchable null
    variant = options.get("variant") or config.settings.corpus.variant
    job_id = options.get("_job_id")
    stop = (lambda: job_queue.is_cancelled(job_id)) if job_id is not None else None
    result = use_cases.index.collect_data(built, variant=variant, build_index=False, stop=stop)
    if wanted != "all" and result.refused:
        raise Final(f"{wanted} was not cut: {result.refused[wanted]}")
    # the report reads rows, not the index; a refused source has none of this cut, nor has one a cancel left uncut
    for source in (s for s in built if s.name not in result.refused and s.name not in result.left):
        job_queue.enqueue(
            "analyze_source",
            {"source": source.name, "variant": variant, "mode": "indexed"},
        )
    # a failing index build must not cost three retries of re-embedding 13k chunks
    try:
        use_cases.index.ensure_vector_index(variant)
        _report_depth()
    except Exception as e:
        log.error("index.vector_index_failed", variant=variant, error=str(e))
        # the same dedup bootstrap does: three retries would queue three builds on one lane
        if not job_queue.pending_of_type("build_vector_index", variant=variant):
            job_queue.enqueue("build_vector_index", {"variant": variant})
    # a full reindex goes on past a refused source; the refusals stay on the job's row, not only in the log
    cancelled = {"left_by_cancel": result.left} if result.left else {}
    copies = {"lower_copies_dropped": result.lower_copies_dropped} if result.lower_copies_dropped else {}
    return {"sources": len(built) - len(result.left), "refused": result.refused, **cancelled, **copies,
            "phases": result.phases}


@register("build_vector_index")
def build_vector_index(options: dict) -> None:
    import use_cases.index

    use_cases.index.ensure_vector_index(options.get("variant") or config.settings.corpus.variant)
    _report_depth()


# indexing moves the depth, and a person runs the preflight, so it is read here
def _report_depth() -> None:
    import search_depth
    from orm.sync_db import engine
    from sqlalchemy import text

    # the plan and reltuples move on ANALYZE: right answer to a stale question otherwise
    with engine.connect() as conn:
        conn.execute(text("ANALYZE data_chunks"))
        conn.commit()
    search_depth.forget()
    for row in search_depth.audit():
        if row["serving_uses_index"]:
            log.info("depth.after_index", **row)
        else:
            log.error("depth.serves_without_the_index", **row)


@register("analyze_source")
def analyze_source(options: dict) -> None:
    import use_cases.ingest_quality as ingest_quality

    ingest_quality.analyze(
        options["source"],
        variant=options.get("variant") or config.settings.corpus.variant,
        mode=options.get("mode", "indexed"),
    )


@register("embed_questions")
def embed_questions(options: dict) -> None:
    require_embedder_ready()
    clear_the_engine_for("embedding")
    size = config.settings.ingestion.batch_size
    label = llm.embedder_label()
    with Session() as session:
        # a vector from another embedder is as missing as none: search would refuse it anyway
        pending = session.scalars(select(Question).where(question_needs_embedding(label))).all()
        for i in range(0, len(pending), size):
            batch = pending[i : i + size]
            # the label of the embedder that made these vectors, even if the role moved mid-job
            made_by, vectors = llm.embed_labelled([q.original_text for q in batch])
            for question, vector in zip(batch, vectors, strict=True):
                question.embedding = vector
                question.embedded_by = made_by
            session.commit()
    log.info("worker.embed_questions", embedded=len(pending))
