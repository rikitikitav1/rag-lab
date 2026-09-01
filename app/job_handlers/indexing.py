import config
import job_queue
import llm
import logging_setup
from models.eval import Question
from orm.sync_db import Session
from sqlalchemy import select

from .base import register, require_embedder_ready

log = logging_setup.get_logger(__name__)


@register("index_data")
def index_data(options: dict) -> None:
    import sources.factory
    import use_cases.index

    require_embedder_ready()
    built = list(sources.factory.all_sources())
    # written by the bootstrap and read by nobody: a job for one source re-indexed all 177
    wanted = options.get("source") or "all"
    if wanted != "all":
        built = [s for s in built if s.name == wanted]
        if not built:
            known = sorted(s.name for s in sources.factory.all_sources())
            raise ValueError(f"no such source: {wanted!r}; known: {known}")
    # resolved once: the call below took it bare and requeued itself with an unmatchable null
    variant = options.get("variant") or config.settings.corpus.variant
    use_cases.index.collect_data(built, variant=variant, build_index=False)
    # the report reads rows, not the index, so a failing build must not take it down
    for source in built:
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


@register("build_vector_index")
def build_vector_index(options: dict) -> None:
    import use_cases.index

    use_cases.index.ensure_vector_index(
        options.get("variant") or config.settings.corpus.variant
    )
    _report_depth()


# indexing moves the depth, and a person runs the preflight, so it is read here
def _report_depth() -> None:
    from orm.sync_db import engine
    from sqlalchemy import text
    from use_cases import search_depth

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
    size = config.settings.ingestion.batch_size
    with Session() as session:
        pending = session.scalars(
            select(Question).where(Question.embedding.is_(None))
        ).all()
        for i in range(0, len(pending), size):
            batch = pending[i : i + size]
            vectors = llm.request_embeddings_batch([q.original_text for q in batch])
            for question, vector in zip(batch, vectors, strict=True):
                question.embedding = vector
            session.commit()
    log.info("worker.embed_questions", embedded=len(pending))
