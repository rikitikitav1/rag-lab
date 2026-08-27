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
    # resolved once for this job: the call below used to take it bare and requeue itself
    # with a null that nothing could match
    variant = options.get("variant") or config.settings.corpus.variant
    use_cases.index.collect_data(built, variant=variant, build_index=False)
    # queued before the index is built: the report reads rows, not the index, and a
    # failing build must not take the whole coverage report down with it
    for source in built:
        job_queue.enqueue(
            "analyze_source",
            {"source": source.name, "variant": variant, "mode": "indexed"},
        )
    # the rows are in and the coverage jobs are queued; a failing index build must not
    # cost three retries of re-embedding 13k chunks. It is loud and it is separate work
    try:
        use_cases.index.ensure_vector_index(variant)
    except Exception as e:
        log.error("index.vector_index_failed", variant=variant, error=str(e))
        # the same dedup bootstrap does: three retries of index_data would otherwise
        # leave three builds of one index waiting on the single-threaded lane
        if not job_queue.pending_of_type("build_vector_index", variant=variant):
            job_queue.enqueue("build_vector_index", {"variant": variant})


@register("build_vector_index")
def build_vector_index(options: dict) -> None:
    import use_cases.index

    use_cases.index.ensure_vector_index(
        options.get("variant") or config.settings.corpus.variant
    )


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
