import config
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
    use_cases.index.collect_data(list(sources.factory.all_sources()))


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
