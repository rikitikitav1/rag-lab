from dataclasses import dataclass, field

import config
import llm
import logging_setup
from models.corpus import DataChunk, DataSource
from orm.sync_db import Session
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy_utils import Ltree
from timing_wrappers import measure_elapsed

log = logging_setup.get_logger(__name__)


@dataclass
class IndexResult:
    sources: int
    chunks: int
    elapsed: float = 0.0
    model: str = field(default_factory=lambda: llm.resolve_name("embedding"))

    def __str__(self) -> str:
        return (
            f"Model: {self.model}, elapsed: {self.elapsed}s, "
            f"sources: {self.sources}, chunks: {self.chunks}"
        )


def _provision_source(session, source) -> DataSource:
    url = getattr(source, "url", None)
    values = {
        "name": source.name,
        "kind": "git" if url else "local",
        "git_url": url,
        "path": getattr(source, "path", None),
    }
    stmt = (
        pg_insert(DataSource)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["name"],
            set_={k: v for k, v in values.items() if k != "name"},
        )
        .returning(DataSource)
    )
    data_source = session.scalar(select(DataSource).from_statement(stmt))
    session.execute(delete(DataChunk).where(DataChunk.source_id == data_source.id))
    session.commit()
    return data_source


def _chunk(source_id, doc) -> DataChunk:
    return DataChunk(
        source_id=source_id,
        source=doc.source,
        content=doc.content,
        category=Ltree(doc.category),
        language=doc.language,
        chunk_index=doc.chunk_index,
    )


def _flush(session, chunks: list[DataChunk], embed_size: int) -> int:
    for i in range(0, len(chunks), embed_size):
        batch = chunks[i : i + embed_size]
        vectors = llm.request_embeddings_batch([c.content for c in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            chunk.embedding = vector
    session.add_all(chunks)
    session.commit()
    return len(chunks)


@measure_elapsed
def collect_data(sources, commit_size=None, embed_size=None) -> IndexResult:
    commit_size = commit_size or config.settings.ingestion.commit_size
    embed_size = embed_size or config.settings.ingestion.batch_size
    log.info("index.start", sources=len(sources))
    total, buffer = 0, []

    with Session() as session:
        for source in sources:
            data_source = _provision_source(session, source)
            for file in source.discover():
                for doc in source.to_documents(file):
                    buffer.append(_chunk(data_source.id, doc))
                    if len(buffer) >= commit_size:
                        total += _flush(session, buffer, embed_size)
                        log.info("index.committed", committed=total)
                        buffer = []
        if buffer:
            total += _flush(session, buffer, embed_size)

    log.info("index.done", chunks=total)
    return IndexResult(sources=len(sources), chunks=total)
