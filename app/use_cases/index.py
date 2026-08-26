import hashlib
import re
from dataclasses import dataclass, field

import config
import llm
import logging_setup
from models.corpus import DataChunk, DataSource
from orm.sync_db import Session
from sqlalchemy import delete, select
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy_utils import Ltree
from timing_wrappers import measure_elapsed

log = logging_setup.get_logger(__name__)

# the name reaches DDL as a literal; 22 chars of prefix + 36 + 4 of suffix fit in 63
VARIANT_RE = re.compile(r"^[a-z0-9_]{1,36}$")
# the default 64MB is smaller than the vectors themselves, and pgvector then builds the slow way
MAINTENANCE_WORK_MEM = "512MB"


def check_variant(name: str) -> str:
    if not VARIANT_RE.match(name or ""):
        raise ValueError(f"corpus variant '{name}' must match {VARIANT_RE.pattern}")
    return name


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


def _provision_source(session, source, variant) -> DataSource:
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
    session.execute(
        delete(DataChunk).where(
            DataChunk.source_id == data_source.id, DataChunk.variant == variant
        )
    )
    session.commit()
    return data_source


# whitespace must not decide whether two repositories hold the same answer
def _body_hash(body: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", body).strip().encode()).hexdigest()


def _chunk(source_id, doc, variant) -> DataChunk:
    return DataChunk(
        source_id=source_id,
        source=doc.source,
        variant=variant,
        content=doc.content,
        content_hash=_body_hash(doc.body or doc.content),
        section=doc.section,
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
def collect_data(sources, commit_size=None, embed_size=None, variant=None) -> IndexResult:
    commit_size = commit_size or config.settings.ingestion.commit_size
    embed_size = embed_size or config.settings.ingestion.batch_size
    variant = check_variant(variant or config.settings.corpus.variant)
    config.settings.corpus.policy(variant)
    log.info("index.start", sources=len(sources), variant=variant)
    total, buffer = 0, []

    with Session() as session:
        for source in sources:
            data_source = _provision_source(session, source, variant)
            for file in source.discover():
                for doc in source.to_documents(file):
                    buffer.append(_chunk(data_source.id, doc, variant))
                    if len(buffer) >= commit_size:
                        total += _flush(session, buffer, embed_size)
                        log.info("index.committed", committed=total)
                        buffer = []
        if buffer:
            total += _flush(session, buffer, embed_size)

    _ensure_vector_index(variant)
    log.info("index.done", chunks=total, variant=variant)
    return IndexResult(sources=len(sources), chunks=total)


# built after the bulk insert: hnsw over finished data is cheaper than maintained row by row
def _ensure_vector_index(variant: str) -> None:
    name = f"data_chunks_embedding_{check_variant(variant)}_idx"
    with Session() as session:
        # the session carries statement_timeout=30s, and an hnsw build outlives it
        session.execute(sa_text("SET LOCAL statement_timeout = 0"))
        session.execute(sa_text(f"SET LOCAL maintenance_work_mem = '{MAINTENANCE_WORK_MEM}'"))
        session.execute(
            sa_text(
                f"CREATE INDEX IF NOT EXISTS {name} ON data_chunks "
                f"USING hnsw (embedding vector_cosine_ops) WHERE variant = '{variant}'"
            )
        )
        session.commit()
    log.info("index.vector_index_ready", name=name, work_mem=MAINTENANCE_WORK_MEM)
