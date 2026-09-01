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


# fullmatch: `$` admits a trailing newline, and the name reaches DDL twice
def check_variant(name: str) -> str:
    if not VARIANT_RE.fullmatch(name or ""):
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
    session.commit()
    return data_source


# on its own it left the source empty for as long as the embeddings took
def _replace_chunks(session, source_id: int, variant: str, chunks: list, embed_size: int) -> int:
    for i in range(0, len(chunks), embed_size):
        batch = chunks[i : i + embed_size]
        for chunk, vector in zip(
            batch, llm.request_embeddings_batch([c.content for c in batch]), strict=True
        ):
            chunk.embedding = vector
    session.execute(
        delete(DataChunk).where(
            DataChunk.source_id == source_id, DataChunk.variant == variant
        )
    )
    session.add_all(chunks)
    session.commit()
    return len(chunks)


# whitespace must not decide whether two repositories hold the same answer
def _body_hash(body: str) -> str:
    normalised = re.sub(r"\s+", " ", body).strip().encode()
    # a content fingerprint for deduplication, never a credential
    return hashlib.md5(normalised, usedforsecurity=False).hexdigest()


def _prefix_len(doc) -> int | None:
    # only when the body really is the tail: a guessed length hands the metrics nothing real
    if doc.body is None or not doc.content.endswith(doc.body):
        return None
    return len(doc.content) - len(doc.body)


def _chunk(source_id, doc, variant) -> DataChunk:
    return DataChunk(
        source_id=source_id,
        source=doc.source,
        variant=variant,
        content=doc.content,
        content_hash=_body_hash(doc.body or doc.content),
        section=doc.section,
        prefix_len=_prefix_len(doc),
        category=Ltree(doc.category),
        language=doc.language,
        chunk_index=doc.chunk_index,
    )


@measure_elapsed
def collect_data(sources, embed_size=None, variant=None, build_index=True) -> IndexResult:
    embed_size = embed_size or config.settings.ingestion.batch_size
    variant = check_variant(variant or config.settings.corpus.variant)
    policy = config.settings.corpus.policy(variant)
    log.info("index.start", sources=len(sources), variant=variant)
    total = 0

    with Session() as session:
        for source in sources:
            data_source = _provision_source(session, source, variant)
            # the whole source at once, and the cut digest reads the same method
            buffer = [_chunk(data_source.id, doc, variant) for doc in source.documents(policy)]
            total += _replace_chunks(session, data_source.id, variant, buffer, embed_size)
            log.info("index.committed", source=source.name, chunks=len(buffer), total=total)

    if build_index:
        ensure_vector_index(variant)
    log.info("index.done", chunks=total, variant=variant)
    return IndexResult(sources=len(sources), chunks=total)


# the one owner of the name, so the three readers ask here
VECTOR_INDEX_PREFIX = "data_chunks_embedding_"


def vector_index_name(variant: str) -> str:
    return f"{VECTOR_INDEX_PREFIX}{check_variant(variant)}_idx"


def has_vector_index(variant: str) -> bool:
    with Session() as session:
        return bool(
            session.scalar(
                sa_text("SELECT to_regclass(:name) IS NOT NULL").bindparams(
                    name=vector_index_name(variant)
                )
            )
        )


# built after the bulk insert: hnsw over finished data is cheaper than maintained row by row
def ensure_vector_index(variant: str) -> None:
    name = vector_index_name(variant)
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
