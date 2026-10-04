from dataclasses import dataclass, field

import config
import llm
import logging_setup
from corpus_keys import body_hash, check_variant, vector_index_name
from errors import StandFault
from models.corpus import DataChunk, DataSource, Stage
from orm.sync_db import Session
from sources import files
from sqlalchemy import cast as sa_cast
from sqlalchemy import delete, select, update
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from timing_wrappers import measure_elapsed

log = logging_setup.get_logger(__name__)

# the default 64MB is smaller than the vectors themselves, and pgvector then builds the slow way
MAINTENANCE_WORK_MEM = "512MB"


@dataclass
class IndexResult:
    sources: int
    chunks: int
    elapsed: float = 0.0
    # rows a source file names that were not cut: declared or raw ones wait for the accept door, empty ones for a folder
    refused: dict[str, str] = field(default_factory=dict)
    # the sources a cancel left uncut, in the order they would have come
    left: list[str] = field(default_factory=list)
    # {source: {the more trusted source that keeps the text: chunks}}, the copies this cut took out
    lower_copies_dropped: dict[str, dict[str, int]] = field(default_factory=dict)
    model: str = field(default_factory=lambda: llm.resolve_name("embedding"))

    def __str__(self) -> str:
        return f"Model: {self.model}, elapsed: {self.elapsed}s, sources: {self.sources}, chunks: {self.chunks}"


def _provision_source(session, source, variant) -> DataSource:
    values = files.row_of(source.settings, source.name)
    insert = pg_insert(DataSource).values(**values)
    stmt = insert.on_conflict_do_update(
        index_elements=["name"], set_=files.upserted(insert, DataSource.__table__, list(values))
    ).returning(DataSource)
    data_source = session.scalar(select(DataSource).from_statement(stmt))
    session.commit()
    return data_source


# a question with no vector, or one another embedder wrote: the bootstrap and the job ask the same
def question_needs_embedding(label: str | None):
    from models.eval import Question

    missing = Question.embedding.is_(None)
    return missing if label is None else missing | Question.embedded_by.is_distinct_from(label)


# on its own it left the source empty for as long as the embeddings took
def _replace_chunks(session, source_id: int, variant: str, chunks: list, embed_size: int) -> int:
    for i in range(0, len(chunks), embed_size):
        batch = chunks[i : i + embed_size]
        label, vectors = llm.embed_labelled([c.content for c in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            chunk.embedding = vector
            chunk.embedded_by = label
    session.execute(delete(DataChunk).where(DataChunk.source_id == source_id, DataChunk.variant == variant))
    session.add_all(chunks)
    session.commit()
    return len(chunks)


# whitespace must not decide whether two repositories hold the same answer
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
        content_hash=body_hash(doc.body or doc.content),
        versions=list(doc.versions),
        section=doc.section,
        prefix_len=_prefix_len(doc),
        category=doc.category,
        tags=list(doc.tags),
        language=doc.language,
        chunk_index=doc.chunk_index,
    )


@measure_elapsed
def collect_data(sources, embed_size=None, variant=None, build_index=True, stop=None) -> IndexResult:
    embed_size = embed_size or config.settings.ingestion.batch_size
    variant = check_variant(variant or config.settings.corpus.variant)
    policy = config.settings.corpus.policy(variant)
    log.info("index.start", sources=len(sources), variant=variant)
    total = 0

    with Session() as session:
        refused, left = {}, []
        for n, source in enumerate(sources):
            # a cancel is read between sources: one source is replaced whole or not at all
            if stop is not None and stop():
                left = [s.name for s in sources[n:]]
                log.warning("index.cancelled", done=n, left=len(left))
                break
            data_source = _provision_source(session, source, variant)
            if data_source.stage != Stage.accepted:
                log.warning("index.refused_stage", source=source.name, stage=data_source.stage)
                refused[source.name] = f"stage {data_source.stage}, not accepted"
                continue
            # the whole source at once, and the cut digest reads the same method
            buffer = [_chunk(data_source.id, doc, variant) for doc in source.documents(policy)]
            # a folder this host lacks, or one of PDFs where the markdown should be, must not empty a source in silence
            if not buffer:
                log.error("index.refused_empty", source=source.name, root=str(source.root))
                refused[source.name] = f"no documents under {source.root}"
                continue
            total += _replace_chunks(session, data_source.id, variant, buffer, embed_size)
            # the digest of the rules this cut read, merged in the base so two variants cut at once keep both
            session.execute(
                update(DataSource)
                .where(DataSource.id == data_source.id)
                .values(
                    indexed_with=DataSource.indexed_with.op("||")(
                        sa_cast({variant: files.digest(source.settings)}, JSONB)
                    ),
                    indexed_rules=DataSource.indexed_rules.op("||")(
                        sa_cast({variant: files.cut_rules(source.settings)}, JSONB)
                    ),
                )
            )
            session.commit()
            # texts merged across versions: the smoke of a second version reads it against the preregistered ceiling
            merged = getattr(source, "merged", None)
            log.info("index.committed", source=source.name, chunks=len(buffer), total=total, merged=merged)

    cut = [s.name for s in sources if s.name not in refused and s.name not in left]
    dropped = _drop_lower_copies(variant, cut)
    if build_index and not left:
        ensure_vector_index(variant)
    log.info("index.done", chunks=total, variant=variant)
    return IndexResult(sources=len(sources) - len(refused) - len(left), chunks=total, refused=refused, left=left,
                       lower_copies_dropped=dropped)


# every cut, the job's and the CLI's, keeps one copy of a shared text; a failure here must not cost the embedding
def _drop_lower_copies(variant: str, names: list[str]) -> dict:
    from use_cases import dedup

    try:
        return dedup.drop_lower_copies(variant, names) if names else {}
    except StandFault:
        raise
    except Exception as e:
        log.error("index.dedup_failed", variant=variant, error=str(e))
        return {}


# the one owner of the name, so the three readers ask here
def has_vector_index(variant: str) -> bool:
    with Session() as session:
        return bool(
            session.scalar(sa_text("SELECT to_regclass(:name) IS NOT NULL").bindparams(name=vector_index_name(variant)))
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
