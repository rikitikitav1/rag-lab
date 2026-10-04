from datetime import datetime
from enum import StrEnum

from orm import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship


# the model is the one place that decides what a value may be; this column was plain text
class Verdict(StrEnum):
    ok = "ok"
    dirty = "dirty"
    broken = "broken"


# where an added source stands: declared by hand, converted to a raw folder, or accepted for indexing
class Stage(StrEnum):
    declared = "declared"
    raw = "raw"
    accepted = "accepted"


# how far a copy of a text is trusted when two sources hold it word for word, highest first
class Trust(StrEnum):
    official = "official"
    book = "book"
    notes = "notes"


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    kind: Mapped[str | None] = mapped_column(String(32))
    git_url: Mapped[str | None]
    path: Mapped[str | None]
    active: Mapped[bool] = mapped_column(default=True)
    ingest_quality: Mapped[Verdict | None] = mapped_column(
        Enum(Verdict, native_enum=False, values_callable=lambda e: [m.value for m in e])
    )
    ingest_variant: Mapped[str | None]
    ingest_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingest_reports: Mapped[dict] = mapped_column(JSONB, default=dict)
    # declared by hand, converted to a raw folder, or accepted for indexing
    stage: Mapped[Stage] = mapped_column(
        Enum(Stage, native_enum=False, values_callable=lambda e: [x.value for x in e]),
        default=Stage.declared,
        server_default="declared",
    )
    language: Mapped[str | None]
    licence: Mapped[str | None]
    origin: Mapped[dict | None] = mapped_column(JSONB)
    # the whole declaration, reader and rules included, as a source file or the seed wrote it
    declaration: Mapped[dict | None] = mapped_column(JSONB)
    # written by the seed from a source file, which speaks for the row
    seeded: Mapped[bool] = mapped_column(default=False, server_default="false")
    raw: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    # per variant, the digest of the source file the chunks were cut by
    indexed_with: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    # per variant, the rules that digest was taken over, so a moved digest names its fields
    indexed_rules: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    chunks: Mapped[list["DataChunk"]] = relationship(back_populates="data_source", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"DataSource(id={self.id!r}, name={self.name!r}, kind={self.kind!r}, git_url={self.git_url!r})"


class DataChunk(Base):
    __tablename__ = "data_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id", ondelete="CASCADE"))
    source: Mapped[str]
    variant: Mapped[str]
    section: Mapped[str | None]
    content_hash: Mapped[str | None]
    prefix_len: Mapped[int | None]
    content: Mapped[str]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    embedded_by: Mapped[str | None]
    chunk_index: Mapped[int]
    category: Mapped[str | None]
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default="{}")
    versions: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default="{}")
    language: Mapped[str]
    content_tsv: Mapped[str | None] = mapped_column(TSVECTOR)
    data_source: Mapped["DataSource"] = relationship(back_populates="chunks")
