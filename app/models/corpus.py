from datetime import datetime
from enum import StrEnum

from orm import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import LtreeType


# the model is the one place that decides what a value may be; this column was plain text
class Verdict(StrEnum):
    ok = "ok"
    dirty = "dirty"
    broken = "broken"


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    git_url: Mapped[str | None]
    path: Mapped[str | None]
    active: Mapped[bool] = mapped_column(default=True)
    ingest_quality: Mapped[Verdict | None] = mapped_column(
        Enum(Verdict, native_enum=False, values_callable=lambda e: [m.value for m in e])
    )
    ingest_variant: Mapped[str | None]
    ingest_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingest_reports: Mapped[dict] = mapped_column(JSONB, default=dict)
    chunks: Mapped[list["DataChunk"]] = relationship(
        back_populates="data_source", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"DataSource(id={self.id!r}, name={self.name!r}, "
            f"kind={self.kind!r}, git_url={self.git_url!r})"
        )


class DataChunk(Base):
    __tablename__ = "data_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE")
    )
    source: Mapped[str]
    variant: Mapped[str]
    section: Mapped[str | None]
    content_hash: Mapped[str | None]
    prefix_len: Mapped[int | None]
    content: Mapped[str]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    chunk_index: Mapped[int]
    category: Mapped[str] = mapped_column(LtreeType)
    language: Mapped[str]
    content_tsv: Mapped[str | None] = mapped_column(TSVECTOR)
    data_source: Mapped["DataSource"] = relationship(back_populates="chunks")
