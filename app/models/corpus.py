from orm import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import LtreeType


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    git_url: Mapped[str | None]
    path: Mapped[str | None]
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
    content: Mapped[str]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    chunk_index: Mapped[int]
    category: Mapped[str] = mapped_column(LtreeType)
    language: Mapped[str]
    content_tsv: Mapped[str | None] = mapped_column(TSVECTOR)
    data_source: Mapped["DataSource"] = relationship(back_populates="chunks")
