from datetime import datetime

from orm import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(64), unique=True)
    original_text: Mapped[str]
    normalized_text: Mapped[str | None]
    reference_answer: Mapped[str | None]
    marked_sources: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    set_name: Mapped[str | None]
    language: Mapped[str | None]
    kind: Mapped[str | None]
    status: Mapped[str | None]
    source_question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))

    def __repr__(self) -> str:
        return f"Question(id={self.id!r}, text={self.original_text[:40]!r})"


class QuestionLog(Base):
    __tablename__ = "question_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_name: Mapped[str | None]
    question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id"))
    answered: Mapped[bool]
    answer: Mapped[str | None]
    context: Mapped[str | None]
    sources: Mapped[list | None] = mapped_column(JSONB)
    models: Mapped[dict] = mapped_column(JSONB, default=dict)
    prompts: Mapped[dict] = mapped_column(JSONB, default=dict)
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    elapsed: Mapped[float | None]
    faithfulness: Mapped[str | None]
    relevance: Mapped[str | None]
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    question: Mapped["Question | None"] = relationship()

    def __repr__(self) -> str:
        return f"QuestionLog(id={self.id!r}, run_name={self.run_name!r})"
