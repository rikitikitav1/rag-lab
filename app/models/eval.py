import hashlib
from datetime import datetime
from enum import StrEnum

from orm import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Enum, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.registry import Pipeline


# candidate as generated, accepted once a run may read it, refused with its reason; the model holds the values
class QuestionStatus(StrEnum):
    candidate = "candidate"
    accepted = "accepted"
    refused = "refused"


CANDIDATE, ACCEPTED, REFUSED = QuestionStatus.candidate, QuestionStatus.accepted, QuestionStatus.refused


# the uniqueness key of `questions`: four writers computed it independently
def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(64), unique=True)
    original_text: Mapped[str]
    normalized_text: Mapped[str | None]
    reference_answer: Mapped[str | None]
    marked_sources: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    # {"file", "section", "version"} instead of marks; none_as_null, or a JSON null fails the one-kind check
    gold: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    # the two languages of one fact, and the section's words the generated answer rests on
    pair_id: Mapped[str | None]
    evidence: Mapped[str | None]
    set_name: Mapped[str | None]
    language: Mapped[str | None]
    kind: Mapped[str | None]
    # a set made by hand is accepted as is
    status: Mapped[QuestionStatus] = mapped_column(
        Enum(QuestionStatus, native_enum=False, values_callable=lambda e: [x.value for x in e]),
        default=QuestionStatus.accepted,
        server_default="accepted",
    )
    # why the acceptance refused it or left it for the judge; the status says which
    acceptance_why: Mapped[str | None]
    # the reader's word on a generated question: None until it is asked
    answerable_by_reader: Mapped[bool | None]
    # {identifier: sections of the source that hold it}, for the identifiers of the question its gold section holds
    anchors: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    # {"block", "block_sha", "char"}: the block the generator read and where the evidence starts in it
    evidence_at: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    source_question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    embedded_by: Mapped[str | None]

    def __repr__(self) -> str:
        return f"Question(id={self.id!r}, text={self.original_text[:40]!r})"


class QuestionLog(Base):
    __tablename__ = "question_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_name: Mapped[str | None]
    pipeline: Mapped[str] = mapped_column(default=Pipeline.single_shot.value)
    question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id"))
    # what was asked, on the row: `questions` is a live table and this is a record of one run
    question_text: Mapped[str | None]
    reference_answer: Mapped[str | None]
    answered: Mapped[bool]
    answer: Mapped[str | None]
    context: Mapped[str | None]
    # the same chunks the join above was built from, one per element
    contexts: Mapped[list | None] = mapped_column(JSONB)
    chunks: Mapped[list | None] = mapped_column(JSONB)
    # what a replay needs: the turns, not the tool results `contexts` already holds
    transcript: Mapped[list | None] = mapped_column(JSONB)
    sources: Mapped[list | None] = mapped_column(JSONB)
    models: Mapped[dict] = mapped_column(JSONB, default=dict)
    prompts: Mapped[dict] = mapped_column(JSONB, default=dict)
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    elapsed: Mapped[float | None]
    faithfulness: Mapped[str | None]
    relevance: Mapped[str | None]
    completeness: Mapped[str | None]
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    # false when the run said no judge follows it: the sweep must not judge what nobody asked about
    judge_wanted: Mapped[bool] = mapped_column(server_default=text("true"), default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    question: Mapped["Question | None"] = relationship()

    def __repr__(self) -> str:
        return f"QuestionLog(id={self.id!r}, run_name={self.run_name!r})"
