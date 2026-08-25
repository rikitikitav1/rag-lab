from datetime import datetime
from enum import StrEnum

from orm import Base
from sqlalchemy import Enum, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Role(StrEnum):
    generation = "generation"
    embedding = "embedding"
    judging = "judging"
    paraphrasing = "paraphrasing"


class Purpose(StrEnum):
    generate_answer = "generate.answer"
    judge_faithfulness = "judge.faithfulness"
    judge_relevance = "judge.relevance"
    judge_completeness = "judge.completeness"
    paraphrase_question = "paraphrase.question"
    translate_question = "translate.question"
    agent_system = "agent.system"
    agent_fallback = "agent.fallback"
    agent_tool_match = "agent.tool_match"
    agent_no_evidence = "agent.no_evidence"


class Status(StrEnum):
    available = "available"
    loading = "loading"
    ready = "ready"


class Pipeline(StrEnum):
    single_shot = "single_shot"
    agent = "agent"


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    status: Mapped[Status] = mapped_column(
        Enum(Status, native_enum=False), default=Status.available
    )

    def __repr__(self) -> str:
        return f"Model(id={self.id!r}, name={self.name!r}, status={self.status!r})"


class ModelRole(Base):
    __tablename__ = "model_roles"

    role: Mapped[Role] = mapped_column(Enum(Role, native_enum=False), primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id", ondelete="RESTRICT"))
    model: Mapped[Model] = relationship()

    def __repr__(self) -> str:
        return f"ModelRole(role={self.role!r}, model_id={self.model_id!r})"


class Prompt(Base):
    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("purpose", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    purpose: Mapped[Purpose] = mapped_column(
        Enum(Purpose, native_enum=False, values_callable=lambda e: [m.value for m in e])
    )
    version: Mapped[int]
    template: Mapped[str]
    active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self) -> str:
        return f"Prompt(purpose={self.purpose!r}, version={self.version!r}, active={self.active!r})"
