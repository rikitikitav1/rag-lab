import re
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from orm import Base
from sqlalchemy import Enum, ForeignKey, Numeric, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Role(StrEnum):
    generation = "generation"
    embedding = "embedding"
    judging = "judging"
    paraphrasing = "paraphrasing"


# shared by every door that takes a model name; `fullmatch` because `$` matches before a newline
MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*(:[a-zA-Z0-9._-]+)?$")
# the shape check lived on the HTTP door alone, and a job is a second door onto one pull
ALLOWED_REGISTRIES = ("hf.co", "registry.ollama.ai")


MAX_MODEL_NAME = 128


def refuse_unknown_registry(name: str) -> None:
    # length and shape here too: the job door used to get only the half below
    if not name or len(name) > MAX_MODEL_NAME or not MODEL_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid model name: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts) or len(parts) > 3:
        raise ValueError(f"invalid model name: {name!r}")
    if len(parts) == 3 and parts[0].lower() not in ALLOWED_REGISTRIES:
        raise ValueError(f"model registry host not allowed: {parts[0]!r}")


class Purpose(StrEnum):
    generate_answer = "generate.answer"
    judge_faithfulness = "judge.faithfulness"
    judge_relevance = "judge.relevance"
    judge_completeness = "judge.completeness"
    paraphrase_question = "paraphrase.question"
    translate_question = "translate.question"
    question_from_heading = "question.from_heading"
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


class EngineKind(StrEnum):
    ollama = "ollama"
    vllm = "vllm"
    openai_compatible = "openai_compatible"


# what the engine takes from the machine; `remote` is an answer, not a missing value
class Placement(StrEnum):
    gpu = "gpu"
    cpu = "cpu"
    gpu_and_cpu = "gpu+cpu"
    remote = "remote"


class Engine(Base):
    __tablename__ = "engines"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    kind: Mapped[EngineKind] = mapped_column(Enum(EngineKind, native_enum=False))
    # the address and the key live in the environment; a row you can read a key out of leaks
    env_prefix: Mapped[str]
    placement: Mapped[Placement] = mapped_column(Enum(Placement, native_enum=False))
    budget: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), default=None)
    spent: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=0)
    # taken at the door and released at the end: two jobs pass the same check at once otherwise
    reserved: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self) -> str:
        return f"Engine(id={self.id!r}, name={self.name!r}, kind={self.kind!r})"


class Model(Base):
    __tablename__ = "models"
    # one name on two engines is two rows, and the same weights under two quantisations are two too
    __table_args__ = (UniqueConstraint("engine_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    engine_id: Mapped[int] = mapped_column(ForeignKey("engines.id", ondelete="RESTRICT"))
    engine: Mapped[Engine] = relationship()
    # null is not "the same weights": a comparison reads it as unknown and refuses
    weights: Mapped[str | None] = mapped_column(default=None)
    quant: Mapped[str | None] = mapped_column(default=None)
    status: Mapped[Status] = mapped_column(
        Enum(Status, native_enum=False), default=Status.available
    )

    def __repr__(self) -> str:
        return f"Model(id={self.id!r}, name={self.name!r}, engine_id={self.engine_id!r})"


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
