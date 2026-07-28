from datetime import datetime
from enum import StrEnum

from orm import Base
from sqlalchemy import DateTime, Enum, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column


class ExperimentStatus(StrEnum):
    draft = "draft"
    running = "running"
    aggregated = "aggregated"
    concluded = "concluded"
    failed = "failed"


_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.draft: {ExperimentStatus.running},
    ExperimentStatus.running: {ExperimentStatus.aggregated, ExperimentStatus.failed},
    ExperimentStatus.aggregated: {ExperimentStatus.concluded},
    ExperimentStatus.failed: {ExperimentStatus.running},
    ExperimentStatus.concluded: set(),
}


def can_advance(src: ExperimentStatus, dst: ExperimentStatus) -> bool:
    return dst in _TRANSITIONS.get(src, set())


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None]
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, native_enum=False), default=ExperimentStatus.draft
    )
    dataset: Mapped[str]
    sample_size: Mapped[int | None]
    sample_seed: Mapped[int | None]
    question_ids: Mapped[list | None] = mapped_column(JSONB)
    data_prep: Mapped[dict] = mapped_column(JSONB, default=dict)
    procedure: Mapped[dict] = mapped_column(JSONB, default=dict)
    param: Mapped[str]
    param_values: Mapped[list] = mapped_column(JSONB, default=list)
    run_names: Mapped[list] = mapped_column(JSONB, default=list)
    results: Mapped[dict | None] = mapped_column(JSONB)
    conclusion: Mapped[str | None]
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    elapsed: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"Experiment(id={self.id!r}, param={self.param!r}, status={self.status!r})"
