from datetime import datetime
from enum import StrEnum

from orm import Base
from sqlalchemy import DateTime, Enum, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column


# same dataset and state machine, and the unit of a run is not: ranks, in minutes
class ExperimentKind(StrEnum):
    generation = "generation"
    retrieval = "retrieval"
    # the answers are held still and the judge moves: the arms are copies of one run
    rejudge = "rejudge"


# every key a report answers a reader with, written by all three kinds whether they hold it
READING_KEYS = ("source_run", "pairing", "multiplicity", "ranking", "arms", "deltas")


class ExperimentStatus(StrEnum):
    draft = "draft"
    running = "running"
    aggregated = "aggregated"
    concluded = "concluded"
    failed = "failed"


_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.draft: {ExperimentStatus.running},
    ExperimentStatus.running: {ExperimentStatus.aggregated, ExperimentStatus.failed},
    # a rejudge may gain an arm after its report is read: a copy and a judge job, not a rerun
    ExperimentStatus.aggregated: {ExperimentStatus.concluded, ExperimentStatus.running},
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
    kind: Mapped[ExperimentKind] = mapped_column(
        Enum(ExperimentKind, native_enum=False), default=ExperimentKind.generation
    )
    dataset: Mapped[str]
    sample_size: Mapped[int | None]
    sample_seed: Mapped[int | None]
    question_ids: Mapped[list | None] = mapped_column(JSONB)
    data_prep: Mapped[dict] = mapped_column(JSONB, default=dict)
    procedure: Mapped[dict] = mapped_column(JSONB, default=dict)
    param: Mapped[str]
    param_values: Mapped[list] = mapped_column(JSONB, default=list)
    # param has to be a key here, or the headline disagrees with the grid
    axes: Mapped[dict] = mapped_column(JSONB, default=dict)
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
