from datetime import datetime
from enum import StrEnum

from orm import Base
from sqlalchemy import Enum, Index, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column


class JobStatus(StrEnum):
    new = "new"
    running = "running"
    done = "done"
    error = "error"
    paused = "paused"
    cancelled = "cancelled"


# todo: allowed jobs list take from worker handlers


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("idx_jobs_queue_status_apply_since", "queue", "status", "apply_since"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str]
    queue: Mapped[str] = mapped_column(default="default")
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), default=JobStatus.new
    )
    options: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[dict | None] = mapped_column(JSONB)
    elapsed: Mapped[float | None]
    apply_since: Mapped[datetime] = mapped_column(server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"Job(id={self.id!r}, type={self.type!r}, status={self.status!r})"
