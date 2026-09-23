from datetime import datetime

from orm import Base
from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column


# what a run promised before it produced a row: the form the stand can refuse against
class Preregistration(Base):
    __tablename__ = "preregistrations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    population: Mapped[dict] = mapped_column(JSONB, default=dict)
    arms: Mapped[dict] = mapped_column(JSONB, default=dict)
    # every column name here is checked against `evals.columns` before the row is written
    closing: Mapped[dict] = mapped_column(JSONB, default=dict)
    guards: Mapped[list] = mapped_column(JSONB, default=list)
    # floor, veto, stop rules, price, expectations: kept verbatim, not parsed
    declared: Mapped[dict] = mapped_column(JSONB, default=dict)
    closed_with: Mapped[dict | None] = mapped_column(JSONB, default=None)

    def __repr__(self) -> str:
        return f"Preregistration(name={self.name!r})"
