from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.sql import Select


class Page(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str | None = None
    sort_order: Literal["asc", "desc"] = "desc"


def apply_sort_limit_offset(
    stmt: Select,
    sort_map: dict,
    sort_by: str | None,
    sort_order: str = "asc",
    limit: int = 100,
    offset: int = 0,
    default_sort: str = "id",
):
    sort_column = sort_map.get(sort_by, sort_map[default_sort])

    if sort_order == "desc":
        stmt = stmt.order_by(sort_column.desc())
    else:
        stmt = stmt.order_by(sort_column.asc())

    return stmt.limit(limit).offset(offset)
