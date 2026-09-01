from typing import Literal

from fastapi import HTTPException
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


# every listing wrote this loop out again, and `question_log` wrote one `if` per field
def apply_in_filters(stmt: Select, filters: dict) -> Select:
    for column, values in filters.items():
        if values is not None:
            stmt = stmt.where(column.in_(values))
    return stmt


# a range given backwards can never match, and four listings answered it with an empty page
def refuse_backwards_range(low, high, low_name: str, high_name: str) -> None:
    if low is not None and high is not None and low > high:
        raise HTTPException(
            status_code=400, detail=f"{low_name} must be earlier than {high_name}"
        )


def apply_created_between(stmt: Select, column, created_from, created_to) -> Select:
    refuse_backwards_range(created_from, created_to, "created_from", "created_to")
    if created_from is not None:
        stmt = stmt.where(column >= created_from)
    if created_to is not None:
        stmt = stmt.where(column <= created_to)
    return stmt
