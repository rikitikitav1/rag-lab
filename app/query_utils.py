from sqlalchemy.sql import Select


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
