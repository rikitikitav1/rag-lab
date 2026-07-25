from models.jobs import Job
from query_utils import apply_sort_limit_offset
from sqlalchemy import select


def test_sort_desc_with_limit_offset():
    stmt = apply_sort_limit_offset(
        select(Job),
        {"id": Job.id, "created_at": Job.created_at},
        sort_by="created_at",
        sort_order="desc",
        limit=10,
        offset=5,
    )
    sql = str(stmt)
    assert "created_at DESC" in sql
    assert "LIMIT" in sql
    assert "OFFSET" in sql


def test_unknown_sort_falls_back_to_default():
    stmt = apply_sort_limit_offset(
        select(Job), {"id": Job.id}, sort_by="whatever", sort_order="asc"
    )
    assert "ORDER BY" in str(stmt)
