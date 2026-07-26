import pytest
from models.jobs import Job
from pydantic import ValidationError
from query_utils import Page, apply_sort_limit_offset
from sqlalchemy import select


def test_page_defaults():
    p = Page()
    assert p.limit == 100
    assert p.offset == 0
    assert p.sort_by is None
    assert p.sort_order == "desc"


def test_page_limit_cap():
    assert Page(limit=1000).limit == 1000
    with pytest.raises(ValidationError):
        Page(limit=1001)


def test_page_sort_order_literal():
    with pytest.raises(ValidationError):
        Page(sort_order="descending")


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
