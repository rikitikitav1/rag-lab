import config
import corpus_search
import search_scope
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/categories", tags=["categories"])


class Category(BaseModel):
    name: str
    group: str
    level: int = 0
    chunks: int


class Tag(BaseModel):
    name: str
    chunks: int


@router.get("")
def list_categories(
    only_top: bool | None = None,
    category: str | None = Query(default=None, pattern=search_scope.CATEGORY_RE.pattern),
) -> list[Category]:
    rows = corpus_search.list_categories(only_top=only_top, category=category, variant=config.settings.corpus.variant)
    return [Category(name=name, group=group, chunks=n, level=0 if only_top else 1) for name, group, n in rows]


@router.get("/tags")
def list_tags(limit: int = Query(default=50, ge=1, le=1000)) -> list[Tag]:
    tags = corpus_search.list_tags(limit, variant=config.settings.corpus.variant)
    return [Tag(name=name, chunks=n) for name, n in tags]
