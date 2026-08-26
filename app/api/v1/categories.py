import config
from fastapi import APIRouter
from pydantic import BaseModel

import db

router = APIRouter(prefix="/categories", tags=["categories"])


class Category(BaseModel):
    name: str
    level: int = 0
    chunks: int


@router.get("")
def list_categories(
    only_top: bool | None = None, category: str | None = None
) -> list[Category]:
    rows = db.list_categories(only_top=only_top, category=category, variant=config.settings.corpus.variant)
    return [
        Category(name=row[0], chunks=row[1], level=row[0].count(".")) for row in rows
    ]
