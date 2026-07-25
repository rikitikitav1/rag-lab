from typing import TypeVar

from fastapi import HTTPException
from orm.base import Base
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T", bound=Base)


async def get_or_404(
    model: type[T],
    id: int,
    session: AsyncSession,
) -> T:
    obj = await session.get(model, id)

    if obj is None:
        raise HTTPException(
            status_code=404,
            detail=f"{model.__name__} with id={id} not found",
        )

    return obj


def ensure_not_active(obj: Base) -> None:
    if getattr(obj, "active", False):
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete the active {type(obj).__name__}; activate another first",
        )
