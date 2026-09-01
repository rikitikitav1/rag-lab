from typing import TypeVar

from pgvector.asyncpg import register_vector
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from orm import dsn

T = TypeVar("T")


def postgres_url():
    return dsn.postgres_url("postgresql+asyncpg")


engine = create_async_engine(
    postgres_url(),
    pool_size=dsn.POOL_SIZE,
    max_overflow=dsn.MAX_OVERFLOW,
    pool_pre_ping=True,
    connect_args={
        "timeout": dsn.CONNECT_TIMEOUT_SECONDS,
        "server_settings": {
            "statement_timeout": str(dsn.STATEMENT_TIMEOUT_MS),
            "plan_cache_mode": dsn.PLAN_CACHE_MODE,
        },
    },
)


@event.listens_for(engine.sync_engine, "connect")
def _register_vector(dbapi_conn, _):
    dbapi_conn.run_async(register_vector)


session_factory = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


async def get_session():
    async with session_factory() as session:
        yield session


async def commit_and_refresh(session: AsyncSession, obj: T) -> T:
    await session.commit()
    await session.refresh(obj)
    return obj
