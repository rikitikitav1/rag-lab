from typing import TypeVar

import config
from pgvector.asyncpg import register_vector
from sqlalchemy import URL, event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

T = TypeVar("T")


def postgres_url():
    p = config.settings.postgres
    return URL.create(
        "postgresql+asyncpg",
        username=p.user,
        host=p.host,
        port=p.port,
        database=p.dbname,
    )


engine = create_async_engine(
    postgres_url(),
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    connect_args={
        "timeout": 5,
        "server_settings": {
            "statement_timeout": "30000",
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
