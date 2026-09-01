from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from orm import dsn


def postgres_url():
    return dsn.postgres_url("postgresql+psycopg")


engine = create_engine(
    postgres_url(),
    pool_size=dsn.POOL_SIZE,
    max_overflow=dsn.MAX_OVERFLOW,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": dsn.CONNECT_TIMEOUT_SECONDS,
        # no hnsw.ef_search here: the depth is resolved per search, and `auto` needs a connection
        "options": (
            f"-c statement_timeout={dsn.STATEMENT_TIMEOUT_MS}"
            f" -c plan_cache_mode={dsn.PLAN_CACHE_MODE}"
        ),
    },
)


@event.listens_for(engine, "connect")
def _register_vector(dbapi_conn, _):
    register_vector(dbapi_conn)


Session = sessionmaker(engine)


def get_session():
    with Session() as session:
        yield session
