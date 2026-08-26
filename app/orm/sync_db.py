import config
from pgvector.psycopg import register_vector
from sqlalchemy import URL, create_engine, event
from sqlalchemy.orm import sessionmaker


def postgres_url():
    p = config.settings.postgres
    return URL.create(
        "postgresql+psycopg",
        username=p.user,
        host=p.host,
        port=p.port,
        database=p.dbname,
    )


engine = create_engine(
    postgres_url(),
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 5,
        # a generic plan cannot prove a partial index predicate from a bound parameter
        "options": (
            "-c statement_timeout=30000 -c plan_cache_mode=force_custom_plan"
            f" -c hnsw.ef_search={config.settings.retrieval.ef_search}"
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
