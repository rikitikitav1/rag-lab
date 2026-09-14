# a throwaway database built from the dump, for the rules that live in sql and nowhere else
import os
import uuid

import pytest
from sqlalchemy import create_engine, make_url, text

SCHEMA = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
ADMIN = os.getenv("TEST_POSTGRES_URL", "postgresql+psycopg://postgres@localhost:5432/postgres")


def _reachable() -> bool:
    probe = create_engine(ADMIN, connect_args={"connect_timeout": 3})
    try:
        probe.connect().close()
        return True
    except Exception:
        return False
    finally:
        probe.dispose()


pytestmark = pytest.mark.skipif(
    not _reachable(), reason=f"no postgres at {ADMIN}; start the stack or set TEST_POSTGRES_URL"
)


# the body of the `db` fixture in conftest: one database per module, dropped after
def scratch():
    name = f"ragtest_{uuid.uuid4().hex[:12]}"
    admin = create_engine(ADMIN, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    # parsed, not split: a query string after the database name was dropped with it
    scratch = create_engine(make_url(ADMIN).set(database=name), isolation_level="AUTOCOMMIT")
    with scratch.connect() as c:
        c.execute(text(_loadable(open(SCHEMA, encoding="utf-8").read())))
    # the dump empties `search_path` for its session, and the pool hands that session on
    scratch.dispose()
    yield scratch
    scratch.dispose()
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


# pg_dump writes psql meta-commands; the driver speaks sql, so they are dropped rather than sent
def _loadable(dump: str) -> str:
    return "\n".join(x for x in dump.splitlines() if not x.startswith("\\"))
