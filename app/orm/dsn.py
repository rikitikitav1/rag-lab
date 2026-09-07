import os

import config
from sqlalchemy import URL

# one owner: a timeout changed on one side used to leave the other where it was
POOL_SIZE = 5
MAX_OVERFLOW = 5
CONNECT_TIMEOUT_SECONDS = 5
STATEMENT_TIMEOUT_MS = 30000
# a generic plan cannot prove a partial index predicate from a bound parameter
PLAN_CACHE_MODE = "force_custom_plan"


def postgres_url(driver: str):
    p = config.settings.postgres
    # the compose hostname resolves inside the network only; a script on the host says where
    host = os.getenv("POSTGRES_HOST") or p.host
    return URL.create(driver, username=p.user, host=host, port=p.port, database=p.dbname)
