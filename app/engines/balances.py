from datetime import UTC, datetime

import requests
from models.registry import Engine, EngineKind
from orm.sync_db import Session
from sqlalchemy import select

from .core import EngineSpec, Unconfigured, base_url, bearer
from .lookup import COLUMNS

TIMEOUT = 15


def _gonka_key(spec: EngineSpec) -> dict:
    seen = requests.get(f"{base_url(spec)}/v1/auth/key", headers=bearer(spec), timeout=TIMEOUT)
    seen.raise_for_status()
    return {"balance": seen.json()["data"]["balance"], "unit": "usd"}


# each broker shapes its service route its own way, so the row names a reader and the code keeps it
READERS = {"none": None, "gonka_key": _gonka_key}


def refuse_unknown(name: str) -> None:
    if name not in READERS:
        raise ValueError(f"unknown balance reader {name}; known: {', '.join(sorted(READERS))}")


# a cloud that cannot say what is left says why, instead of dropping out of the summary
def read(spec: EngineSpec, reader: str) -> dict:
    out = {"engine": spec.name, "reader": reader, "balance": None, "unit": None, "why": None,
           "read_at": datetime.now(UTC).isoformat(timespec="seconds")}
    if READERS.get(reader) is None:
        return {**out, "why": f"no balance reader named; PATCH /v1/engine/{spec.id} with balance_reader"}
    try:
        return {**out, **READERS[reader](spec)}
    except Unconfigured as e:
        return {**out, "why": str(e)}
    except requests.HTTPError as e:
        return {**out, "why": f"{spec.name} answered http {e.response.status_code}"}
    except (requests.RequestException, KeyError, ValueError) as e:
        return {**out, "why": f"{spec.name} did not say its balance: {type(e).__name__}"}


# only a cloud has a broker with a quota to ask
def summary() -> list[dict]:
    with Session() as session:
        rows = session.execute(
            select(*COLUMNS, Engine.balance_reader)
            .where(Engine.kind == EngineKind.openai_compatible)
            .order_by(Engine.id)
        ).all()
    return [read(EngineSpec(*row[:-1]), row[-1]) for row in rows]
