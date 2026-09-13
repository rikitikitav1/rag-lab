import time
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
    # the broker sends a binary float, and its tail of nines read as a debit of its own
    return {"balance": round(float(seen.json()["data"]["balance"]), 8), "unit": "usd"}


# each broker shapes its service route its own way, so the row names a reader and the code keeps it
NO_READER = "none"
GONKA = "gonka_key"
READERS = {NO_READER: None, GONKA: _gonka_key}


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
    except Exception as e:
        return {**out, "why": f"{spec.name} did not say its balance: {type(e).__name__}"}


# only a cloud has a broker with a quota to ask
def _clouds() -> list[tuple[EngineSpec, str]]:
    with Session() as session:
        rows = session.execute(
            select(*COLUMNS, Engine.balance_reader)
            .where(Engine.kind == EngineKind.openai_compatible)
            .order_by(Engine.id)
        ).all()
    return [(EngineSpec(*row[:-1]), row[-1]) for row in rows]


_HELD: dict[int, tuple[float, dict]] = {}
HELD_SECONDS = 10


# the door's answer is held a few seconds: a loop on it spent the rate limit a running job needs
def summary() -> list[dict]:
    now, out = time.monotonic(), []
    for spec, reader in _clouds():
        held = _HELD.get(spec.id)
        if held is None or now - held[0] > HELD_SECONDS:
            held = _HELD[spec.id] = (now, read(spec, reader))
        out.append(held[1])
    return out


# every cloud that can say its balance, not the job's roles: a map of roles drifts and a new override goes unread
def readable() -> list[tuple[EngineSpec, str]]:
    return [(spec, reader) for spec, reader in _clouds() if READERS.get(reader) is not None]
