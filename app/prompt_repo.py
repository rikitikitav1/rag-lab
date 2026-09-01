from models.registry import Prompt, Purpose
from orm.sync_db import Session
from sqlalchemy import select


# template and version together: read apart they were two sessions per judged axis
def active(purpose: Purpose) -> tuple[str, int]:
    with Session() as session:
        row = session.execute(
            select(Prompt.template, Prompt.version).where(
                Prompt.purpose == purpose, Prompt.active
            )
        ).first()
    if row is None:
        raise RuntimeError(f"no active prompt for purpose {purpose}")
    return row.template, row.version


def active_template(purpose: Purpose) -> str:
    return active(purpose)[0]


def active_version(purpose: Purpose) -> int:
    return active(purpose)[1]


# one read for a record's worth of versions: a row naming four prompts opened four sessions
def active_versions(purposes) -> dict[str, int]:
    wanted = list(purposes)
    with Session() as session:
        rows = session.execute(
            select(Prompt.purpose, Prompt.version).where(
                Prompt.purpose.in_(wanted), Prompt.active
            )
        ).all()
    found = {p: v for p, v in rows}
    missing = [p for p in wanted if p not in found]
    if missing:
        raise RuntimeError(f"no active prompt for purposes {[p.name for p in missing]}")
    return {p.name: found[p] for p in wanted}


def template_of(purpose: Purpose, version: int) -> str:
    with Session() as session:
        template = session.scalar(
            select(Prompt.template).where(
                Prompt.purpose == purpose, Prompt.version == version
            )
        )
    if template is None:
        raise RuntimeError(f"no prompt {purpose} version {version}")
    return template
