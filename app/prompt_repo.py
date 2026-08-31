from models.registry import Prompt, Purpose
from orm.sync_db import Session
from sqlalchemy import select


def active_template(purpose: Purpose) -> str:
    with Session() as session:
        template = session.scalar(
            select(Prompt.template).where(Prompt.purpose == purpose, Prompt.active)
        )
    if template is None:
        raise RuntimeError(f"no active prompt for purpose {purpose}")
    return template


def active_version(purpose: Purpose) -> int:
    with Session() as session:
        version = session.scalar(
            select(Prompt.version).where(Prompt.purpose == purpose, Prompt.active)
        )
    if version is None:
        raise RuntimeError(f"no active prompt for purpose {purpose}")
    return version


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


def active(purpose: Purpose) -> tuple[str, int]:
    """Template and version together: read apart they were two sessions per judged axis."""
    with Session() as session:
        row = session.execute(
            select(Prompt.template, Prompt.version).where(
                Prompt.purpose == purpose, Prompt.active
            )
        ).first()
    if row is None:
        raise RuntimeError(f"no active prompt for purpose {purpose}")
    return row.template, row.version
