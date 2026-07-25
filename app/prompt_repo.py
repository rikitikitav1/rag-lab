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
