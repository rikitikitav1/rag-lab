import hashlib
import time

import job_queue
from fastapi import APIRouter, Depends, File, Form, UploadFile
from models.eval import Question
from orm.async_db import get_session
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/questions", tags=["questions"])


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse(content: str) -> list[tuple[str, str, list[str]]]:
    seen: set[str] = set()
    rows: list[tuple[str, str, list[str]]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        question, sep, raw = line.partition("|")
        question = question.strip()
        if not question:
            continue
        h = _text_hash(question)
        if h in seen:
            continue
        seen.add(h)
        if not sep or raw.strip().upper() == "NONE":
            marked: list[str] = []
        else:
            marked = [x.strip() for x in raw.split(",") if x.strip()]
        rows.append((h, question, marked))
    return rows


class ImportResponse(BaseModel):
    set_name: str
    parsed: int
    inserted: int
    run_name: str | None


@router.post("/import", response_model=ImportResponse)
async def import_questions(
    file: UploadFile = File(...),
    set_name: str = Form(...),
    language: str | None = Form(default=None),
    run: bool = Form(default=False),
    run_name: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
):
    content = (await file.read()).decode("utf-8")
    parsed = _parse(content)

    inserted = 0
    if parsed:
        values = [
            {
                "text_hash": h,
                "original_text": question,
                "set_name": set_name,
                "language": language,
                "marked_sources": marked,
            }
            for h, question, marked in parsed
        ]
        result = await session.scalars(
            pg_insert(Question)
            .values(values)
            .on_conflict_do_nothing(index_elements=["text_hash"])
            .returning(Question.id)
        )
        inserted = len(result.all())

    job_queue.add_job(session, "embed_questions", {})

    resolved_run = None
    if run:
        resolved_run = run_name or f"{set_name}_{int(time.time())}"
        job_queue.add_job(
            session,
            "eval_run",
            {"run_name": resolved_run, "set_name": set_name, "question_ids": None},
        )

    await session.commit()
    return ImportResponse(
        set_name=set_name,
        parsed=len(parsed),
        inserted=inserted,
        run_name=resolved_run,
    )
