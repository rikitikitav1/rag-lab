import os
from dataclasses import dataclass

QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "questions.md")


@dataclass
class Question:
    text: str
    marked_sources: list[str]


def load_questions(path: str = QUESTIONS_PATH) -> list[Question]:
    with open(path, "r", encoding="utf-8") as f:
        questions = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):
                question, _, raw = stripped_line.partition("|")
                marked = (
                    [] if raw.strip() == "NONE" else [x.strip() for x in raw.split(",")]
                )
                questions.append(Question(text=question.strip(), marked_sources=marked))
    return questions


def load_logs(run_name=None):
    from models.eval import QuestionLog
    from orm.sync_db import Session
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    with Session() as session:
        stmt = select(QuestionLog).options(selectinload(QuestionLog.question))
        if run_name:
            stmt = stmt.where(QuestionLog.run_name == run_name)
        return list(session.scalars(stmt))
