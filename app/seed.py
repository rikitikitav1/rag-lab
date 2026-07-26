import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import config
import logging_setup
from models.eval import Question
from models.registry import Prompt, Purpose
from orm.sync_db import Session
from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

log = logging_setup.get_logger(__name__)

_PROMPT_RE = re.compile(r"^(?P<purpose>.+)\.v(?P<version>\d+)\.txt$")


@dataclass
class PromptFile:
    purpose: Purpose
    version: int
    text: str


def load_prompt_files(prompts_dir=config.settings.prompts_dir) -> list[PromptFile]:
    result: list[PromptFile] = []
    for path in Path(prompts_dir).glob("*.txt"):
        match = _PROMPT_RE.match(path.name)
        if not match:
            continue
        try:
            purpose = Purpose[match.group("purpose")]
        except KeyError:
            raise ValueError(
                f"unknown prompt purpose in filename: {path.name}"
            ) from None
        result.append(
            PromptFile(
                purpose=purpose,
                version=int(match.group("version")),
                text=path.read_text(encoding="utf-8"),
            )
        )
    return result


def _existing_keys(session) -> set[tuple[Purpose, int]]:
    rows = session.execute(select(Prompt.purpose, Prompt.version)).all()
    return {tuple(row) for row in rows}


def seed_prompts() -> None:
    with Session() as session:
        existing = _existing_keys(session)

        by_purpose: dict[Purpose, list[PromptFile]] = defaultdict(list)
        for file in load_prompt_files():
            if (file.purpose, file.version) not in existing:
                by_purpose[file.purpose].append(file)

        for purpose, new_files in by_purpose.items():
            # freshest version becomes active only when no active prompt exists yet
            has_active = session.scalar(
                select(exists().where(Prompt.purpose == purpose, Prompt.active))
            )
            freshest = max(new_files, key=lambda f: f.version)
            for file in new_files:
                active = not has_active and file is freshest
                session.add(
                    Prompt(
                        purpose=purpose,
                        version=file.version,
                        template=file.text,
                        active=active,
                    )
                )
                log.info(
                    "seed.prompt",
                    purpose=str(purpose),
                    version=file.version,
                    active=active,
                )

        session.commit()


QUESTIONS_TSV = "questions.tsv"
ANSWERS_JSONL = "answers_interview.jsonl"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reference_answers(path=ANSWERS_JSONL) -> dict[str, str]:
    answers = {}
    file = Path(path)
    if not file.exists():
        return answers
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            answers[record["question"]] = record.get("reference_answer")
    return answers


def _question_rows() -> list[dict]:
    answers = _reference_answers()
    lines = Path(QUESTIONS_TSV).read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")

    rows, seen = [], set()
    for line in lines[1:]:
        if not line.strip():
            continue
        row = dict(zip(header, line.split("\t"), strict=False))
        text = row["original_text"]
        text_hash = _text_hash(text)
        if text_hash in seen:
            continue
        seen.add(text_hash)
        rows.append(
            {
                "text_hash": text_hash,
                "original_text": text,
                "set_name": row["set_name"] or None,
                "language": row["language"] or None,
                "kind": row["kind"] or None,
                "marked_sources": [s for s in row["marked_sources"].split(",") if s],
                "reference_answer": answers.get(text),
            }
        )
    return rows


def seed_questions() -> None:
    rows = _question_rows()
    if not rows:
        return
    size = config.settings.ingestion.commit_size
    with Session() as session:
        for i in range(0, len(rows), size):
            stmt = pg_insert(Question).values(
                rows[i : i + size]
            ).on_conflict_do_nothing(index_elements=["text_hash"])
            session.execute(stmt)
            session.commit()
    log.info("seed.questions", total=len(rows))


def main() -> None:
    logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))
    seed_prompts()
    seed_questions()


if __name__ == "__main__":
    main()
