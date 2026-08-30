import csv
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
from models.mcp_integration import McpIntegration
from models.registry import Prompt, Purpose
from orm.sync_db import Session
from sqlalchemy import exists, select, update
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
            if has_active:
                log.warning(
                    "seed.prompt_inactive",
                    purpose=str(purpose),
                    version=freshest.version,
                    hint="activate via POST /v1/prompt/{id}/activate",
                )
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


EXPORTED_SETS = Path("datasets/questions")


# a paraphrase is worthless without its original: the section-level gold comes from that link
def _exported_rows() -> list[dict]:
    answers = _reference_answers()
    rows = []
    for file in sorted(EXPORTED_SETS.glob("*.tsv")):
        with file.open(encoding="utf-8", newline="") as fh:
            # csv, not split: the writer quotes fields containing quotes, and a naive split keeps them
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                origin = row.get("source_question_text") or None
                rows.append(
                    {
                        "text_hash": _text_hash(row["original_text"]),
                        "original_text": row["original_text"],
                        "set_name": row["set_name"] or None,
                        "language": row["language"] or None,
                        "kind": row["kind"] or None,
                        "marked_sources": [s for s in row["marked_sources"].split(",") if s],
                        "reference_answer": answers.get(origin or row["original_text"]),
                        # every row carries the key, or the insert would drop it for the whole batch
                        "source_question_id": None,
                        "_source_text": origin,
                    }
                )
    return rows


# after the insert, never before: a set's originals can live in the same exported batch,
# and a lookup that runs first finds nothing while `on_conflict_do_nothing` makes the
# missing link permanent. Unlinked, a question takes its own text as the gold heading
def _link_originals(session, rows: list[dict]) -> None:
    wanted = {r.get("_source_text") for r in rows if r.get("_source_text")}
    if not wanted:
        return
    ids = dict(
        session.execute(
            select(Question.text_hash, Question.id).where(
                Question.text_hash.in_([_text_hash(t) for t in wanted])
            )
        ).all()
    )
    linked = 0
    for row in rows:
        origin = row.pop("_source_text", None)
        target = ids.get(_text_hash(origin)) if origin else None
        if target is None:
            continue
        linked += session.execute(
            update(Question)
            .where(
                Question.text_hash == row["text_hash"],
                Question.source_question_id.is_(None),
            )
            .values(source_question_id=target)
        ).rowcount
    session.commit()
    log.info("seed.linked_originals", linked=linked, wanted=len(wanted))


def _insert_questions(session, rows: list[dict]) -> None:
    size = config.settings.ingestion.commit_size
    for i in range(0, len(rows), size):
        stmt = pg_insert(Question).values(
            rows[i : i + size]
        ).on_conflict_do_nothing(index_elements=["text_hash"])
        session.execute(stmt)
        session.commit()


def seed_questions() -> None:
    rows = _question_rows()
    with Session() as session:
        if rows:
            _insert_questions(session, rows)
        exported = _exported_rows()
        if exported:
            # the link is resolved after the insert, so the batch goes in without it and
            # `_source_text` stays out of the statement: it is a lookup key, not a column
            _insert_questions(
                session,
                [{k: v for k, v in row.items() if k != "_source_text"} for row in exported],
            )
            _link_originals(session, exported)
    log.info("seed.questions", total=len(rows), exported=len(exported))


MCP_INTEGRATIONS = [
    {
        "name": "deepwiki",
        "url": "https://mcp.deepwiki.com/mcp",
        "auth": None,
    },
    {
        "name": "hf",
        "url": "https://huggingface.co/mcp",
        "auth": {"type": "bearer", "token_env": "HF_TOKEN"},
    },
    {
        "name": "context7",
        "url": "https://mcp.context7.com/mcp",
        "auth": {
            "type": "header",
            "header": "CONTEXT7_API_KEY",
            "value_env": "CONTEXT7_API_KEY",
        },
    },
]


def seed_mcp_integrations() -> None:
    with Session() as session:
        rows = session.execute(
            select(McpIntegration.name, McpIntegration.url)
        ).all()
        names = {row.name for row in rows}
        urls = {row.url for row in rows}
        fresh = [
            item
            for item in MCP_INTEGRATIONS
            if item["name"] not in names and item["url"] not in urls
        ]
        if fresh:
            session.execute(pg_insert(McpIntegration).values(fresh))
            session.commit()
    log.info(
        "seed.mcp_integrations",
        seeded=len(fresh),
        skipped=len(MCP_INTEGRATIONS) - len(fresh),
    )


def main() -> None:
    logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))
    seed_prompts()
    seed_questions()
    seed_mcp_integrations()


if __name__ == "__main__":
    main()
