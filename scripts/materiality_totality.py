"""Does every materiality predicate answer on every row, or does NULL swallow some of them."""

import argparse
import json
from pathlib import Path

from job_handlers.judging import GUEST_MATERIAL
from models.eval import Question, QuestionLog
from orm.sync_db import Session
from sqlalchemy import func, not_, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with Session() as session:
        rows = session.scalar(select(func.count()).select_from(QuestionLog)) or 0
        out = {"rows": rows, "keys": {}}
        for name, clause in GUEST_MATERIAL.items():
            base = select(func.count()).select_from(QuestionLog).outerjoin(
                Question, QuestionLog.question_id == Question.id
            )
            yes = session.scalar(base.where(clause)) or 0
            no = session.scalar(base.where(not_(clause))) or 0
            out["keys"][name] = {"yes": yes, "no": no, "total": yes + no, "answers_every_row":
                                 yes + no == rows}
    out["total"] = all(k["answers_every_row"] for k in out["keys"].values())
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    raise SystemExit(0 if out["total"] else 1)


if __name__ == "__main__":
    main()
