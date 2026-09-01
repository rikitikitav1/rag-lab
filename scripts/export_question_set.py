"""A criterion set lives in a file, not only in a database that gets recreated."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "questions"

# reference_answer is markdown with newlines: it stays derivable from the original, not flattened
COLUMNS = (
    "set_name", "language", "kind", "marked_sources", "original_text", "source_question_text",
)

QUERY = """
SELECT q.set_name, q.language, q.kind, q.marked_sources, q.original_text,
       o.original_text AS source_question_text
FROM questions q
LEFT JOIN questions o ON o.id = q.source_question_id
WHERE q.set_name = :set_name
ORDER BY q.id
"""


def export(set_name: str) -> Path:
    from orm.sync_db import engine
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text(QUERY), {"set_name": set_name}).mappings().all()
    if not rows:
        raise SystemExit(f"set '{set_name}' is empty, nothing to export")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{set_name}.tsv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([
                row["set_name"] or "",
                row["language"] or "",
                row["kind"] or "",
                ",".join(row["marked_sources"] or []),
                row["original_text"].replace("\t", " ").replace("\n", " "),
                (row["source_question_text"] or "").replace("\t", " ").replace("\n", " "),
            ])
    print(f"wrote {path} ({len(rows)} questions)")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: export_question_set.py <set_name> [<set_name> ...]")
    for name in sys.argv[1:]:
        export(name)
