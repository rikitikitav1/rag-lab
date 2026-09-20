"""Three classes of a candidate chunk: the gold section, its neighbour in the gold file, a stranger."""

from evals.retrieval_metrics import is_gold
from use_cases.retrieval_compare import clean_gold, heading_text

SCHEMA = 1

GOLD = "gold_section"
NEIGHBOUR = "gold_file_other_section"
STRANGER = "stranger"

READS = (
    "retention is read on the gold section alone; the floor on dropped strangers is read on the"
    " third class; the neighbour is neither, because the file was marked and the section was not"
)


# the stand's own predicate, by its own functions: an SQL rewrite of it drifted twice before
def classify(candidate: dict, marked, gold_heading: str | None) -> str:
    if not is_gold(candidate.get("source") or "", marked or ()):
        return STRANGER
    heading = heading_text(candidate.get("section"))
    return GOLD if heading and heading == clean_gold(gold_heading) else NEIGHBOUR


def of_row(row: dict) -> list[str]:
    return [
        classify(candidate, row.get("marked_sources"), row.get("gold_heading"))
        for candidate in row.get("candidates") or ()
    ]


# the denominator of retention: a row whose gold section never reached the pool cannot keep it
def report(rows: list[dict]) -> dict:
    counts = {GOLD: 0, NEIGHBOUR: 0, STRANGER: 0}
    with_gold_section = with_gold_file = 0
    for row in rows:
        classes = of_row(row)
        for name in classes:
            counts[name] += 1
        with_gold_section += GOLD in classes
        with_gold_file += any(c in (GOLD, NEIGHBOUR) for c in classes)
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "candidates": sum(counts.values()),
        "by_class": counts,
        "rows_with_the_gold_section": with_gold_section,
        "rows_with_the_gold_file": with_gold_file,
        "reads": READS,
    }
