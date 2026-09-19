# the one rule "this chunk belongs to a marked file", asked of python and of the sql that mirrors it

from evals.retrieval_metrics import is_gold
from real_db import pytestmark  # noqa: F401
from sqlalchemy import text

CASES = [
    ("notes/kafka/partitions.md", ["notes/kafka/partitions.md"]),
    ("notes/kafka/partitions.md", ["notes/kafka"]),
    ("notes/kafka/partitions-old.md", ["notes/kafka/partitions.md"]),
    ("archive/notes/kafka/partitions.md", ["notes/kafka/partitions.md"]),
    ("notes/redis/keys.md", ["notes/kafka"]),
]


def test_the_sql_that_finds_chunks_of_a_marked_file_says_what_is_gold_says(db):
    # `_sections_under` writes this predicate in sql; a rewrite of it drifted twice before
    with db.connect() as c:
        for source, marked in CASES:
            said = c.execute(
                text("SELECT (" + " OR ".join(
                    f"position(:m{i} in :source) > 0" for i in range(len(marked))
                ) + ")"),
                {"source": source, **{f"m{i}": m for i, m in enumerate(marked)}},
            ).scalar()
            assert bool(said) is is_gold(source, marked), f"{source} against {marked}"
