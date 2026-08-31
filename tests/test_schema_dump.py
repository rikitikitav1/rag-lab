import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from use_cases.index import VECTOR_INDEX_PREFIX  # noqa: E402

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def test_the_dump_carries_no_index_that_belongs_to_a_variant():
    # 20260827000003 states the rule: a variant is a line in the config, so its vector
    # index is built at runtime. `dbmate dump` reads a live database and puts back every
    # index indexed on that machine, which is what this catches
    lines = [
        line for line in SCHEMA.read_text().splitlines()
        if "CREATE INDEX" in line and VECTOR_INDEX_PREFIX in line
    ]
    assert lines == [], "regenerate the dump and drop the per-variant hnsw indexes"


def test_every_migration_on_disk_is_in_the_dump():
    applied = {
        line.split("('")[1].split("')")[0]
        for line in SCHEMA.read_text().splitlines()
        if line.strip().startswith("('") and line.strip().endswith(");") or
        (line.strip().startswith("('") and line.strip().endswith("),"))
    }
    on_disk = {
        f.name.split("_")[0]
        for f in (SCHEMA.parent / "migrations").glob("*.sql")
    }
    assert on_disk - applied == set(), "a migration nobody dumped is a schema nobody has"


def test_no_enum_shaped_check_came_back_into_the_dump():
    # 20260827000007 and 000008 state the rule: an enum is validated in the model, so a
    # new value needs no migration. A dump taken on a database with a re-added check is
    # how the rule would come back without anybody deciding to bring it
    checks = [
        line.strip() for line in SCHEMA.read_text().splitlines()
        if "CHECK" in line and "= ANY (ARRAY[" in line
    ]
    assert checks == [], "an enum belongs to the model, not to a constraint"


def test_every_config_comment_is_at_most_three_lines():
    # the owner's rule, and config.yaml had run to 235 comment lines of 519 before it was
    # applied: the reasoning belongs in the arc log and the journal, and a `# tuned: file=`
    # line is how a value points at them
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    run, worst, where = 0, 0, 0
    for i, line in enumerate(root.joinpath("config.yaml").read_text().splitlines(), 1):
        run = run + 1 if line.strip().startswith("#") else 0
        if run > worst:
            worst, where = run, i
    assert worst <= 3, f"comment block of {worst} lines ending at config.yaml:{where}"

# what the tree holds today. It may go down and must never go up: a comment longer than
# this is a discussion, and its home is the arc log or a journal entry. Scoping the check
# to the branch diff instead was green before the commit and red after it, because the
# diff a test reads is the committed one
LONG_COMMENT_BLOCKS = 31


def _long_comment_blocks() -> list[str]:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    found = []
    for source in sorted(root.glob("app/**/*.py")) + sorted(root.glob("scripts/**/*.py")):
        run, start = 0, 0
        for i, line in enumerate(source.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                start = i if run == 0 else start
                run += 1
                continue
            if run > 3:
                found.append(f"{source.relative_to(root)}:{start} ({run} lines)")
            run = 0
    return found


def test_no_new_comment_block_runs_past_three_lines():
    blocks = _long_comment_blocks()
    assert len(blocks) <= LONG_COMMENT_BLOCKS, (
        f"{len(blocks)} blocks over three lines against a ceiling of "
        f"{LONG_COMMENT_BLOCKS}: {blocks[-5:]}"
    )
