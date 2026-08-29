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
