from pathlib import Path

from use_cases.index import VECTOR_INDEX_PREFIX

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def test_the_dump_carries_no_index_that_belongs_to_a_variant():
    # a variant is a line in the config, so its vector index is built at runtime
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
    # an enum is validated in the model, so a new value needs no migration
    checks = [
        line.strip() for line in SCHEMA.read_text().splitlines()
        if "CHECK" in line and "= ANY (ARRAY[" in line
    ]
    assert checks == [], "an enum belongs to the model, not to a constraint"


def test_a_model_belongs_to_an_engine_and_says_which_weights_it_is():
    # the same name lives on two engines, and the same weights come in two quantisations
    from models.registry import Engine, Model, Placement

    assert Model.__table__.c.engine_id.nullable is False
    unique = {tuple(sorted(col.name for col in c.columns))
              for c in Model.__table__.constraints
              if c.__class__.__name__ == "UniqueConstraint"}
    assert ("engine_id", "name") in unique, "one name on two engines is two rows"
    assert Model.__table__.c.weights_id.nullable and Model.__table__.c.quant.nullable, (
        "not recorded is a state of its own, and a comparison must refuse on it"
    )
    # a join key spelled by hand drifts, so the weights are a row and the model points at it
    assert "weights" not in Model.__table__.c, "the free-text column was replaced by the table"
    assert Engine.__table__.c.env_prefix.nullable is False
    assert "base_url" not in Engine.__table__.c, "the address lives in the environment"
    assert Placement.remote.value == "remote", "cloud is an answer, not a missing value"
