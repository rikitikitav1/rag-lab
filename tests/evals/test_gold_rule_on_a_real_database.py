# the one rule "this chunk belongs to a marked file", asked of python and of the sql that mirrors it

from corpus_keys import Gold
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
            assert bool(said) is Gold.coerce(marked).holds_file(source), f"{source} against {marked}"


EXACT = [
    # (chunk source, chunk section, chunk versions)
    ("arango/3.12/aql.md", "AQL > Operators", ["3.12"]),
    ("arango/3.12/aql.md", "AQL > Operators > Comparison", ["3.12"]),
    ("arango/3.12/aql.md", "AQL > Operators overview", ["3.12"]),
    ("arango/3.12/aql.md", "AQL", ["3.12"]),
    ("arango/3.12/aql.md", "AQL > Operators", ["3.11"]),
    ("arango/3.12/aql.md", "AQL > Operators", []),
    ("arango/3.12/aql-old.md", "AQL > Operators", ["3.12"]),
    ("arango/3.12/aql.md", None, ["3.12"]),
]


# the exact gold is one test in python and in sql: a sub-section holds, a sibling sharing a prefix does not
def test_the_exact_gold_says_the_same_in_python_and_in_sql(db):
    from corpus_keys import Gold

    file = ("arango/3.12/aql.md",)
    for gold in (Gold(file, "AQL > Operators", "3.12"), Gold(file, "AQL > Operators")):
        columns = ("CAST(:source AS text)", "CAST(:section AS text)", "CAST(:versions AS text[])")
        clause, params = gold.section_sql(*columns)
        with db.connect() as c:
            for source, section, versions in EXACT:
                said = c.execute(
                    text(f"SELECT coalesce(({clause}), false)"),
                    {"source": source, "section": section, "versions": versions, **params},
                ).scalar()
                assert bool(said) is gold.holds_section(source, section, versions), (gold, source, section, versions)


# a set's unreachable questions counted by both golds: an older mark by containment, an exact gold by its section
def test_unreachable_questions_are_counted_by_both_golds(db, monkeypatch):
    import json

    import db as stand

    monkeypatch.setattr(stand, "engine", db)
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("DELETE FROM questions"))
        c.execute(text("INSERT INTO data_sources (id, name, kind, active) VALUES (1, 'arango', 'git', true)"))
        c.execute(
            text(
                "INSERT INTO data_chunks (source_id, source, content, chunk_index, language, variant, section,"
                " versions) VALUES (1, 'arango/aql.md', 'c', 0, 'en', 'v', 'AQL > Operators', ARRAY['3.12'])"
            )
        )
        rows = [
            ("old_hit", ["arango"], None),
            ("old_miss", ["nowhere.md"], None),
            ("new_hit", [], {"file": "arango/aql.md", "section": "AQL", "version": "3.12"}),
            ("new_miss", [], {"file": "arango/aql.md", "section": "AQL > Functions", "version": None}),
            ("new_old_version", [], {"file": "arango/aql.md", "section": "AQL", "version": "3.11"}),
        ]
        for i, (set_name, marks, gold) in enumerate(rows):
            c.execute(
                text(
                    "INSERT INTO questions (original_text, text_hash, set_name, marked_sources, gold)"
                    " VALUES ('q', :h, :s, :m, CAST(:g AS jsonb))"
                ),
                {"h": f"h{i}", "s": set_name, "m": marks, "g": json.dumps(gold) if gold else None},
            )
        c.commit()

    assert stand.unreachable_by_set(variant="v") == [("new_miss", 1), ("new_old_version", 1), ("old_miss", 1)]


# a question carries one kind of gold: the base refuses both, and a gold with no section refuses to be read
def test_a_question_holds_one_kind_of_gold(db):
    import pytest
    from corpus_keys import Gold
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match="questions_one_kind_of_gold"), db.connect() as c:
        c.execute(
            text(
                "INSERT INTO questions (original_text, text_hash, marked_sources, gold)"
                " VALUES ('q', 'both', ARRAY['a.md'], CAST('{\"file\": \"a.md\", \"section\": \"A\"}' AS jsonb))"
            )
        )
    with pytest.raises(ValueError, match="names its file and section"):
        Gold.of([], {"file": "a.md", "section": None})


# a cleanup that renames a page's root moves the gold to the same file and leaf; evidence settles a doubled leaf
def test_golds_follow_their_leaf_to_the_new_root(db, monkeypatch):
    import json

    from models.eval import Question
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker
    from use_cases import gold_reanchor

    monkeypatch.setattr(gold_reanchor, "engine", db)
    monkeypatch.setattr(gold_reanchor, "Session", sessionmaker(bind=db))
    chunks = [
        ("d/a.md", "Title > Setup", "install it"),
        ("d/a.md", "Title > Other", "other text"),
        ("d/b.md", "X > Run", "nothing here"),
        ("d/b.md", "Y > Run", "the exact evidence words"),
    ]
    golds = {
        "moved": ("d/a.md", "Old > Setup", "install it"),
        "by_evidence": ("d/b.md", "Z > Run", "the exact evidence words"),
        "file_gone": ("d/gone.md", "Old > Setup", None),
        "leaf_gone": ("d/a.md", "Old > Missing", None),
        "evidence_gone": ("d/a.md", "Old > Other", "words no chunk holds"),
        "respelled": ("d/a.md", "Old > Set up", "install it"),
    }
    with db.connect() as c:
        c.execute(text("TRUNCATE data_sources CASCADE"))
        c.execute(text("DELETE FROM questions"))
        c.execute(text("INSERT INTO data_sources (id, name, kind, active) VALUES (1, 'd', 'git', true)"))
        for i, (source, section, content) in enumerate(chunks):
            c.execute(
                text(
                    "INSERT INTO data_chunks (source_id, source, content, chunk_index, language, variant, section)"
                    " VALUES (1, :source, :content, :i, 'en', 'v', :section)"
                ),
                {"source": source, "content": content, "i": i, "section": section},
            )
        for i, (name, (file, section, evidence)) in enumerate(golds.items()):
            c.execute(
                text(
                    "INSERT INTO questions (original_text, text_hash, set_name, gold, evidence)"
                    " VALUES (:q, :h, 's', CAST(:g AS jsonb), :e)"
                ),
                {"q": name, "h": f"h{i}", "g": json.dumps({"file": file, "section": section}), "e": evidence},
            )
        c.commit()

    assert gold_reanchor.reanchor("s", "v", dry=True)["moved"] == 2
    summary = gold_reanchor.reanchor("s", "v")
    assert {k: summary[k] for k in ("moved", "moved_by_evidence", "file_gone", "leaf_gone")} == {
        "moved": 2, "moved_by_evidence": 1, "file_gone": 1, "leaf_gone": 2}
    with sessionmaker(bind=db)() as session:
        sections = {q.original_text: q.gold["section"] for q in session.scalars(select(Question))}
        named = {q.id: q.original_text for q in session.scalars(select(Question))}
    left = {(e["file"], e["section"], e["why"], tuple(named[i] for i in e["question_ids"])) for e in summary["left"]}
    assert left == {("d/gone.md", "Old > Setup", "file_gone", ("file_gone",)),
                    ("d/a.md", "Old > Missing", "leaf_gone", ("leaf_gone",)),
                    ("d/a.md", "Old > Other", "leaf_gone", ("evidence_gone",))}
    assert sections == {"moved": "Title > Setup", "by_evidence": "Y > Run", "file_gone": "Old > Setup",
                        "leaf_gone": "Old > Missing", "evidence_gone": "Old > Other", "respelled": "Title > Setup"}
