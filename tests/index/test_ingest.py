import ingest
import pytest

# the ceiling every variant of the corpus declares today
CEILING = 1024


def test_split_by_size_short_is_untouched():
    assert ingest.split_by_size("short text", max_size=CEILING) == ["short text"]


def test_split_by_size_respects_max():
    long = "a" * (CEILING * 2 + 50)
    parts = ingest.split_by_size(long, max_size=CEILING)
    assert len(parts) >= 2
    assert all(len(p) <= CEILING for p in parts)


def test_split_by_size_prefers_paragraph_boundary():
    para = "x" * (CEILING - 10)
    parts = ingest.split_by_size(para + "\n\n" + para, max_size=CEILING)
    assert parts == [para, para]


@pytest.mark.parametrize(
    "path, expected",
    [
        ("databases/postgresql/locks.md", ["databases", "postgresql", "locks"]),
        ("foo/ba r@x.md", ["foo", "ba r@x"]),
        ("single.md", ["single"]),
    ],
)
def test_a_file_path_becomes_the_tags_the_old_category_path_carried(path, expected):
    from sources.base import Base

    assert Base.tags_for(None, path) == expected


def _reader(categories, by_path=None):
    from types import SimpleNamespace

    from sources.base import Base

    reader = SimpleNamespace(name="s", settings=SimpleNamespace(categories=categories, category_by_path=by_path or {}))
    return lambda rel: Base.category_for(reader, rel)


def test_a_file_takes_the_longest_path_prefix_then_the_sources_one_category_then_none():
    assert _reader(["redis"])("docs/a.md") == "redis"
    assert _reader([])("docs/a.md") is None
    by_path = {"docs/": "redis", "docs/pg/": "postgresql"}
    assert _reader(["redis", "postgresql"], by_path)("docs/pg/a.md") == "postgresql"
    assert _reader(["redis", "postgresql"], by_path)("docs/a.md") == "redis"


def test_a_source_of_several_categories_refuses_a_file_under_no_prefix():
    with pytest.raises(ValueError, match="under no category_by_path"):
        _reader(["redis", "postgresql"], {"docs/": "redis"})("other/a.md")


def test_a_label_names_a_category_or_every_category_of_a_group():
    import db

    assert db.categories_of("redis") == ["redis"]
    assert "postgresql" in db.categories_of("databases") and "redis" in db.categories_of("databases")
    assert db.categories_of("interview") == []


def _doc(file: str, body: str, section: str, i: int = 0):
    from sources.base import Doc

    return Doc(
        content=f"# H\n{body}",
        source=file,
        category="c",
        language="en",
        chunk_index=i,
        title="H",
        links=[],
        tags=[],
        body=body,
        section=section,
    )


_ON = {"drop_boilerplate": True}


def _dropped(docs, policy=_ON):
    from sources.base import drop_wide_boilerplate

    kept = drop_wide_boilerplate(docs, policy)
    return [(d.source, d.body) for d in docs if d not in kept]


def test_a_block_repeated_across_half_the_files_is_dropped():
    nav = "see the index"
    docs = [_doc(f"f{i}.md", nav, "topic") for i in range(4)]
    docs += [_doc(f"f{i}.md", f"answer {i}", "topic", 1) for i in range(4)]
    assert [f for f, _ in _dropped(docs)] == [f"f{i}.md" for i in range(4)]


def test_the_only_carrier_of_its_section_stays():
    # hygiene that removes the answer is not hygiene: the exception saved six gold sections
    nav = "see the index"
    docs = [_doc(f"f{i}.md", nav, "topic") for i in range(4)]
    # f3.md holds nothing under `topic` but the shared block, so its chunk is the carrier
    docs += [_doc(f"f{i}.md", f"answer {i}", "topic", 1) for i in range(3)]
    assert [f for f, _ in _dropped(docs)] == ["f0.md", "f1.md", "f2.md"]


def test_the_rule_is_off_unless_the_variant_asks_for_it():
    nav = "see the index"
    docs = [_doc(f"f{i}.md", nav, "topic") for i in range(4)]
    docs += [_doc(f"f{i}.md", f"answer {i}", "topic", 1) for i in range(4)]
    assert _dropped(docs, {}) == []


def test_a_source_of_two_files_is_left_alone():
    # the same floor the coverage metric uses: a block cannot stand in most of two files
    nav = "see the index"
    docs = [_doc(f"f{i}.md", nav, "topic") for i in range(2)]
    docs += [_doc(f"f{i}.md", f"answer {i}", "topic", 1) for i in range(2)]
    assert _dropped(docs) == []


def test_a_source_is_replaced_in_one_transaction_or_not_at_all(monkeypatch):
    # the delete committed on its own, so the source stood empty while its embeddings ran
    import inspect

    from use_cases import index

    source = inspect.getsource(index._provision_source)
    assert "delete(DataChunk)" not in source, "the delete belongs with the insert that replaces"

    replace = inspect.getsource(index._replace_chunks)
    assert replace.index("delete(DataChunk)") < replace.index("session.add_all")
    assert replace.count("session.commit()") == 1, "one commit, so the pair is atomic"
    assert replace.index("embed_labelled") < replace.index("delete(DataChunk)"), (
        "embed first: the old rows must outlive the slow part"
    )


def test_a_tag_is_stored_in_the_alphabet_the_filter_takes():
    from sources.base import labels

    assert labels(["Java & JVM", "React", "react", "ba r@x", " "]) == ["java___jvm", "react", "ba_r_x"]


def test_a_source_of_several_categories_names_every_loose_file_before_cutting(tmp_path):
    from types import SimpleNamespace

    from sources.base import Base

    reader = SimpleNamespace(
        name="s",
        root=tmp_path,
        settings=SimpleNamespace(categories=["redis", "postgresql"], category_by_path={"docs/": "redis"}),
        rel_of=lambda f: str(f.relative_to(tmp_path)),
    )
    found = [tmp_path / "docs" / "a.md", tmp_path / "x.md", tmp_path / "y" / "z.md"]
    with pytest.raises(ValueError, match="2 files under no category_by_path"):
        Base._refuse_uncategorised(reader, found)
