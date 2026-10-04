import pytest
from sources.base import Doc
from sources.declaration import SourceFile
from sources.versioned import merge_versions, version_neutral


def _doc(body, section="A > B"):
    return Doc(
        content=body,
        body=body,
        source="s/a.md",
        category="postgresql",
        language="en",
        chunk_index=0,
        title=None,
        links=[],
        tags=[],
        section=section,
    )


def test_the_same_text_in_two_versions_is_one_row_with_both():
    docs, merged = merge_versions([("18", [_doc("same"), _doc("new in 18")]), ("17", [_doc("same")])], "postgresql")

    assert [(d.body, d.versions) for d in docs] == [("same", ["18", "17"]), ("new in 18", ["18"])]
    assert merged == 1


def test_copies_inside_one_version_stay_rows_and_meet_their_kth_copy():
    docs, _ = merge_versions([("18", [_doc("x"), _doc("x")]), ("17", [_doc("x")])], "postgresql")

    assert [d.versions for d in docs] == [["18", "17"], ["18"]]


def test_the_version_in_a_heading_or_the_text_does_not_split_a_merge():
    newest = _doc("PostgreSQL 18 locks rows", section="PostgreSQL 18 > Locks")
    older = _doc("PostgreSQL 17 locks rows", section="PostgreSQL 17 > Locks")
    docs, _ = merge_versions([("18", [newest]), ("17", [older])], "postgresql")

    assert len(docs) == 1 and docs[0].versions == ["18", "17"] and docs[0].body == "PostgreSQL 18 locks rows"


def test_a_bare_number_is_not_a_version():
    assert version_neutral("the limit is 17", "postgresql", ["18", "17"]) == "the limit is 17"
    assert version_neutral("PostgreSQL 17 adds it", "postgresql", ["18", "17"]) == "PostgreSQL {version} adds it"


def _file(**kw):
    return SourceFile(name="pg", language="en", licence="PostgreSQL", categories=["postgresql"], **kw)


def test_a_git_source_names_a_ref_per_version_and_a_folder_source_a_folder():
    _file(git={"repo": "https://x/pg"}, versions={"18": {"ref": "REL_18_STABLE"}})
    _file(folder="datasets/pg18", versions={"18": {"folder": "datasets/pg18"}})
    with pytest.raises(ValueError, match="name no folder"):
        _file(git={"repo": "https://x/pg"}, versions={"18": {"folder": "datasets/pg18"}})
    with pytest.raises(ValueError, match="exactly one"):
        _file(git={"repo": "https://x/pg"}, versions={"18": {}})


def test_versions_are_the_maps_own_newest_first():
    from sources import files

    ok = _file(git={"repo": "https://x/pg"}, versions={"18": {"ref": "a"}, "17": {"ref": "b"}})
    files._refuse_unmapped({"pg": ok})
    with pytest.raises(ValueError, match="not listed"):
        files._refuse_unmapped({"pg": _file(git={"repo": "https://x/pg"}, versions={"9": {"ref": "a"}})})
    with pytest.raises(ValueError, match="newest first"):
        files._refuse_unmapped(
            {"pg": _file(git={"repo": "https://x/pg"}, versions={"17": {"ref": "b"}, "18": {"ref": "a"}})}
        )


def test_a_versions_chunks_are_spelled_by_the_source_not_by_its_checkout(tmp_path):
    from sources.base import Base

    root = tmp_path / "pg@17"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.md").write_text("# A\n\n## B\n\ntext about locks\n")
    settings = _file(git={"repo": "https://x/pg"}, versions={"18": {"ref": "a"}, "17": {"ref": "b"}})
    import config

    docs = Base(root, settings, name="pg", version="17").documents(config.settings.corpus.policy("clean_1024"))

    assert docs and all(d.source.startswith("pg/docs/") and "@" not in d.source for d in docs)
    assert all(d.versions == ["17"] for d in docs)


def test_a_folder_source_with_versions_names_its_newest_folder_as_its_own():
    _file(folder="datasets/pg18", versions={"18": {"folder": "datasets/pg18"}, "17": {"folder": "datasets/pg17"}})
    with pytest.raises(ValueError, match="newest version's folder"):
        _file(folder="datasets/pg", versions={"18": {"folder": "datasets/pg18"}})


def test_a_declared_version_that_did_not_clone_fails_its_source(monkeypatch):
    from sources import factory

    settings = _file(git={"repo": "https://x/pg"}, versions={"18": {"ref": "a"}, "17": {"ref": "b"}})
    monkeypatch.setattr(factory, "_accepted", lambda *a: ({"pg": settings}, {"pg": {"source": "pg", "raw": {}}}))
    monkeypatch.setattr(factory, "provision", lambda specs: {name: None for name, _, _ in specs})
    with pytest.raises(LookupError, match="did not clone"):
        list(factory.sources())


# a converted row is read from its raw folder; a row the index cannot read is said and left out
def test_the_index_reads_a_converted_row_from_its_raw_folder_and_leaves_out_what_it_cannot_read(monkeypatch):
    from sources import factory
    from sources.converted import Converted
    from sources.declaration import Declaration

    book = Declaration(name="book", folder="inbox/book")
    site = Declaration(name="site", urls=["https://a/b.html"])
    rows = {"book": {"source": "book", "raw": {"folder": "raw/book_1"}}, "site": {"source": "site", "raw": {}}}
    monkeypatch.setattr(factory, "_accepted", lambda *a: ({"book": book, "site": site}, rows))
    monkeypatch.setattr(factory, "provision", lambda specs: {})
    built = list(factory.sources())
    assert len(built) == 1 and isinstance(built[0], Converted) and str(built[0].root) == "raw/book_1"


# a row onboarding marked as markdown is read from its own tree by its declared reader, with the row's name
def test_the_index_reads_an_onboarded_markdown_row_from_its_tree_by_its_reader(monkeypatch):
    from sources import factory
    from sources.converted import Converted
    from sources.declaration import Declaration

    family = Declaration(name="repos", git_family={"base_url": "https://github.com/x", "repos": ["a", "b"]})
    rows = {
        "a": {"source": "repos", "raw": {"root": "datasets/raw_sources/_fetched/a/repo", "root_kind": "tree"}},
        "b": {"source": "repos", "raw": {}},
    }
    monkeypatch.setattr(factory, "_accepted", lambda *a: ({"repos": family}, rows))
    cloned = []
    monkeypatch.setattr(factory, "provision", lambda specs: cloned.extend(specs) or {n: None for n, *_ in specs})
    built = list(factory.sources())
    assert [(s.name, str(s.root)) for s in built] == [("a", "datasets/raw_sources/_fetched/a/repo")]
    assert not isinstance(built[0], Converted)
    assert [name for name, *_ in cloned] == ["b"], "the family's other row is still cloned"


# the tree onboarding fetched sits in a folder named `repo`; its chunks name the row, as the veto and the marks do
def test_a_family_row_onboarded_from_its_fetched_tree_spells_its_chunks_by_the_row(monkeypatch, tmp_path):
    import config
    from sources import factory
    from sources.declaration import Declaration

    tree = tmp_path / "_fetched" / "go-interview-questions" / "repo"
    tree.mkdir(parents=True)
    (tree / "README.md").write_text("# Go\n\n## Channels\n\nA channel carries values between goroutines.\n")
    family = Declaration(
        name="interview",
        tags=["interview"],
        tag_from_name="-interview-questions",
        git_family={"base_url": "https://github.com/x", "repos": ["go-interview-questions"]},
    )
    rows = {"go-interview-questions": {"source": "interview", "raw": {"root": str(tree), "root_kind": "tree"}}}
    monkeypatch.setattr(factory, "_accepted", lambda *a: ({"interview": family}, rows))
    monkeypatch.setattr(factory, "provision", lambda specs: {})
    [reader] = factory.sources()
    docs = reader.documents(config.settings.corpus.policy("clean_1024"))

    assert docs and {d.source for d in docs} == {"go-interview-questions/README.md"}


# the grader reads a chunk by `source#chunk_index`; an older version's own text must not answer to the newest's address
def test_texts_of_two_versions_under_one_file_keep_apart_addresses():
    def at(body, index):
        doc = _doc(body)
        doc.chunk_index = index
        return doc

    docs, _ = merge_versions(
        [("18", [at("shared", 0), at("only in 18", 1)]), ("17", [at("shared", 0), at("only in 17", 1)])], "postgresql"
    )

    addresses = [(d.source, d.chunk_index) for d in docs]
    assert len(set(addresses)) == len(docs) == 3
    assert [d.chunk_index for d in docs if d.versions[0] == "18"] == [0, 1], "the newest keeps its own numbers"
