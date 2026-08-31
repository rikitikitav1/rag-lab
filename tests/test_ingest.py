import ingest
import pytest


def test_chunk_markdown_empty():
    assert ingest.chunk_markdown("") == []
    assert ingest.chunk_markdown("   \n ") == []


def test_chunk_markdown_keeps_h1_on_each_section():
    md = "# Title\nintro\n## A\nbody a\n## B\nbody b"
    chunks = ingest.chunk_markdown(md)
    assert chunks[0].startswith("# Title\nintro")
    assert any(c.startswith("# Title\n## A") for c in chunks)
    assert any(c.startswith("# Title\n## B") for c in chunks)


def test_split_by_size_short_is_untouched():
    assert ingest.split_by_size("short text") == ["short text"]


def test_split_by_size_respects_max():
    long = "a" * (ingest.MAX_CHUNK_SIZE * 2 + 50)
    parts = ingest.split_by_size(long)
    assert len(parts) >= 2
    assert all(len(p) <= ingest.MAX_CHUNK_SIZE for p in parts)


def test_split_by_size_prefers_paragraph_boundary():
    para = "x" * (ingest.MAX_CHUNK_SIZE - 10)
    parts = ingest.split_by_size(para + "\n\n" + para)
    assert parts == [para, para]


@pytest.mark.parametrize(
    "path, expected",
    [
        ("databases/postgresql/locks.md", "databases.postgresql.locks"),
        ("foo/ba r@x.md", "foo.ba_r_x"),
        ("single.md", "single"),
    ],
)
def test_path_to_category(path, expected):
    assert ingest.path_to_category(path) == expected


def _doc(file: str, body: str, section: str, i: int = 0):
    from sources.base import Doc

    return Doc(
        content=f"# H\n{body}", source=file, category="c", language="eng",
        chunk_index=i, title="H", links=[], tags=[], body=body, section=section,
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
    # hygiene that removes the answer is not hygiene: on this corpus the exception saved
    # six sections and all six were the gold of a veto question
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


def test_the_legacy_cut_honours_the_ceiling_its_variant_declares():
    # it used to fall through to the module constant, so `baseline` declared a ceiling
    # nothing read and `ingestion.chunk_max_size` decided the frozen variant's cut instead
    from ingest import chunk_markdown

    body = "x" * 900
    content = f"# T\n{body}\n## S\n{body}"
    assert all(len(c) <= 300 for c in chunk_markdown(content, ceiling=300))
    assert any(len(c) > 300 for c in chunk_markdown(content, ceiling=2000))
