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
