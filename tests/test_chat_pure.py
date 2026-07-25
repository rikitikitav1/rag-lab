from use_cases import chat


def _row(src, vector_rank=1, keyword_rank=None, vector_distance=0.1, score=0.5):
    return ("content", src, "cat", 0, vector_rank, keyword_rank, vector_distance, score)


def test_is_ignored_source():
    assert chat.is_ignored_source("notes/index.md") is True
    assert chat.is_ignored_source("notes/real.md") is False


def test_take_sources_dedups_and_drops_ignored():
    rows = [_row("a.md"), _row("a.md"), _row("notes/index.md"), _row("b.md")]
    assert [s.source for s in chat.take_sources(rows)] == ["a.md", "b.md"]


def test_take_sources_rounds_numbers():
    s = chat.take_sources([_row("a.md", vector_distance=0.123456, score=0.987654)])[0]
    assert s.vector_distance == 0.123
    assert s.score == 0.988
