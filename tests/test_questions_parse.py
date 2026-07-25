from api.v1 import questions


def test_parse_basic_and_marked_sources():
    rows = questions._parse("What is X?\nHow does Y work? | src1, src2")
    assert rows[0][1] == "What is X?"
    assert rows[0][2] == []
    assert rows[1][1] == "How does Y work?"
    assert rows[1][2] == ["src1", "src2"]


def test_parse_skips_comments_and_blanks():
    rows = questions._parse("# comment\n\n   \nreal question")
    assert [r[1] for r in rows] == ["real question"]


def test_parse_none_marker_means_no_sources():
    assert questions._parse("q | NONE")[0][2] == []


def test_parse_dedups_by_question_hash():
    rows = questions._parse("same q\nsame q\nsame q | x")
    assert len(rows) == 1


def test_text_hash_is_stable_and_distinct():
    assert questions._text_hash("abc") == questions._text_hash("abc")
    assert questions._text_hash("abc") != questions._text_hash("abd")
