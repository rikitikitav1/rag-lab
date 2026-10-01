from types import SimpleNamespace

from use_cases import section_export


def _doc(source, section, body, versions=()):
    return SimpleNamespace(source=source, section=section, body=body, content=body, versions=list(versions))


# a section is keyed as its chunks are, its chunks' text joined; a path that comes back later is refused, a leaf is not
def test_sections_are_keyed_as_chunks_and_a_repeated_path_is_refused():
    docs = [
        _doc("git/ch3.md", "Pro Git > Branching > Summary", "one two", ["2"]),
        _doc("git/ch3.md", "Pro Git > Branching > Summary", "three", ["2"]),
        _doc("git/ch4.md", "Pro Git > Server > Summary", "four five six"),
        _doc("git/ch4.md", "Pro Git > Server > Examples", "a"),
        _doc("git/ch4.md", "Pro Git > Server > Other", "b"),
        _doc("git/ch4.md", "Pro Git > Server > Examples", "c"),
        _doc("git/intro.md", None, "no heading at all"),
    ]
    out = section_export.export(docs)

    kept = {(r["file"], r["section"]): r for r in out["sections"]}
    assert set(kept) == {
        ("git/ch3.md", "Pro Git > Branching > Summary"),
        ("git/ch4.md", "Pro Git > Server > Summary"),
        ("git/ch4.md", "Pro Git > Server > Other"),
    }
    first = kept[("git/ch3.md", "Pro Git > Branching > Summary")]
    assert first["text"] == "one two\n\nthree" and first["words"] == 3 and first["versions"] == ["2"]
    assert first["chapter"] == "Pro Git > Branching"
    assert {(r["file"], r["section"]) for r in out["refused"]} == {
        ("git/ch4.md", "Pro Git > Server > Examples"),
        ("git/intro.md", None),
    }


# the ceiling spreads by words with no chapter over its own ceiling, and a small source gets what its chapters hold
def test_the_quota_spreads_by_words_under_both_ceilings():
    rows = [{"chapter": "A", "words": 9000}, {"chapter": "B", "words": 900}, {"chapter": "C", "words": 100}]

    assert section_export.chapter_quota(rows, per_source=100, per_chapter=10) == {"A": 10, "B": 10, "C": 10}
    assert section_export.chapter_quota(rows, per_source=12, per_chapter=10) == {"A": 10, "B": 2, "C": 0}
    assert sum(section_export.chapter_quota(rows[:1], per_source=100, per_chapter=10).values()) == 10


# a versioned source hands an older version's own texts after the newest one's; that is its own stream, not a repeat
def test_an_older_versions_text_is_its_own_section_not_a_repeat():
    docs = [
        _doc("db/aql.md", "AQL > Operators", "new text", ["3.12"]),
        _doc("db/aql.md", "AQL > Functions", "shared", ["3.12", "3.11"]),
        _doc("db/aql.md", "AQL > Operators", "old text", ["3.11"]),
    ]
    out = section_export.export(docs)

    assert not out["refused"]
    rows = {(r["section"], tuple(r["versions"])): r["text"] for r in out["sections"]}
    assert rows == {
        ("AQL > Operators", ("3.12",)): "new text",
        ("AQL > Functions", ("3.12", "3.11")): "shared",
        ("AQL > Operators", ("3.11",)): "old text",
    }


# the index's own chunks joined into blocks: a subsection starts one once the block is half full, the ceiling always
def test_blocks_join_chunks_and_break_at_a_subheading_or_the_ceiling(monkeypatch):
    monkeypatch.setattr(section_export, "BLOCK_CHARS", 100)
    small, half = "a" * 20, "b" * 55
    assert section_export.blocks([small, "### Sub\n" + small]) == [small + "\n\n### Sub\n" + small]
    assert section_export.blocks([half, "### Sub\n" + small]) == [half, "### Sub\n" + small]
    assert section_export.blocks([half, half]) == [half, half]
