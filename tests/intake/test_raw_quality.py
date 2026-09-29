from use_cases import raw_quality


def test_an_output_that_matches_its_layer_breaches_nothing():
    layer = "Git stores snapshots, not differences. Every commit is a snapshot of the whole tree."
    signals = raw_quality.conversion_signals("## Snapshots\n\n" + layer, layer)

    assert signals["layer_f1"] > 0.95
    assert 0.9 < signals["output_share"] < 1.2
    assert raw_quality.conversion_breaches(signals) == []


# half a page came back: the words are right, but the share of the layer says what was lost
def test_a_truncated_output_breaches_the_share_band():
    layer = " ".join(f"word{i} text" for i in range(200))
    signals = raw_quality.conversion_signals(" ".join(f"word{i} text" for i in range(50)), layer)

    assert "output_share.band" in raw_quality.conversion_breaches(signals)


# an OCR that puts Latin letters into Cyrillic words is read on the output alone, with no layer to lean on
def test_mixed_script_is_read_without_a_layer():
    signals = raw_quality.conversion_signals("Kоманда сoздаёт вeтку и пeреключает рабочую копию", None)

    assert signals["layer_f1"] is None
    assert "mixed_script.max" in raw_quality.conversion_breaches(signals)


def test_a_row_carries_the_unit_the_arm_every_signal_and_what_it_breached():
    row = raw_quality.unit_row(
        "## Title\n\nSome body text of the page.",
        None,
        {"file": "a.pdf", "pages": [1, 1]},
        {"engine": "mineru", "settings_sha256": "x"},
        1.5,
    )

    assert row["file"] == "a.pdf" and row["engine"] == "mineru" and row["seconds"] == 1.5
    assert {"output_words", "mixed_script", "layer_f1", "breached"} <= set(row)
    assert "chunker_verdict" not in row


# a piece cut alone took its own first heading for the root; a file is cut whole and read a chapter a row
def test_a_file_is_cut_whole_and_read_a_chapter_a_row():
    markdown = "# Book\n\n## One\n\nFirst chapter text here.\n\n## Two\n\nSecond chapter text, a bit longer here."
    rows = raw_quality.section_rows(markdown, "book.md")

    assert [r["section"] for r in rows] == ["Book > One", "Book > Two"]
    assert rows[1]["words"] > rows[0]["words"] > 0
    # the lead is the root alone: coverage is not read there, nor diluted in chapter one
    lead = raw_quality.section_rows("# Book\n\nA lead paragraph.\n\n## One\n\nFirst chapter text here.", "b.md")
    assert [r["section"] for r in lead] == ["Book", "Book > One"]
    assert lead[0]["coverage_by_shape"] and not lead[1]["coverage_by_shape"]
    assert not any("section_coverage.min" in r["breached"] for r in lead)
    piece = raw_quality.unit_row(markdown, None, {"file": "book.md", "pages": None}, {"engine": None}, 0.0)
    signal_rows = {**piece, **rows[0]}
    assert set(raw_quality.SIGNALS) <= set(signal_rows) and "chunker_verdict" in rows[0]


# a short file whose only chapter is its root breached coverage by shape: cheatsheets and redis-doc read bad for it
def test_a_file_that_is_its_root_alone_is_not_read_for_coverage():
    rows = raw_quality.section_rows("### Format\n\n```\n\\033[#m\n```\n\nSome words about the format here.", "ansi.md")

    assert [r["section"] for r in rows] == ["Format"] and rows[0]["coverage_by_shape"]
    assert "section_coverage.min" not in rows[0]["breached"] and rows[0]["chunker_verdict"] != "broken"


# the band against the layer was read on Docling, so another engine's layer signals are kept but not judged
def test_layer_gates_judge_only_the_engines_the_band_was_read_on():
    layer = " ".join(f"word{i} text" for i in range(200))
    signals = raw_quality.conversion_signals(" ".join(f"word{i} text" for i in range(50)), layer)

    assert "output_share.band" in raw_quality.conversion_breaches(signals, "docling")
    assert raw_quality.conversion_breaches(signals, "mineru") == []


# a raw report cuts through the index's own door, so a variant with another chunker is read with that chunker
def test_the_raw_gates_cut_with_the_variants_chunker(monkeypatch):
    import config

    seen = []

    def spy(content, root, policy, file):
        seen.append(policy)
        return iter([])

    monkeypatch.setattr(raw_quality, "cuts_of", spy)
    raw_quality.chunker_gates("## A\n\ntext", "a.md")

    assert seen == [config.settings.corpus.policy(config.settings.corpus.variant)]


# code with angle brackets is text: only a real tag is markup, so a Rails template keeps its words
def test_angle_brackets_in_code_are_not_markup():
    from use_cases import raw_quality

    text = raw_quality.plain("```\n<%= link_to 'Home', root_path %>\nif a < b and c > d:\n```\n<b>bold</b>")
    assert "link" in text and "root" in text and "a < b" in text and "bold" in text and "<b>" not in text


# a file read against itself: outline titles found as headings, an open fence and one-line code counted
def test_a_file_is_checked_against_its_own_outline_and_fences():
    markdown = "# Chapter 2 Getting Started\n\n## 2.1 Installing\n\ntext\n\n```\nls\n```\n\n```\na\nb\n```\n"
    markdown += "\n```\nopen\n"
    check = raw_quality.self_check(markdown, ["Getting Started", "2.1 Installing", "2.2 Running", "3"])
    assert check == {
        "outline": 3,
        "outline_found": 2,
        "fences_unbalanced": 1,
        "code_blocks": 2,
        "code_one_line": 1,
    }


def test_a_second_reading_losing_cells_is_taken_only_within_the_source_s_slack():
    first, second = {"layer_f1": 0.69, "table_cells": 44}, {"layer_f1": 0.97, "table_cells": 42}

    assert not raw_quality.better_reading(first, second)
    assert raw_quality.better_reading(first, second, 0.05)
    assert not raw_quality.better_reading(first, {**second, "table_cells": 40}, 0.05)


# a book's matter is left out of the report's sections, as the index and the questions leave it out
def test_the_report_does_not_judge_a_books_matter():
    from use_cases import raw_quality

    body = " ".join(["word"] * 300)
    markdown = f"# Book\n\n## Index\n\nalpha, 12\n\nbeta, 14\n\n## Locks\n\n{body}\n"
    sections = {row["section"] for row in raw_quality.section_rows(markdown, "book.pdf")}
    assert not any("Index" in (s or "") for s in sections) and any("Locks" in (s or "") for s in sections)
