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
    # the lead before the first chapter joins it rather than breaching coverage alone
    lead = raw_quality.section_rows("# Book\n\nA lead paragraph.\n\n## One\n\nFirst chapter text here.", "b.md")
    assert [r["section"] for r in lead] == ["Book > One"] and "section_coverage.min" not in lead[0]["breached"]
    # only the root itself is a lead: two top-level chapters stay two
    assert list(raw_quality._with_lead({"A": [1], "B": [2]})) == ["A", "B"]
    assert raw_quality._with_lead({"Book": [1], "Book > One": [2]}) == {"Book > One": [1, 2]}
    assert rows[1]["words"] > rows[0]["words"] > 0
    piece = raw_quality.unit_row(markdown, None, {"file": "book.md", "pages": None}, {"engine": None}, 0.0)
    signal_rows = {**piece, **rows[0]}
    assert set(raw_quality.SIGNALS) <= set(signal_rows) and "chunker_verdict" in rows[0]


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
