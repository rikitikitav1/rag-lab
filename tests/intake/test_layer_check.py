from use_cases import layer_check


def _totals(rows):
    out = {}
    for _, found in rows:
        for name, n in found.items():
            out[name] = out.get(name, 0) + n
    return {k: v for k, v in out.items() if v}


def test_each_page_gets_the_markdown_that_holds_its_words():
    layers = ["alpha beta gamma delta", "epsilon zeta eta theta"]
    got = layer_check.page_slices("alpha beta gamma delta\n\nepsilon zeta eta theta", layers, 7)
    assert [(g["page"], g["start"]) for g in got] == [(7, 0), (8, 24)]


def test_word_defects_are_counted_against_the_layer_and_code_is_left_out():
    layers = ["the file was written separately and the lock-free queue\nsep￾\r\narate words here"]
    markdown = "the fi le was written separatelyand the lockfree queue\n\n```\nfi le\n```\n\nseparate words here"
    totals = _totals(layer_check.check_document(markdown, layers, 1))
    assert totals == {"split": 1, "glued": 1, "run_together": 1}


def test_escapes_entities_and_wrappers_count_in_prose_only_and_jsx_is_not_a_wrapper():
    layers = ["x"]
    markdown = "a \\_b &lt; c <span class='n'>d</span> <div>e</div>\n\n`f\\_g &lt;`"
    totals = _totals(layer_check.check_document(markdown, layers, 1, trust_words=False))
    assert totals == {"escapes": 1, "entities": 1, "html_tags": 2}


def test_a_lone_pipe_counts_once_on_the_page_its_line_starts_and_a_cut_row_is_no_pipe():
    layers = ["one two three four", "five six seven eight"]
    markdown = "one two three four\n\n|\n\n| five | six |\n|---|---|\n| seven | eight |"
    assert _totals(layer_check.check_document(markdown, layers, 1, trust_words=False)).get("lone_pipes") == 1


def test_a_page_whose_words_are_missing_from_the_markdown_is_text_lost():
    words = " ".join(f"word{n}" for n in range(30))
    rows = layer_check.check_document("<!-- formula-not-decoded -->", [words], 1, trust_words=False)
    assert rows[0][1]["text_lost"] == 1 and rows[0][1]["formulas"] == 1


def test_word_checks_are_skipped_where_the_layer_is_not_trusted():
    rows = layer_check.check_document("the fi le", ["the file"], 1, trust_words=False)
    assert not any(rows[0][1][name] for name in layer_check.WORD_CHECKS)
    assert layer_check.defects(rows[0][1]) == 0


def test_a_long_glued_token_is_split_in_linear_time():
    import time

    vocab = {"ab", "a", "b"}
    started = time.monotonic()
    assert layer_check._splits_into("ab" * 40, vocab) is False
    assert layer_check._splits_into("ababab", vocab) is True
    assert time.monotonic() - started < 1


def test_a_picture_address_with_escaped_brackets_in_its_caption_is_left_out_of_prose():
    rows = layer_check.check_document("![Zhang et al. \\[2010\\]](picture:p1-1) text", ["text"], 1, trust_words=False)
    assert rows[0][1]["escapes"] == 0


def test_a_pipe_escaped_inside_a_table_row_is_no_escape():
    markdown = "| a \\| b | c |\n|---|---|\n\nprose \\| here"
    totals = _totals(layer_check.check_document(markdown, ["a b c prose here"], 1, trust_words=False))
    assert totals == {"escapes": 1}


def test_a_caption_heading_and_a_repeated_running_head_are_suspect_and_a_chapter_title_is_not():
    layers = [f"{n} Computer Networks\nbody text of page {n}" for n in (88, 90, 92)]
    heads = "#### 88 Computer Networks\n\n#### 90 Computer Networks"
    markdown = f"## 2 Computer Networks\n\ntext\n\n{heads}\n\n###### Figure 2.6 Web applets\n\ntext"
    kinds = [kind for _, kind in layer_check.heading_defects(markdown, layers)]
    assert kinds == ["running_headings", "running_headings", "caption_headings"]


# a heading line inside a code block is code, as the repairs read it, and the caption words are the repair's own
def test_a_caption_heading_inside_code_is_not_counted():
    found = layer_check.heading_defects("```\n# Table 1 in a listing\n```\n\n# Table 2 made a heading\n", [])
    assert [kind for _, kind in found] == ["caption_headings"]


# the checker reads a fence as the repairs do: an indented fence is code, and a heading inside it is not counted
def test_a_heading_inside_an_indented_fence_is_code_to_the_checker_as_to_the_repairs():
    from use_cases import layer_check, piece_join

    markdown = "## Figure 1 a caption\n\n  ```\n## Figure 2 not a heading\n  ```\n"
    assert [kind for _, kind in layer_check.heading_defects(markdown, [])] == ["caption_headings"]
    assert piece_join.demote_caption_headings(markdown)[1] == 1
