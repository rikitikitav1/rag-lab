from use_cases import piece_join


def test_pieces_meet_with_a_blank_line_when_nothing_was_cut():
    whole, healed = piece_join.join(["## A\n\ntext", "## B\n\nmore"])

    assert whole == "## A\n\ntext\n\n## B\n\nmore" and healed == {"fences": 0, "tables": 0}


# a code block cut by the break: the second piece's opening fence goes, and the block is one again
def test_a_code_block_cut_by_the_break_is_one_block_again():
    whole, healed = piece_join.join(["text\n\n```\nfor i in x:\n", "```\n    print(i)\n```\n\nafter"])

    assert whole == "text\n\n```\nfor i in x:\n    print(i)\n```\n\nafter" and healed["fences"] == 1


def test_an_open_fence_with_no_code_after_it_is_closed():
    whole, healed = piece_join.join(["```\nx = 1", "## Next"])

    assert whole.startswith("```\nx = 1\n```") and healed["fences"] == 1


# a table cut by the break: a repeated header goes, a row read as a header stays a row
def test_a_table_cut_by_the_break_is_one_table_again():
    first = "## T\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    repeated, healed = piece_join.join([first, "| a | b |\n|---|---|\n| 3 | 4 |"])
    assert repeated == "## T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |" and healed["tables"] == 1

    promoted, _ = piece_join.join([first, "| 3 | 4 |\n|---|---|\n| 5 | 6 |"])
    assert promoted == "## T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"

    other, healed = piece_join.join([first, "| x | y | z |\n|---|---|---|\n| 1 | 2 | 3 |"])
    assert "| 2 |\n\n| x" in other and healed["tables"] == 0


# a publisher's chapter page: title and sections all h1; the sections and their subtree go one level down
def test_sections_at_the_title_s_level_go_under_the_title():
    page = "# Chapter 4. Health Probe\n\n# Problem\n\ntext\n\n## Liveness\n\n```\n# not a heading\n```\n\n# Discussion"
    text, moved = piece_join.one_title(page)

    assert text == (
        "# Chapter 4. Health Probe\n\n## Problem\n\ntext\n\n### Liveness\n\n```\n# not a heading\n```\n\n## Discussion"
    )
    assert moved == 3


def test_a_page_with_one_title_or_no_h1_is_left_alone():
    assert piece_join.one_title("# Title\n\n## A\n\n## B") == ("# Title\n\n## A\n\n## B", 0)
    assert piece_join.one_title("## A\n\n## B") == ("## A\n\n## B", 0)


# tables the structure says go on over a break are joined when only blank lines lie between; others stay apart
def test_tables_that_go_on_over_a_break_are_joined():
    markdown = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| a | b |\n|---|---|\n| 3 | 4 |\n\nText\n\n| x |\n|---|\n| 9 |"

    joined, count = piece_join.join_tables(markdown, [True, False, False])
    assert joined == "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nText\n\n| x |\n|---|\n| 9 |" and count == 1

    assert piece_join.join_tables(markdown, [False, False, False]) == (markdown, 0)
    assert piece_join.join_tables(markdown, [True]) == (markdown, 0)


def test_a_page_that_already_nests_its_sections_is_left_alone():
    page = "# Title\n\n## Part\n\n# Second title"

    assert piece_join.one_title(page) == (page, 0)


def _structure(*items):
    texts = [{"label": label, "prov": [{"page_no": page}]} for label, _, page in items]
    for item, (_, text, _) in zip(texts, items, strict=True):
        if text:
            item["text"] = text
    return {"texts": texts, "body": {"children": [{"$ref": f"#/texts/{i}"} for i in range(len(texts))]}}


# a paragraph cut by the page break is one again, a note between moves after it, and a hyphen joins the word
def test_a_paragraph_cut_by_the_page_break_is_one_again():
    structure = _structure(
        ("text", "Буферный кэш в памяти хранит страницы, прочитанные с", 5),
        ("footnote", "1 См. главу 9.", 5),
        ("text", "диска, и отдаёт их процессам по запросу без обра-", 6),
        ("text", "щения к файлу.", 7),
    )
    markdown = (
        "Буферный кэш в памяти хранит страницы, прочитанные с\n\n1 См. главу 9.\n\n"
        "диска, и отдаёт их процессам по запросу без обра-\n\nщения к файлу."
    )
    joined, count = piece_join.join_paragraphs(markdown, structure)
    assert count == 2
    assert joined == (
        "Буферный кэш в памяти хранит страницы, прочитанные с диска, "
        "и отдаёт их процессам по запросу без обращения к файлу.\n\n"
        "1 См. главу 9."
    )


# a float, a heading or a list item between, a lead-in colon, or a capital on the next page: two paragraphs stay two
def test_a_paragraph_is_not_joined_across_a_float_a_heading_a_colon_or_a_capital():
    cases = [
        (("text", "the cache keeps pages read from", 5), ("picture", None, 6), ("text", "disk and serves them", 6)),
        (("text", "the cache keeps pages read from", 5), ("section_header", "disk", 6)),
        (("text", "the cache keeps pages read from", 5), ("list_item", "disk and more", 6)),
        (("text", "the cache keeps these pages:", 5), ("text", "data pages and index pages", 6)),
        (("text", "the cache keeps pages read from", 5), ("text", "Disk pages are read", 6)),
        (("text", "the cache keeps pages read from", 5), ("text", "disk and serves them", 7)),
    ]
    for items in cases:
        markdown = "\n\n".join(text for _, text, _ in items if text)
        assert piece_join.join_paragraphs(markdown, _structure(*items)) == (markdown, 0)


# a margin note beside the column is no half of a paragraph: it is passed over and moves after the joined one
def test_a_margin_note_is_passed_over_and_never_joined():
    def at(label, text, page, left, right):
        return {"label": label, "text": text, "prov": [{"page_no": page, "bbox": {"l": left, "r": right}}]}

    items = [
        at("text", "the buffer cache keeps the pages read from", 5, 100, 400),
        at("text", "p. 211", 5, 420, 470),
        at("text", "disk and serves them", 6, 100, 400),
        at("text", "p. 298", 6, 420, 470),
        at("text", "p. 268", 7, 420, 470),
    ]
    structure = {"texts": items, "body": {"children": [{"$ref": f"#/texts/{i}"} for i in range(len(items))]}}
    markdown = "\n\n".join(item["text"] for item in items)
    joined, count = piece_join.join_paragraphs(markdown, structure)
    assert count == 1
    assert joined == "the buffer cache keeps the pages read from disk and serves them\n\np. 211\n\np. 298\n\np. 268"


# headings take their outline depth, the shallowest on `##`; past level six a heading is bold
def test_headings_take_their_outline_depth_and_level_seven_is_bold():
    def header(text, page):
        return ("section_header", text, page)

    structure = _structure(
        header("2. Installing Kafka", 48),
        header("Environment Setup", 48),
        header("Installing Java", 48),
        header("Installing ZooKeeper", 49),
        header("Not In The Outline", 50),
    )
    outline = [
        (0, "2. Installing Kafka", 48),
        (1, "Environment Setup", 48),
        (2, "Installing Java", 48),
        (2, "Installing ZooKeeper", 49),
        (1, "Installing a Kafka Broker", 54),
    ]
    markdown = (
        "## 2. Installing Kafka\n\n## Environment Setup\n\n## Installing Java\n\n```\n## not a heading\n```\n\n"
        "#### Installing ZooKeeper\n\n### Not In The Outline\n\n####### broker.id"
    )
    relevelled, count = piece_join.relevel(markdown, structure, outline)
    assert relevelled == (
        "## 2. Installing Kafka\n\n### Environment Setup\n\n#### Installing Java\n\n```\n## not a heading\n```\n\n"
        "#### Installing ZooKeeper\n\n### Not In The Outline\n\n**broker.id**"
    )
    assert count == 3


# a source that asks for it has its headings' section numbers outrank an outline that puts a record type on top
def test_a_section_number_outranks_a_flat_outline():
    structure = _structure(("section_header", "1.4 Procedures", 10), ("section_header", "1.5 Dir", 11))
    outline = [(1, "Procedures", 10), (0, "Dir", 11)]
    markdown = "## 1.4 Procedures\n\n## 1.4.22 FpFtruncate\n\n## 1.5 Dir"
    relevelled, _ = piece_join.relevel(markdown, structure, outline, by_number=True)
    assert relevelled == "## 1.4 Procedures\n\n### 1.4.22 FpFtruncate\n\n## 1.5 Dir"
    assert "## 1.4.22 FpFtruncate" in piece_join.relevel(markdown, structure, outline)[0]


# entities are decoded and counted; a paragraph that is only a pipe goes, a pipe inside a fence stays
def test_entities_are_decoded_and_lone_pipes_dropped():
    assert piece_join.decode_entities("=&gt; SELECT a &amp;&amp; b") == ("=> SELECT a && b", 3)
    assert piece_join.decode_entities("plain") == ("plain", 0)
    markdown = "```\nls |\n```\n\n|\n\n```\n|\n```\n\ntext"
    assert piece_join.drop_lone_pipes(markdown) == ("```\nls |\n```\n\n```\n|\n```\n\ntext", 1)


# with no outline a numbered heading takes its level from its number
def test_numbered_headings_level_by_number_with_no_outline():
    markdown = "#### 2.1 Processes\n\n## 2.1.3 Fork\n\n### Not numbered"
    assert piece_join.relevel(markdown, {}, [], by_number=True) == (
        "## 2.1 Processes\n\n### 2.1.3 Fork\n\n### Not numbered",
        2,
    )


# a dash the converter glued away at a line end comes back from the layer; a word the layer has whole stays
def test_a_dash_dropped_at_a_line_end_comes_back_from_the_layer():
    layer = "much the same\u2014\r\nthe only difference, the Berkeley Time-Sharing System, a mainframe"
    markdown = "much the samethe only difference, the Berkeley TimeSharing System, a mainframe"
    assert piece_join.restore_dashes(markdown, layer) == (
        "much the same\u2014the only difference, the Berkeley Time-Sharing System, a mainframe",
        2,
    )
    assert piece_join.restore_dashes("mainframe", "main-frame and mainframe") == ("mainframe", 0)


# a word left broken at its hyphenation is joined when the layer has it whole; a real compound stays
def test_a_word_left_broken_is_joined_from_the_layer():
    layer = "выявлять закономерности и long-lived"
    assert piece_join.join_broken_words("выявлять за - кономерности и long-lived", layer) == (
        "выявлять закономерности и long-lived",
        1,
    )


# an escaped list marker becomes a list item again; an escaped dash inside a line stays
def test_an_escaped_list_marker_is_unescaped():
    assert piece_join.unescape_bullets("\\- LoadBalancer\n  \\- nested\ntext \\- kept") == (
        "- LoadBalancer\n  - nested\ntext \\- kept",
        2,
    )


def test_an_underscore_escaped_outside_code_is_unescaped_and_code_keeps_its_own():
    markdown = "AT\\_STATX\\_SYNC = $0000 and `a\\_b` stay\n```\nx\\_y = 1\n```\nclone\\_flags"
    assert piece_join.unescape_underscores(markdown) == (
        "AT_STATX_SYNC = $0000 and `a\\_b` stay\n```\nx\\_y = 1\n```\nclone_flags",
        3,
    )


def test_a_picture_inlined_as_base64_becomes_an_address_over_its_pieces_pages():
    markdown = "text ![](data:image/jpeg;base64,/9j/AAA) more ![fig](data:image/png;base64,iVB)"
    assert piece_join.inline_pictures(markdown, (100, 110)) == (
        "text ![](picture:pages100-110-1) more ![fig](picture:pages100-110-2)",
        2,
    )


def test_a_word_split_by_a_space_is_joined_when_the_layer_has_it_whole_and_code_is_left_alone():
    layer = "the file was first a parentheses test\nint i = 0"
    markdown = "the fi le was fi rst a p arentheses test\n```\nfi le\n```\nint i = 0 and `fi le`"
    assert piece_join.join_split_words(markdown, layer) == (
        "the file was first a parentheses test\n```\nfi le\n```\nint i = 0 and `fi le`",
        3,
    )


def test_two_whole_words_or_a_number_are_not_joined_even_where_the_layer_glues_them():
    layer = "go backto the start, see 8Formally and hookinto; to go into it"
    assert piece_join.join_split_words("go back to the start, see 8 Formally and hook into", layer)[1] == 0


def test_a_pipe_entity_inside_a_table_cell_stays_an_escaped_pipe_and_prose_decodes():
    markdown = "a &#124; b &lt; c\n\n| op | name |\n|---|---|\n| &#124;&#124; | Boolean OR |"
    assert piece_join.decode_entities(markdown) == (
        "a | b < c\n\n| op | name |\n|---|---|\n| \\|\\| | Boolean OR |",
        4,
    )


def test_only_whole_entities_are_decoded_and_a_url_query_stays_as_written():
    markdown = "see &lt;b&gt; at http://a.org/?id=1&section=2 and &copy=x, code &amp;copy;"
    assert piece_join.decode_entities(markdown) == (
        "see <b> at http://a.org/?id=1&section=2 and &copy=x, code &copy;",
        3,
    )


def test_a_hyphen_inside_a_line_is_the_word_s_own_and_is_not_joined():
    # the layer as the word rules read it, its line-end hyphen marks already joined
    layer = "a wellknown thing, закономерность"
    assert piece_join.join_broken_words("a well-known thing, за - кономерность", layer) == (
        "a well-known thing, закономерность",
        1,
    )


def test_a_number_or_two_whole_words_across_a_spaced_hyphen_stay_apart():
    layer = "layer N2 and WordPressbased and закономерность"
    markdown = "layer N - 2, a WordPress - based server, за - кономерность"
    assert piece_join.join_broken_words(markdown, layer + " WordPress based") == (
        "layer N - 2, a WordPress - based server, закономерность",
        1,
    )


def test_an_identifier_broken_after_its_underscore_is_joined():
    assert piece_join.join_broken_words("home of user_- name.", "home of user_name.") == ("home of user_name.", 1)
