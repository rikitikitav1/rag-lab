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


# two pieces of one chapter level its sections alike, the one without the chapter's own heading too
def test_every_piece_of_a_file_levels_by_the_whole_outline():
    outline = [(0, "Chapter 3", 100), (1, "Pages", 101), (1, "Tuples", 160)]
    first = _structure(("section_header", "Chapter 3", 100), ("section_header", "Pages", 101))
    second = _structure(("section_header", "Tuples", 160))

    assert piece_join.relevel("## Chapter 3\n\n## Pages", first, outline)[0] == "## Chapter 3\n\n### Pages"
    assert piece_join.relevel("## Tuples", second, outline)[0] == "### Tuples"


# a source that asks for it has its headings' section numbers outrank an outline that puts a record type on top
def test_a_section_number_outranks_a_flat_outline():
    structure = _structure(("section_header", "1.4 Procedures", 10), ("section_header", "1.5 Dir", 11))
    outline = [(1, "Procedures", 10), (0, "Dir", 11)]
    markdown = "## 1.4 Procedures\n\n## 1.4.22 FpFtruncate\n\n## 1.5 Dir"
    relevelled, _ = piece_join.relevel(markdown, structure, outline, by_number=True)
    assert relevelled == "## 1.4 Procedures\n\n### 1.4.22 FpFtruncate\n\n## 1.5 Dir"
    assert "### 1.4.22 FpFtruncate" in piece_join.relevel(markdown, structure, outline)[0]


# a heading the outline lacks moves with the outline's heading above it, so a section and its subsection stay apart
def test_a_heading_the_outline_lacks_moves_with_the_one_above_it():
    outline = [(0, "17 Distributed transactions", 745), (1, "17.4 Concurrency control", 758)]
    structure = _structure(("section_header", "17.4 Concurrency control", 758))
    markdown = "## 17.4 Concurrency control\n\n### 17.4.1 Locking"
    relevelled = piece_join.relevel(markdown, structure, outline)
    assert relevelled == ("### 17.4 Concurrency control\n\n#### 17.4.1 Locking", 2)


# a heading the outline raises leaves one it lacks where it was, so a label under it never reaches a chapter's level
def test_a_heading_the_outline_lacks_never_climbs():
    outline = [(0, "Chapter 30: Delete", 90), (1, "Section 30.1: Delete using join", 90)]
    structure = _structure(("section_header", "Section 30.1: Delete using join", 90))
    markdown = "#### Section 30.1: Delete using join\n\n#### OUTPUT"
    relevelled = piece_join.relevel(markdown, structure, outline)
    assert relevelled == ("### Section 30.1: Delete using join\n\n#### OUTPUT", 1)


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


def test_a_caption_made_a_heading_goes_back_to_a_line_outside_code():
    code = "```\n# Table 1 stays\n```"
    markdown = f"###### Figure 2.6 Web applets\n\n#### Таблица 8.4. Символьные типы\n\n## Tables\n\n{code}"
    assert piece_join.demote_caption_headings(markdown) == (
        f"Figure 2.6 Web applets\n\nТаблица 8.4. Символьные типы\n\n## Tables\n\n{code}",
        2,
    )


def test_a_running_head_made_a_heading_is_dropped_after_its_first_time_and_code_is_left_alone():
    layer = "\f".join(f"{n} - 02: variables\r\nbody of page {n}" for n in (50, 52, 54))
    code = "```\n# 54 - 02: variables\n```"
    markdown = f"## 50 - 02: variables\n\ntext\n\n## 52 - 02: variables\n\n{code}\n\n## Exercise 1"
    assert piece_join.drop_running_headings(markdown, layer) == (
        f"## 50 - 02: variables\n\ntext\n\n\n{code}\n\n## Exercise 1",
        1,
    )


# a running head repeated first in a later piece is only seen from the whole file, and the whole file drops it
def test_a_running_head_first_seen_in_a_later_piece_is_dropped_from_the_whole_file():
    from config import settings
    from job_handlers import reading

    layers = ["Chapter 3 Paging\nbody a", "Chapter 3 Paging\nbody b", "Chapter 3 Paging\nbody c", "Chapter 3 Paging\nd"]
    first = piece_join.drop_running_headings("## Chapter 3 Paging\n\ntext a", "\f".join(layers[:2]))[0]
    second = piece_join.drop_running_headings("## Chapter 3 Paging\n\ntext c", "\f".join(layers[2:]))[0]
    rule = settings.intake.route.model_copy(update={"drop_running_headings": True})

    whole, healed = reading.whole_file([first, second], ["text layer", "text layer"], rule, layers)

    assert whole.count("## Chapter 3 Paging") == 1 and healed["running_heads"] == 1


# a man page's title comes back over its NAME from the page's first name, so two pages' NAMEs stop being one section
def test_a_man_page_gets_its_title_back_from_its_name_line():
    from use_cases.piece_join import man_page_titles

    text = ("## SEE ALSO\n\nopen(2)\n\n## NAME\n\ndup, dup2 - duplicate a descriptor\n\n"
            "## SYNOPSIS\n\nnewd = dup(oldd)\n\n"
            "## NAME\n\nexecve - execute a file\n")
    out, count = man_page_titles(text)
    assert count == 2
    assert "# dup\n\n## NAME\n\ndup, dup2 - duplicate a descriptor" in out and "# execve\n\n## NAME" in out
    assert man_page_titles("## Names of things\n\ntext\n") == ("## Names of things\n\ntext\n", 0)
    assert man_page_titles("## NAME\n\nSYNOPSIS\n\nx\n")[1] == 0, "a NAME that lost its line to the scan stays as it is"


# a dash Docling prints as a hyphen comes back from the layer; a pair the layer itself hyphenates stays a hyphen
def test_a_dash_printed_as_a_hyphen_comes_back_from_the_layer():
    from use_cases.piece_join import restore_dashes

    layer = "appealing—it is, pages 511–515, a well-known and a well—known slip"
    out, count = restore_dashes("appealing-it is, pages 511-515, a well-known", layer)
    assert out == "appealing—it is, pages 511–515, a well-known" and count == 2


# an identifier the layer wraps after its underscore is joined back; two words the layer sets apart stay apart
def test_an_identifier_wrapped_after_its_underscore_is_joined_back():
    from use_cases.piece_join import join_wrapped_identifiers

    layer = "olist_order_reviews_\r\ndataset.csv and review_\r\ncomment_message, a_ b"
    out, count = join_wrapped_identifiers("файл olist_order_reviews_ dataset.csv, review_ comment_message, a_ b", layer)
    assert out == "файл olist_order_reviews_dataset.csv, review_comment_message, a_ b" and count == 2


# a table row the page break cut goes on in a row with an empty first cell, and the join finishes the row above with it
def test_a_row_cut_at_the_page_break_is_finished_by_its_continuation():
    from use_cases.piece_join import _join_table

    before = "| Field | Contents | Description |\n|---|---|---|\n| 1 | Device | such as /dev/sda1. Modern |"
    after = ("| Field | Contents | Description |\n|---|---|---|\n|  |  | with a text label instead. |\n"
             "| 2 | Mount point | The directory |")
    head, rest, merged = _join_table(before, after)
    assert merged and head.endswith("| 1 | Device | such as /dev/sda1. Modern with a text label instead. |")
    assert rest == "| 2 | Mount point | The directory |"


# a compound the converter glued at a mid-line hyphen mark gets its hyphen back where the layer spells it so elsewhere
def test_a_compound_glued_at_a_hyphen_mark_takes_the_layers_own_hyphen():
    from use_cases.piece_join import marked_joins, restore_dashes
    from use_cases.route import joined_hyphens

    raw = "p303 exactly-once semantics. p313 support exactly￾once guarantees; de￾signed well"
    out, count = restore_dashes("support exactlyonce guarantees; designed well", joined_hyphens(raw), marked_joins(raw))
    assert out == "support exactly-once guarantees; designed well" and count == 1
