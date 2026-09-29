import re

from use_cases import code_lines


class _Text:
    def __init__(self, chars):
        self.chars = chars

    def count_chars(self):
        return len(self.chars)

    def get_text_range(self, i, n):
        return self.chars[i][4]

    def get_charbox(self, i, loose=False):
        return self.chars[i][:4]

    def close(self):
        pass


class _Page:
    def __init__(self, chars, height=800.0):
        self.chars, self.height = chars, height

    def get_height(self):
        return self.height

    def get_textpage(self):
        return _Text(self.chars)

    def close(self):
        pass


# monospace glyphs of width 6 on rows 12 high; a comma's tight box would sit below its row, the loose one does not
def _line(text, x, y):
    return [(x + 6 * i, y, x + 6 * i + 6, y + 12, ch) for i, ch in enumerate(text) if ch != " "]


def test_a_block_comes_back_with_its_lines_and_indentation():
    chars = _line("def f(a, b):", 100, 700) + _line("    return a_b  # сумма", 100, 688) + _line("outside", 400, 700)
    box = {"l": 99, "r": 300, "t": 713, "b": 687, "coord_origin": "BOTTOMLEFT"}

    assert code_lines.layer_code(_Page(chars), box) == "def f(a, b):\n    return a_b  # сумма"


def test_a_top_left_box_is_turned_to_the_layer_s_origin():
    chars = _line("x = 1", 100, 700)
    box = {"l": 99, "r": 200, "t": 87, "b": 101, "coord_origin": "TOPLEFT"}

    assert code_lines.layer_code(_Page(chars), box) == "x = 1"


def test_fences_are_rebuilt_in_order_and_kept_when_they_do_not_pair(tmp_path, monkeypatch):
    import pypdfium2

    page = _Page(_line("a  =  1", 100, 700))

    class _Document:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return page

        def close(self):
            pass

    monkeypatch.setattr(pypdfium2, "PdfDocument", _Document)
    item = {"label": "code", "prov": [{"page_no": 1, "bbox": {"l": 99, "r": 200, "t": 713, "b": 699}}]}
    markdown = "Text\n\n```\na = 1 glued\n```\n\nMore"

    rebuilt, counts = code_lines.rebuild(markdown, _structure(item), tmp_path / "a.pdf")
    assert rebuilt == "Text\n\n```\na  =  1\n```\n\nMore"
    assert counts == {"rebuilt": 1, "kept": 0, "joined": 0, "duplicates_dropped": 0}

    same, counts = code_lines.rebuild(markdown, _structure(item, item), tmp_path / "a.pdf")
    assert same == markdown and counts == {"rebuilt": 0, "kept": 1, "joined": 0, "duplicates_dropped": 0}


def _structure(*texts, furniture=()):
    items = list(texts) + [{"label": "page_footer", "content_layer": "furniture"} for _ in furniture]
    order = list(range(len(texts)))
    for at in furniture:
        order.insert(at, len(texts) + furniture.index(at))
    return {"texts": items, "body": {"children": [{"$ref": f"#/texts/{i}"} for i in order]}}


# a block the page break cut comes back as one across a running foot; a block on the same page stays apart
def test_a_code_block_cut_by_a_page_break_is_one_again(tmp_path, monkeypatch):
    import pypdfium2

    class _Document:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return _Page([])

        def close(self):
            pass

    monkeypatch.setattr(pypdfium2, "PdfDocument", _Document)

    def code(page):
        return {"label": "code", "prov": [{"page_no": page, "bbox": {"l": 0, "r": 1, "t": 1, "b": 0}}]}

    markdown = "```\nfor i in x:\n```\n\n```\n    print(i)\n```\n\n```\nlater\n```"
    structure = _structure(code(4), code(5), code(5), furniture=(1,))
    joined, counts = code_lines.rebuild(markdown, structure, tmp_path / "a.pdf")

    assert joined == "```\nfor i in x:\n    print(i)\n```\n\n```\nlater\n```" and counts["joined"] == 1


# a proportional face has no one advance to count spaces by, so its box keeps Docling's text
def test_a_block_set_in_a_proportional_face_is_not_rebuilt():
    widths = zip("illWWm", (2, 2, 3, 9, 9, 8), strict=True)
    chars = [(100 + 7 * i, 700, 100 + 7 * i + w, 712, ch) for i, (ch, w) in enumerate(widths)]
    box = {"l": 99, "r": 300, "t": 713, "b": 699, "coord_origin": "BOTTOMLEFT"}

    assert code_lines.layer_code(_Page(chars), box) is None


# a box drawn short of the line's end: the row runs on to its end, and the next box on the page does not give it again
def test_a_short_box_runs_on_to_the_line_s_end_and_a_row_is_given_once():
    page = _Page(_line("resources :comments", 100, 700) + _line("get 'x'", 100, 688))
    short = {"l": 99, "r": 150, "t": 713, "b": 699, "coord_origin": "BOTTOMLEFT"}
    both = {"l": 99, "r": 300, "t": 713, "b": 687, "coord_origin": "BOTTOMLEFT"}
    taken = set()

    assert code_lines.layer_code(page, short, taken) == "resources :comments"
    assert code_lines.layer_code(page, both, taken) == "get 'x'"


# a listing's line numbers, set in another face, neither undo the monospace check nor stay in the code
def test_a_listing_loses_its_line_numbers():
    numbers = [(90, 700 - 12 * i, 93, 712 - 12 * i, str(i + 1)) for i in range(4)]
    listing = ["int main(int argc, char *argv) {", "  int counter = 0;", "  return counter + argc;", "}"]
    code = sum((_line(text, 100, 700 - 12 * i) for i, text in enumerate(listing)), [])
    box = {"l": 85, "r": 300, "t": 713, "b": 651, "coord_origin": "BOTTOMLEFT"}

    assert code_lines.layer_code(_Page(numbers + code), box) == "\n".join(listing)


# a shell session Docling made a heading is fenced from the layer; prose, a signature and an outline heading stay
def test_code_with_no_box_is_fenced_and_an_outline_heading_stays(tmp_path, monkeypatch):
    import re

    import pypdfium2

    session = _line("# telnet localhost 2181", 100, 700) + _line("Trying 127.0.0.1...", 100, 688)
    signature = _line("class mailbox.Mailbox(path)", 100, 650)
    page = _Page(session + signature)

    class _Document:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return page

        def close(self):
            pass

    monkeypatch.setattr(pypdfium2, "PdfDocument", _Document)
    monkeypatch.setattr(code_lines, "_faces", lambda page: [(*c[:4], "Courier") for c in session + signature])
    mono = {"telnet": 1.0, "Brokers": 0.0, "kafka-topics": 1.0, "mailbox": 1.0}
    monkeypatch.setattr(code_lines, "mono_share", lambda page, box, name, glyphs=None, spread=None: mono[box["key"]])

    def item(label, text, key, top, bottom):
        return {
            "label": label,
            "text": text,
            "prov": [{"page_no": 1, "bbox": {"l": 99, "r": 300, "t": top, "b": bottom, "key": key}}],
        }

    structure = _structure(
        item("section_header", "kafka-topics", "kafka-topics", 760, 748),
        item("text", "Brokers keep the log.", "Brokers", 740, 728),
        item("section_header", "telnet localhost 2181 Trying 127.0.0.1...", "telnet", 713, 687),
        item("section_header", "class mailbox.Mailbox(path)", "mailbox", 663, 649),
    )
    markdown = (
        "## kafka-topics\n\nBrokers keep the log.\n\n# telnet localhost 2181 Trying 127.0.0.1...\n\n"
        "##### class mailbox.Mailbox(path)"
    )
    fenced, count = code_lines.fence_mono(markdown, structure, tmp_path / "a.pdf", re.compile("Mono"), ["kafka-topics"])

    assert count == 1
    assert fenced == (
        "## kafka-topics\n\nBrokers keep the log.\n\n```\n# telnet localhost 2181\nTrying 127.0.0.1...\n```\n\n"
        "##### class mailbox.Mailbox(path)"
    )


def test_a_term_in_the_code_face_stays_text_and_a_short_prompt_is_code():
    assert not code_lines._long_or_prompt("hostname")
    assert not code_lines._long_or_prompt("peer Port")
    assert code_lines._long_or_prompt("$ ls")
    assert code_lines._long_or_prompt("kafka-topics.sh --list --bootstrap-server localhost:9092")


# a listing's callouts in a text face leave the fence and follow it as a line, where the source asks for it
def test_callouts_leave_the_listing_as_a_line_after_it(tmp_path, monkeypatch):
    import re

    import pypdfium2

    code = _line("return conn.hget('login:', token)", 100, 700)
    note = _line("Fetch the user", 400, 700)
    page = _Page(code + note)

    class _Document:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return page

        def close(self):
            pass

    monkeypatch.setattr(pypdfium2, "PdfDocument", _Document)
    dropped = frozenset(c[:4] for c in note)
    monkeypatch.setattr(code_lines, "_callouts", lambda page, box, name: (dropped, "Fetch the user"))
    item = {"label": "code", "prov": [{"page_no": 1, "bbox": {"l": 99, "r": 500, "t": 713, "b": 699}}]}
    markdown = "```\nreturn conn.hget('login:', token) Fetch the user\n```\n\nNext"

    rebuilt, counts = code_lines.rebuild(markdown, _structure(item), tmp_path / "a.pdf", callouts=re.compile("Courier"))
    assert rebuilt == "```\nreturn conn.hget('login:', token)\n```\n\nFetch the user\n\nNext"
    assert counts["rebuilt"] == 1


# a code face whose boxes are wider than its step keeps its spaces when counted by glyph origins
def test_a_wide_boxed_face_keeps_its_spaces_by_step():
    line = "x := 1; // Literal characters"
    glyphs = [(100 + 5.4 * i, 700, 100 + 5.4 * i + (7.8 if i > 7 else 5.4), 712, ch) for i, ch in enumerate(line)]
    box = {"l": 99, "r": 300, "t": 713, "b": 699}
    page = _Page([g for g in glyphs if g[4] != " "])
    assert code_lines.layer_code(page, box) != line
    assert code_lines.layer_code(page, box, by_step=True) == line


# a source's own monospace spread decides the equal-advance check, as it does for the code it rebuilds
def test_mono_share_reads_the_source_s_spread():
    import re

    glyphs = [(10 * i, 0, 10 * i + (6 if i % 2 else 6.9), 12, "") for i in range(10)]
    box = {"l": -5, "r": 200, "t": 20, "b": -5}
    assert code_lines.mono_share(_Page([]), box, re.compile("Mono"), glyphs, spread=0.2) == 1.0
    assert code_lines.mono_share(_Page([]), box, re.compile("Mono"), glyphs, spread=0.05) == 0.0


# with the row rules off a listing keeps its numbers and a box drawn short stays short
def test_row_rules_off_keep_the_box_as_drawn():
    chars = _line("1 a = 1", 100, 700) + _line("2 b = 2", 100, 688) + _line("3 c = 3", 100, 676)
    chars += _line(" # more", 142, 676)
    box = {"l": 99, "r": 143, "t": 713, "b": 675}
    assert code_lines.layer_code(_Page(chars), box, rows_by=frozenset()) == "1 a = 1\n2 b = 2\n3 c = 3"
    tally = {}
    drawn = code_lines.layer_code(_Page(chars), box, rows_by=frozenset({"run_on"}), tally=tally)
    assert drawn.endswith("3 c = 3 # more")
    assert tally == {"rows_run_on": 1}
    assert code_lines.layer_code(_Page(chars), box) == "a = 1\nb = 2\nc = 3 # more"


# a line-end hyphen PDFium marks as U+FFFE is a real minus in code drawn from the layer
def test_a_marked_line_end_hyphen_is_a_minus_in_code():
    chars = _line("debug=true app\ufffe", 100, 700)
    box = {"l": 99, "r": 300, "t": 713, "b": 699}
    assert code_lines.layer_code(_Page(chars), box) == "debug=true app-"


def test_each_picture_placeholder_gets_its_page_place_and_caption():
    structure = {
        "body": {"children": [{"$ref": "#/pictures/0"}, {"$ref": "#/texts/0"}, {"$ref": "#/pictures/1"}]},
        "texts": [{"text": "Figure 2.14  Processes [and] channels", "prov": [{"page_no": 87}]}],
        "pictures": [
            {"prov": [{"page_no": 87}], "captions": [{"$ref": "#/texts/0"}]},
            {"prov": [{"page_no": 87}], "captions": []},
        ],
    }
    markdown = "<!-- image -->\n\nFigure 2.14 Processes and channels\n\n<!-- image -->"
    assert code_lines.picture_addresses(markdown, structure) == (
        "![Figure 2.14 Processes \\[and\\] channels](picture:p87-1)\n\nFigure 2.14 Processes and channels\n\n"
        "![](picture:p87-2)",
        2,
    )
    assert code_lines.picture_addresses("<!-- image -->", {"body": {}}) == ("<!-- image -->", 0)


def test_each_formula_placeholder_gets_the_layer_text_under_it_flattened():
    structure = {
        "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}]},
        "texts": [
            {"label": "formula", "orig": "T(n) = T\r\n\r\ndn/2e\r\n\x01\r\n+ O(n)", "prov": [{"page_no": 46}]},
            {"label": "text", "text": "so", "prov": [{"page_no": 46}]},
            {"label": "formula", "orig": "\uf8f1\uf8f2", "prov": [{"page_no": 46}]},
        ],
    }
    markdown = "<!-- formula-not-decoded -->\n\nso\n\n<!-- formula-not-decoded -->"
    assert code_lines.formula_text(markdown, structure) == (
        "T(n) = T dn/2e + O(n)\n\nso\n\n<!-- formula-not-decoded -->",
        1,
    )
    assert code_lines.formula_text("<!-- formula-not-decoded -->", {"body": {}}) == (
        "<!-- formula-not-decoded -->",
        0,
    )


# a fragment box that drew only a row's tail does not take the row from the block that holds it whole
def test_a_row_whose_tail_a_fragment_drew_is_still_given_whole_by_its_block():
    page = _Page(_line("PRIVACY_URL=https://fedoraproject.org/wiki/Legal", 100, 700) + _line("VARIANT=ws", 100, 688))
    tail = {"l": 250, "r": 400, "t": 713, "b": 699, "coord_origin": "BOTTOMLEFT"}
    block = {"l": 99, "r": 400, "t": 713, "b": 687, "coord_origin": "BOTTOMLEFT"}
    taken = set()

    code_lines.layer_code(page, tail, taken)

    assert code_lines.layer_code(page, block, taken) == "PRIVACY_URL=https://fedoraproject.org/wiki/Legal\nVARIANT=ws"


# two listings side by side share their rows' heights; the second is not taken for the first drawn again
def test_two_listings_side_by_side_are_each_given_their_own_rows():
    page = _Page(
        _line("a = 1", 100, 700) + _line("b = 2", 100, 688) + _line("x = 9", 300, 700) + _line("y = 8", 300, 688)
    )
    left = {"l": 99, "r": 200, "t": 713, "b": 687, "coord_origin": "BOTTOMLEFT"}
    right = {"l": 299, "r": 400, "t": 713, "b": 687, "coord_origin": "BOTTOMLEFT"}
    taken = set()

    assert code_lines.layer_code(page, left, taken, rows_by=frozenset({"once"})) == "a = 1\nb = 2"
    assert code_lines.layer_code(page, right, taken, rows_by=frozenset({"once"})) == "x = 9\ny = 8"


# a code box whose every row an earlier box drew is a duplicate: its block goes, not Docling's own misreading of it
def test_a_code_block_that_only_repeats_rows_drawn_before_is_dropped(tmp_path, monkeypatch):
    import pypdfium2

    page = _Page(_line("x = f(y);", 100, 700))

    class _Document:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return page

        def close(self):
            pass

    monkeypatch.setattr(pypdfium2, "PdfDocument", _Document)
    box = {"l": 99, "r": 200, "t": 713, "b": 699}
    first = {"label": "code", "prov": [{"page_no": 1, "bbox": box}]}
    again = {"label": "code", "prov": [{"page_no": 1, "bbox": box}]}
    markdown = "Text\n\n```\nx = f(y\n```\n\n```\nx f(y);\n```\n\nMore"

    rebuilt, counts = code_lines.rebuild(markdown, _structure(first, again), tmp_path / "a.pdf")
    assert rebuilt == "Text\n\n```\nx = f(y);\n```\n\n\n\nMore" and counts["duplicates_dropped"] == 1


# a page left open is closed by the garbage collector in whichever thread collects, and PDFium is not thread-safe
def test_every_page_opened_while_reading_code_is_closed_before_the_call_returns(tmp_path, monkeypatch):
    import pypdfium2
    from pypdf import PdfWriter

    open_at_close = []
    close = pypdfium2.PdfDocument.close

    def counting_close(document, *args, **kwargs):
        open_at_close.append(sum(1 for ref in document._kids if ref() is not None and ref().raw))
        return close(document, *args, **kwargs)

    monkeypatch.setattr(pypdfium2.PdfDocument, "close", counting_close)

    writer = PdfWriter()
    writer.add_blank_page(600, 800)
    with (tmp_path / "a.pdf").open("wb") as f:
        writer.write(f)
    item = {"label": "code", "prov": [{"page_no": 1, "bbox": {"l": 99, "r": 200, "t": 713, "b": 699}}]}
    box = {"l": 99, "r": 300, "t": 713, "b": 687}
    text = {"label": "text", "text": "telnet localhost 2181", "prov": [{"page_no": 1, "bbox": box}]}
    code_lines.rebuild("```\nx\n```", _structure(item), tmp_path / "a.pdf")
    code_lines.fence_mono("telnet localhost 2181", _structure(text), tmp_path / "a.pdf", re.compile("Mono"))
    assert open_at_close and not any(open_at_close)


# the one reader of glyph faces gives a line-end hyphen as the code reader does, a real `-`
def test_the_face_reader_and_the_code_reader_see_one_text_at_a_line_end_hyphen(monkeypatch):
    import pypdfium2.raw as pdfium_c

    monkeypatch.setattr(pdfium_c, "FPDFText_GetFontInfo", lambda *a: 0)
    monkeypatch.setattr(_Text, "raw", None, raising=False)
    page = _Page(_line("app￾", 100, 700))

    faces = "".join(g[5] for g in code_lines.page_glyphs(page))
    assert faces == "".join(c[4] for c in code_lines._chars(page)) == "app-"


# the seam's row reader groups rows as the code reader does: by name, else by one advance over eight glyphs or more
def test_the_seams_rows_are_the_code_readers_rows(monkeypatch):
    from config import settings
    from use_cases import route

    def glyphs(text, y, face, widths):
        return [(10.0 * i, y, 10.0 * i + w, y + 12, face, c) for i, (c, w) in enumerate(zip(text, widths, strict=True))]

    rows = (
        glyphs("int x = 1;", 700, "Courier", [6] * 10)
        + glyphs("Some prose", 680, "Times", [5, 7, 4, 9, 3, 6, 8, 4, 5, 7])
        + glyphs("nameless1", 660, "", [6] * 9)
    )
    monkeypatch.setattr(code_lines, "page_glyphs", lambda page: rows)
    rule = settings.intake.route.model_copy(update={"mono_faces": ["Courier"], "seam_margin": 0.0})

    assert route.mono_rows(_Page([]), rule) == [True, False, True]


# a code block that reads a lone pipe is still a code block; only a pipe between two halves is stepped over
def test_a_code_block_reading_a_pipe_keeps_its_place_among_the_code_blocks():
    structure = _structure(
        {"label": "code", "text": "|", "prov": [{"page_no": 3}]},
        {"label": "text", "text": "|", "prov": [{"page_no": 3}]},
        {"label": "code", "text": "a = 1", "prov": [{"page_no": 3}]},
        {"label": "code", "text": "b = 2", "prov": [{"page_no": 4}]},
    )
    assert code_lines.continued(structure)["code"] == [False, True, False], "one mark per code block, three blocks"
