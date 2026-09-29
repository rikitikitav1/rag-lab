import json

import pytest
from config import settings
from engines import converter_tools
from job_handlers import convert, converting

# the stand's own rules, named where a test reads with them rather than taken as a silent default
STAND = settings.intake.route


class _Reply:
    def json(self):
        return {"docling-serve": "test"}


# a cancelled book stopped holding the queue only when a cancel was read between its pieces
def test_a_cancelled_conversion_stops_before_its_next_piece(tmp_path, monkeypatch):
    gold = tmp_path / "files"
    (gold / "books").mkdir(parents=True)
    (gold / "books" / "a.pdf").write_bytes(b"%PDF")
    settings = tmp_path / "converters" / "docling" / "settings"
    settings.mkdir(parents=True)
    (settings / "default.json").write_text(json.dumps({"tool": "docling", "fields": {}, "pages_per_chunk": 50}))
    monkeypatch.setattr(convert, "GOLD", gold)
    monkeypatch.setattr(convert, "RUNS", gold / "runs")
    monkeypatch.setattr(convert, "SETTINGS", tmp_path / "converters")
    monkeypatch.setattr(converting, "SETTINGS", tmp_path / "converters")
    monkeypatch.setattr(converting, "ROOT", tmp_path)
    monkeypatch.setattr(convert, "converter_for", lambda tool: object())
    monkeypatch.setattr(convert, "take", lambda spec: None)
    monkeypatch.setattr(convert.converter, "reading", lambda spec: (None, {"tool": "docling", "build": None}))
    monkeypatch.setattr(convert, "tool_version", lambda spec, tool: {"docling-serve": "test"})
    monkeypatch.setattr(convert, "_pages", lambda path: 120)
    monkeypatch.setattr(convert.job_queue, "is_cancelled", lambda job_id: True)
    monkeypatch.setattr(convert.reading, "route_sha", lambda rule=None: "live-rules")
    called = []
    monkeypatch.setattr(convert, "convert", lambda *args: called.append(args))

    convert.convert_source(
        {"settings": "docling/default", "language": "en", "inputs": ["books/a.pdf"], "out": "run", "_job_id": 7}
    )

    assert called == []
    record = json.loads((gold / "runs" / "run" / "record.json").read_text())
    assert record["cancelled_before"] == "books/a.pdf 1-50"
    # the plain arm is fingerprinted by the live rules, so a rule moved before it resumes sends it to a new out
    assert record["route_sha256"] == "live-rules" and record["word_rules"] is False


# a tool without an adapter is refused by name, not sent Docling's API
def test_a_tool_without_an_adapter_is_refused():
    import pytest
    from errors import Final

    with pytest.raises(Final, match="no adapter for the converter nougat"):
        converting.convert(object(), "nougat", None, [], None, 1)


class _Reply200:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


# a MinerU restart forgot the uploads and every piece failed on a file id from the process before
def test_a_mineru_upload_is_not_reused_across_a_restart(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    uploads = []

    def post(url, json, timeout):
        uploads.append(url)
        return _Reply200({"id": "u", "file": {"id": f"file-{len(uploads)}"}})

    monkeypatch.setattr(converter_tools.requests, "post", post)
    monkeypatch.setattr(converter_tools, "_MINERU_FILES", {})

    first = converter_tools._mineru_file("http://converter", pdf, 1.0)
    again = converter_tools._mineru_file("http://converter", pdf, 1.0)
    after_restart = converter_tools._mineru_file("http://converter", pdf, 2.0)

    assert first == again != after_restart
    assert len(uploads) == 2


# a tool's engine is the one declared for it; its supervisor only says it is up and runs that tool
def test_the_engine_is_the_one_declared_for_the_tool(monkeypatch):
    import dataclasses

    from errors import Final
    from stand_specs import CONVERTER

    docling = dataclasses.replace(CONVERTER, name="converter-docling")
    mineru = dataclasses.replace(CONVERTER, id=10, name="converter-mineru", env_prefix="CONVERTER_MINERU")
    tools = {docling.name: "docling", mineru.name: "docling"}
    monkeypatch.setattr(converting.engines, "card_engines", lambda kind: [docling, mineru])
    monkeypatch.setattr(converting.converter, "reading", lambda spec: (None, {"tool": tools[spec.name]}))

    assert converting.converter_for("docling") is docling
    with pytest.raises(Final, match="converter-mineru runs docling, not mineru"):
        converting.converter_for("mineru")
    with pytest.raises(Final, match="no converter engine is declared for marker"):
        converting.converter_for("marker")


# a hung piece restarts its child in the handler, never inside the tool's adapter
def test_a_timeout_restarts_the_child(monkeypatch):
    restarted = []
    monkeypatch.setitem(
        converter_tools.PIECE, "docling", lambda *a: {"status": "timeout", "errors": ["no result in 1 s"]}
    )
    monkeypatch.setattr(converting, "restart_holder", restarted.append)
    monkeypatch.setattr(converting, "reading_key", lambda *a: None)

    result = converting.convert("spec", "docling", None, [], None, 1)

    assert restarted == ["spec"] and result["errors"][-1] == "child restarted"


# a piece read once is read again from its kept reading, until its file, pages, settings or converter build move
def test_a_kept_reading_skips_the_tool(tmp_path, monkeypatch):
    calls = []
    build = {"built_at": "b1", "files": {}}

    def tool(spec, path, fields, chunk, ceiling):
        calls.append(chunk)
        return {"status": "success", "errors": [], "markdown": "# read", "seconds": 5.0}

    monkeypatch.setitem(converter_tools.PIECE, "docling", tool)
    monkeypatch.setattr(converting, "READINGS", tmp_path / "readings")
    monkeypatch.setattr(converting.converter, "reading", lambda spec: (None, {"build": build}))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF one")

    held = []
    first = converting.convert("spec", "docling", pdf, [("do_ocr", "false")], (1, 50), 60, hold=lambda: held.append(1))
    again = converting.convert("spec", "docling", pdf, [("do_ocr", "false")], (1, 50), 60, hold=lambda: held.append(2))
    converting.convert("spec", "docling", pdf, [("do_ocr", "false")], (51, 100), 60)
    build["built_at"] = "b2"
    converting.convert("spec", "docling", pdf, [("do_ocr", "false")], (1, 50), 60)

    assert again == {**first, "seconds": 0.0, "read_seconds": 5.0, "cached": True}
    assert calls == [(1, 50), (51, 100), (1, 50)]
    assert held == [1]


# a partial reading, a timeout or an unstamped converter keeps nothing, so the next run reads the piece again
def test_a_failed_or_unstamped_reading_is_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(converting, "READINGS", tmp_path / "readings")
    monkeypatch.setattr(converting, "restart_holder", lambda spec: None)
    monkeypatch.setitem(converter_tools.PIECE, "docling", lambda *a: {"status": "timeout", "errors": []})
    monkeypatch.setattr(converting.converter, "reading", lambda spec: (None, {"build": {"built_at": "b"}}))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    converting.convert("spec", "docling", pdf, [], None, 1)
    monkeypatch.setitem(converter_tools.PIECE, "docling", lambda *a: {"status": "partial_success", "errors": []})
    converting.convert("spec", "docling", pdf, [], None, 1)
    monkeypatch.setitem(converter_tools.PIECE, "docling", lambda *a: {"status": "success", "errors": []})
    monkeypatch.setattr(converting.converter, "reading", lambda spec: (None, {}))
    converting.convert("spec", "docling", pdf, [], None, 1)

    assert not (tmp_path / "readings").exists()


def test_a_resume_under_another_language_is_refused():
    record = {"converted": {"a": {}}, "tool": "docling", "settings_sha256": "s", "build": None, "language": "en"}
    with pytest.raises(convert.Final, match="language"):
        convert._refuse_another_configuration(
            record, {"tool": "docling", "settings_sha256": "s", "build": None, "language": "ru"}
        )


# an intake arm reads the gold as the corpus does: the shared plan, the reread, and the join of its pieces
def test_an_intake_arm_reads_through_the_corpus_path_and_joins_its_pieces(tmp_path, monkeypatch):
    from job_handlers import reading
    from use_cases import route

    gold = tmp_path / "files"
    (gold / "books").mkdir(parents=True)
    (gold / "books" / "a.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(convert, "GOLD", gold)
    monkeypatch.setattr(convert, "RUNS", gold / "runs")
    monkeypatch.setattr(convert, "converter_for", lambda tool: object())
    monkeypatch.setattr(convert, "take", lambda spec: None)
    monkeypatch.setattr(convert.converter, "reading", lambda spec: (None, {"tool": "docling", "build": None}))
    monkeypatch.setattr(convert, "tool_version", lambda spec, tool: {"docling-serve": "test"})
    monkeypatch.setattr(convert, "_pages", lambda path: 2)
    monkeypatch.setattr(convert.route, "layer_texts", lambda path, rule=None: ["one", "two"])
    monkeypatch.setattr(route, "route", lambda path, rule=None: [route.Run("docling", (1, 2), "text layer")])
    monkeypatch.setattr(route, "seamless_pieces", lambda path, pages, size, rule=None: [(1, 1), (2, 2)])
    texts = {(1, 1): "## A\n\n| a | b |\n|---|---|\n| 1 | 2 |", (2, 2): "| a | b |\n|---|---|\n| 3 | 4 |"}

    def fake_piece(file, engine, piece, settings, language, rule=None):
        structure = {"piece": list(piece)}
        return {"markdown": texts[piece], "structure": structure, "seconds": 1.0, "status": None, "code": None}

    monkeypatch.setattr(reading, "convert_piece", fake_piece)

    convert.convert_source(
        {"settings": "docling/default", "language": "en", "inputs": ["books/a.pdf"], "out": "run", "intake": True}
    )

    run = gold / "runs" / "run"
    assert (run / "books__a.pdf.md").read_text() == "## A\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    entry = json.loads((run / "record.json").read_text())["converted"]["books/a.pdf"]
    assert sorted(entry["chunks"]) == ["1-1", "2-2"] and entry["joins_healed"]["tables"] == 1
    assert entry["chunks"]["1-1"]["settings"] == "docling/default"
    assert json.loads((run / "books__a.pdf.parts" / "2-2.docling.json").read_text()) == {"piece": [2, 2]}


# a store's file read over its own page range, in pieces of the run's size, as onboarding reads a piece of a book
def test_an_intake_arm_reads_a_page_range_of_a_store_file(tmp_path, monkeypatch):
    from job_handlers import reading
    from use_cases import route

    inbox = tmp_path / "inbox"
    (inbox / "books").mkdir(parents=True)
    (inbox / "books" / "b.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(convert, "INBOX", inbox)
    monkeypatch.setattr(convert, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(convert, "converter_for", lambda tool: object())
    monkeypatch.setattr(convert, "take", lambda spec: None)
    monkeypatch.setattr(convert.converter, "reading", lambda spec: (None, {"tool": "docling", "build": None}))
    monkeypatch.setattr(convert, "tool_version", lambda spec, tool: {"docling-serve": "test"})
    monkeypatch.setattr(convert, "_pages", lambda path: 300)
    monkeypatch.setattr(convert.route, "layer_texts", lambda path, rule=None: [f"page {n}" for n in range(1, 301)])
    monkeypatch.setattr(route, "route", lambda path, rule=None: [route.Run("docling", (1, 300), "text layer")])
    monkeypatch.setattr(route, "seamless_pieces", lambda path, pages, size, rule=None: convert.pieces(*pages, size))
    read = []

    def fake_piece(file, engine, piece, settings, language, rule=None):
        read.append(piece)
        text = " ".join(f"page {n}" for n in range(piece[0], piece[1] + 1))
        return {"markdown": text, "structure": None, "seconds": 1.0, "status": None, "code": None}

    monkeypatch.setattr(reading, "convert_piece", fake_piece)
    options = {"settings": "docling/default", "language": "en", "inputs": ["books/b.pdf"], "out": "slice"}
    options |= {"intake": True, "root": "inbox", "pages": {"books/b.pdf": (100, 110)}, "pages_per_chunk": 5}

    convert.convert_source(options)

    assert read == [(100, 104), (105, 109), (110, 110)]
    record = json.loads((tmp_path / "runs" / "slice" / "record.json").read_text())
    assert record["pages_per_chunk"] == 5 and record["converted"]["books/b.pdf"]["range"] == [100, 110]


def test_a_range_without_intake_is_refused():
    from job_specs import ConvertSource

    with pytest.raises(ValueError, match="need intake"):
        ConvertSource(settings="docling/default", language="en", inputs=["a.pdf"], out="x", pages={"a.pdf": (1, 2)})


def test_a_run_s_knobs_are_checked_and_need_intake():
    from job_specs import ConvertSource

    base = {"settings": "docling/default", "language": "en", "inputs": ["a.pdf"], "out": "x"}
    assert ConvertSource(**base, intake=True, knobs={"reread_settings": "docling/pypdfium2"}).knobs
    with pytest.raises(ValueError):
        ConvertSource(**base, intake=True, knobs={"seam_margin": 0.9})
    with pytest.raises(ValueError, match="need intake"):
        ConvertSource(**base, knobs={"seam_window": 1})


# an intake arm's route stamp moves when an input's own source moves a knob, not only when the stand's rule does
def test_an_arm_s_route_stamp_moves_with_a_source_s_knob(monkeypatch):
    from job_handlers import reading

    monkeypatch.setattr(reading, "load_settings", lambda name: ({}, name))
    origin = {"intake": {}}
    monkeypatch.setattr(convert, "_declaration", lambda name: origin)
    options = {"inputs": ["books/a.pdf"], "settings": "docling/default", "sources": {"books/a.pdf": "a"}}
    before = convert._arm_route_sha(options, "docling")
    origin["intake"] = {"seam_window": 5}
    assert convert._arm_route_sha(options, "docling") != before
    origin["intake"] = {"settings": {"docling": "docling/pypdfium2"}}
    assert convert._arm_route_sha(options, "docling") != before


# the reread's floor was set on the layer as PDFium gives it; only the word rules read the layer with its hyphens joined
def test_the_word_rules_read_the_joined_layer_and_the_reread_floor_the_raw_one(monkeypatch):
    from job_handlers import reading

    layer = "a sep￾\r\narate word"
    seen = []
    monkeypatch.setattr(
        reading,
        "convert_piece",
        lambda *a, **k: {"markdown": "a sep- arate word", "structure": None, "seconds": 0.0, "status": None},
    )
    monkeypatch.setattr(
        reading.raw_quality, "conversion_signals", lambda markdown, text: seen.append(text) or {"layer_f1": 1.0}
    )

    done, _, reread = reading.read_piece("a.pdf", "docling", (1, 1), "docling/default", {}, "en", layer, STAND)

    assert done["markdown"] == "a separate word" and done["code"]["words_joined"] == 1
    assert seen == [layer] and reread is None


# a reading no code rule ran on, as MinerU's, gets its entities decoded; a Docling reading is not decoded twice
def test_entities_are_decoded_on_a_reading_the_code_rules_did_not_touch(monkeypatch):
    from job_handlers import reading

    monkeypatch.setattr(reading.raw_quality, "conversion_signals", lambda markdown, text: {"layer_f1": None})
    for code, expected, count in ((None, "#include <signal.h>", 2), ({"rebuilt": 0}, "a &amp;lt; b", None)):
        piece = {"markdown": "#include &lt;signal.h&gt;" if code is None else "a &amp;lt; b", "code": code}
        monkeypatch.setattr(reading, "convert_piece", lambda *a, piece=piece, **k: {**piece, "structure": None})
        done, _, _ = reading.read_piece("a.pdf", "mineru", (1, 1), None, {}, "en", None, STAND)
        assert done["markdown"] == expected and (done["code"] or {}).get("entities_decoded") == count


# a markdown file read as it is keeps its author's escapes and entities: no word rule touches it
def test_a_file_read_without_an_engine_is_left_as_its_author_wrote_it(tmp_path):
    from job_handlers import reading

    page = tmp_path / "a.md"
    page.write_text("\\- item &lt;div&gt; and\\_that")

    done, _, reread = reading.read_piece(page, None, None, None, {}, "en", None, STAND)

    assert done["markdown"] == "\\- item &lt;div&gt; and\\_that" and not done.get("code") and reread is None


def _table_item(page):
    return {"label": "table", "prov": [{"page_no": page}]}


def _reading(markdown, pages):
    tables = [_table_item(p) for p in pages]
    return {"markdown": markdown, "structure": {"body": {"children": []}, "tables": tables, "_order": tables}}


# the second reading keeps its prose and takes back a table the first reading kept with more cells, on the same page
def test_a_second_reading_lost_on_cells_keeps_its_prose_and_takes_the_first_readings_tables(monkeypatch):
    from job_handlers import reading

    monkeypatch.setattr(reading.code_lines, "reading_order", lambda st: [("t", t) for t in st.get("_order", [])])
    first = _reading("ass u m in g text\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nmore", [3])
    second = _reading("assuming text\n\n| a b |\n|---|\n\nmore", [3])

    markdown, spliced = reading.splice_tables(second, first)
    assert spliced == 1 and markdown == "assuming text\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nmore"
    # a table the prose reading kept whole stays, and a count the structure does not match leaves the reading alone
    assert reading.splice_tables(first, second) == (first["markdown"], 0)
    assert reading.splice_tables(second, _reading(first["markdown"], [3, 4])) == (second["markdown"], 0)


def _formula(page, orig):
    return {"label": "formula", "prov": [{"page_no": page}], "orig": orig}


# the second reading's backend reads a math font's operators as control marks; the first reading's formula comes back
def test_a_formula_that_lost_its_operators_comes_back_from_the_first_reading(monkeypatch):
    from job_handlers import reading

    monkeypatch.setattr(reading.code_lines, "reading_order", lambda st: [("f", f) for f in st.get("_order", [])])
    lost = _formula(3, "r \x01x - y\x03 z")
    kept = _formula(3, "r =( x + y ) z")
    prose = {"markdown": "text\n\nr x - y z\n\nmore", "structure": {"_order": [lost]}}
    first = {"markdown": "text\n\nr =( x + y ) z\n\nmore", "structure": {"_order": [kept]}}

    assert reading.splice_formulas(prose, first) == ("text\n\nr =( x + y ) z\n\nmore", 1)
    assert reading.splice_formulas(first, prose) == (first["markdown"], 0), "a formula with its operators stays"


# a short formula is taken back on its own line, not where the same words first stand in prose
def test_a_formula_is_spliced_on_its_own_line_and_not_in_the_prose_before_it(monkeypatch):
    from job_handlers import reading

    monkeypatch.setattr(reading.code_lines, "reading_order", lambda st: [("f", f) for f in st.get("_order", [])])
    prose = {"markdown": "set x 1 before\n\nx 1\n\nmore", "structure": {"_order": [_formula(3, "x \x011")]}}
    first = {"markdown": "", "structure": {"_order": [_formula(3, "x = 1")]}}

    assert reading.splice_formulas(prose, first) == ("set x 1 before\n\nx = 1\n\nmore", 1)


# a second reading taken whole for its prose still gets back a formula whose operators only the first one kept
def test_a_reading_taken_whole_takes_back_the_formulas_the_first_one_kept(monkeypatch):
    from config import settings
    from job_handlers import reading

    monkeypatch.setattr(reading.code_lines, "reading_order", lambda st: [("f", f) for f in st.get("_order", [])])
    first = {"markdown": "prose\n\nr =( x + y ) z", "structure": {"_order": [_formula(1, "r =( x + y ) z")]}}
    second = {"markdown": "prose read better\n\nr x - y z", "structure": {"_order": [_formula(1, "r \x01x - y\x03 z")]}}
    readings = iter([first, second])
    monkeypatch.setattr(reading, "convert_piece", lambda *a, **k: {**next(readings), "seconds": 0.0, "status": None})
    monkeypatch.setattr(reading, "_dashes_back", lambda done, *a: done)
    monkeypatch.setattr(reading, "load_settings", lambda name: ({}, "sha"))
    def agreement(markdown, text):
        return {"layer_f1": 0.9 if "better" in markdown else 0.5}

    monkeypatch.setattr(reading.raw_quality, "conversion_signals", agreement)
    rule = settings.intake.route.model_copy(update={"splice_tables": True})

    done, taken, reread = reading.read_piece("a.pdf", "docling", (1, 1), "docling/default", {}, "en", "layer", rule)

    assert taken == rule.reread_settings and reread["taken"] and reread["formulas_spliced"] == 1
    assert done["markdown"] == "prose read better\n\nr =( x + y ) z"


# a source's knobs reach a reader only if every step is handed its rule; a missed one fails, never reads the stand's
def test_no_step_of_the_reading_path_falls_back_to_the_stands_rules():
    import inspect

    from job_handlers import reading
    from use_cases import route

    steps = (
        reading.plan, reading.convert_piece, reading.read_piece, reading.route_sha, converting.code_lines_of,
        route.route, route.page_signals, route.seamless_pieces, route.mono_rows,
    )
    empty = inspect.Parameter.empty
    assert [s.__name__ for s in steps if inspect.signature(s).parameters["rule"].default is not empty] == []
