import pytest
from job_handlers import onboard
from job_handlers.base import Final
from job_handlers.converting import pieces
from models.corpus import DataSource
from use_cases import source_intake


# a PDF run is cut by the seam rule at the settings' piece size, a whole file is one unit
def test_a_run_is_cut_into_pieces_of_the_settings_size(tmp_path, monkeypatch):
    from use_cases import route

    runs = {".pdf": [route.Run("docling", (1, 120), "text layer")], ".md": [route.Run(None, None, "markdown")]}
    monkeypatch.setattr(onboard.route, "route", lambda file, rule=None: runs[file.suffix])
    monkeypatch.setattr(onboard.route, "seamless_pieces", lambda file, pages, size, rule=None: pieces(*pages, size))
    names, loaded = {"docling": "docling/default"}, {}

    planned = onboard.reading.plan(tmp_path / "a.pdf", names, loaded)
    assert [(piece, name) for _, piece, name in planned] == [
        ((1, 50), "docling/default"),
        ((51, 100), "docling/default"),
        ((101, 120), "docling/default"),
    ]
    assert [piece for _, piece, _ in onboard.reading.plan(tmp_path / "a.md", names, loaded)] == [None]


def _rows(key, *rows):
    return [{"breached": list(b), key: words} for words, b in rows]


# a raw source is marked bad with its reasons when too much of its text breaches, and nothing stops the run
def test_the_verdict_names_what_was_breached():
    ok = _rows("output_words", (100, ()), (100, ()))
    assert onboard._verdict(ok, [])[:2] == ("ok", {})
    # a breach in a tenth of the text is dirty, however many rows it is out of
    dirty = _rows("words", (900, ()), (100, ("prefix_dominates.max",)))
    assert onboard._verdict(ok, dirty)[:2] == ("dirty", {"prefix_dominates.max": 1})
    # the same one row breaching a third of the text is bad
    verdict, reasons, shares = onboard._verdict(ok, _rows("words", (200, ()), (100, ("dup_in_file.max",))))
    assert verdict == "bad" and shares["sections"] == 0.3333


# an empty piece has no words of its own, and weighs as the mean rather than as nothing
def test_an_empty_breaching_piece_weighs_as_the_mean():
    rows = _rows("output_words", (100, ()), (100, ()), (0, ("output_words.empty",)))
    assert onboard.bad_share(rows, "output_words") == 0.3333


def test_a_folder_origin_stays_inside_the_stand(tmp_path):
    source = DataSource(name="a", kind="local", origin={"folder": "../../etc"})

    with pytest.raises(Final, match="not a folder of the stand"):
        source_intake.gather(source, tmp_path / "fetched", tmp_path)


def test_the_door_refuses_a_missing_or_empty_folder_before_the_queue(tmp_path):
    from sources.declaration import Declaration

    (tmp_path / "empty").mkdir()
    (tmp_path / "book").mkdir()
    (tmp_path / "book" / "a.pdf").write_bytes(b"%PDF")

    def declared(folder):
        return Declaration(name="a", language="en", licence="MIT", folder=folder)

    assert "not a folder of the stand" in source_intake.declaration_refusal(declared("typo"), tmp_path)
    assert "no files" in source_intake.declaration_refusal(declared("empty"), tmp_path)
    assert source_intake.declaration_refusal(declared("book"), tmp_path) is None


def test_a_second_onboard_of_one_source_is_refused_while_the_first_waits():
    source = DataSource(name="a", kind="local", stage="declared")

    assert source_intake.onboard_refusal(source, None) is None
    assert "job 7" in source_intake.onboard_refusal(source, 7)


def test_a_run_names_settings_files_of_each_tool_only():
    from job_specs import Refused, check

    check("onboard_source", {"source": "a", "settings": {"docling": "docling/default"}})
    with pytest.raises(Refused, match="not a settings file of that tool"):
        check("onboard_source", {"source": "a", "settings": {"docling": "mineru/ocr"}})


class _Session:
    def __init__(self, source):
        self.source = source

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def scalar(self, stmt):
        return self.source

    def get(self, model, id):
        return self.source

    def expunge(self, obj):
        pass

    def commit(self):
        pass


# the whole job with the engines stubbed: pieces, resume, the file assembled in page order, the row and the stage
def test_the_job_turns_a_declared_folder_into_a_raw_source(tmp_path, monkeypatch):
    import json

    from models.corpus import Stage
    from use_cases import route

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note on snapshots and branches.")
    (inbox / "b.pdf").write_bytes(b"%PDF")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)

    def fake_route(file, rule=None):
        if file.suffix == ".pdf":
            return [route.Run("docling", (1, 120), "text layer", {"absorbed": []})]
        return [route.Run(None, None, "markdown")]

    monkeypatch.setattr(onboard.route, "route", fake_route)
    monkeypatch.setattr(onboard.route, "seamless_pieces", lambda file, pages, size, rule=None: pieces(*pages, size))
    layers = [f"page {n} words here" for n in range(1, 121)]
    monkeypatch.setattr(onboard.route, "layer_texts", lambda file, rule=None: layers)
    calls = []

    def fake_piece(file, engine, piece, settings, language, rule=None):
        calls.append((file.name, piece))
        if engine is None:
            return {"markdown": file.read_text(), "seconds": 0.0, "status": None, "structure": None}
        text = " ".join(f"page {n} words here" for n in range(piece[0], piece[1] + 1))
        failed = piece == (51, 100)
        return {
            "markdown": f"## Pages {piece[0]}\n\n{text}",
            "structure": None,
            "seconds": 1.0,
            "status": {"status": "failure" if failed else "success", "errors": ["boom"] if failed else []},
        }

    monkeypatch.setattr(onboard.reading, "convert_piece", fake_piece)

    onboard.onboard_source({"source": "demo"})

    folder = next((tmp_path / "raw").glob("demo_*"))
    record = json.loads((folder / "record.json").read_text())
    assert sorted(record["units"]) == ["a.md", "b.pdf#1-50", "b.pdf#101-120", "b.pdf#51-100"]
    assert "conversion.failed" in record["units"]["b.pdf#51-100"]["breached"]
    assert record["units"]["b.pdf#1-50"]["engine"] == "docling"
    assert record["units"]["b.pdf#1-50"]["layer_f1"] > 0.95
    whole = (folder / "files" / f"{onboard._stem('b.pdf')}.md").read_text()
    assert whole.index("## Pages 1") < whole.index("## Pages 51") < whole.index("## Pages 101")
    assert source.stage == Stage.raw
    assert source.raw["verdict"] == "bad" and source.raw["reasons"]["conversion.failed"] == 1
    assert "code_version" in source.raw["code"]

    # a second run of the same settings resumes: only the failed piece is converted again
    calls.clear()
    onboard.onboard_source({"source": "demo"})
    assert calls == [("b.pdf", (51, 100))]

    # a piece whose markdown left the folder is converted again
    calls.clear()
    (folder / "pieces" / f"{onboard._stem('b.pdf#1-50')}.md").unlink()
    onboard.onboard_source({"source": "demo"})
    assert calls == [("b.pdf", (1, 50)), ("b.pdf", (51, 100))]

    # a changed file is converted again, and a file that left takes its rows with it
    calls.clear()
    (inbox / "a.md").write_text("## Notes\n\nAnother note.")
    (inbox / "b.pdf").unlink()
    onboard.onboard_source({"source": "demo"})
    assert calls == [("a.md", None)]
    assert sorted(json.loads((folder / "record.json").read_text())["units"]) == ["a.md"]


def test_two_keys_never_share_a_file():
    assert onboard._stem("docs/a.pdf#1-10") != onboard._stem("docs_a.pdf#1-10")


def test_two_urls_ending_alike_are_two_files(tmp_path, monkeypatch):
    class _Got:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield self.url.encode()

    monkeypatch.setattr(source_intake.fetch.requests, "get", lambda url, **kw: _Got(url))
    a, fresh_a = source_intake._download("https://x.org/intro/index.html", tmp_path)
    b, _ = source_intake._download("https://x.org/api/index.html", tmp_path)
    again, fresh_again = source_intake._download("https://x.org/intro/index.html", tmp_path)
    assert a != b and a.suffix == ".html" and a.read_text() != b.read_text()
    assert fresh_a and not fresh_again and again == a
    assert not list(tmp_path.rglob("*.part"))


# an unreadable pdf is a marked row of the source, never the end of the job
def test_an_unreadable_pdf_is_marked_not_raised(tmp_path, monkeypatch):
    from models.corpus import Stage

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note on snapshots and branches.")
    (inbox / "b.pdf").write_bytes(b"not a pdf")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)

    onboard.onboard_source({"source": "demo"})

    assert source.stage == Stage.raw
    assert source.raw["reasons"]["file.unreadable"] == 1


# a stylesheet beside the chapters is left out and named, and the verdict reads the chapters alone
def test_a_file_no_engine_reads_is_skipped_not_failed(tmp_path, monkeypatch):
    from models.corpus import Stage

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note on snapshots and branches.")
    (inbox / "style.css").write_text("body { margin: 0 }")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)

    onboard.onboard_source({"source": "demo"})

    assert source.stage == Stage.raw
    assert source.raw["skipped"] == {"style.css": ".css"}
    assert source.raw["units"] == 1 and "file.unreadable" not in source.raw["reasons"]


def test_a_git_path_stays_inside_the_clone(tmp_path):
    with pytest.raises(Final, match="not a folder of the repository"):
        source_intake._clone({"repo": "https://x.org/r.git", "path": "../../"}, tmp_path / "repo")


def test_a_cancel_stops_before_the_next_piece(tmp_path, monkeypatch):
    from models.corpus import Stage

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note.")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.job_queue, "is_cancelled", lambda id: True)

    onboard.onboard_source({"source": "demo", "_job_id": 7})

    assert source.stage == Stage.declared


def test_a_source_under_a_job_or_under_questions_is_not_removed():
    source = DataSource(name="a", kind="local", stage="raw")

    assert source_intake.removal_refusal(source, None, 0) is None
    assert "job 9" in source_intake.removal_refusal(source, 9, 0)
    assert "3 questions" in source_intake.removal_refusal(source, None, 3)


def test_the_searched_variant_and_a_variant_under_a_job_are_not_removed():
    assert source_intake.variant_refusal("smoke_x", "baseline", None) is None
    assert "searches" in source_intake.variant_refusal("baseline", "baseline", None)
    assert "job 4" in source_intake.variant_refusal("smoke_x", "baseline", 4)
    assert "not a variant name" in source_intake.variant_refusal("../x", "baseline", None)


def test_removal_takes_the_sources_own_folders_and_not_a_neighbours_with_a_longer_name(tmp_path):
    for name in ("a_0b6a96c6", "a_1234abcd", "a_b_0b6a96c6", "a-x_0b6a96c6", "_fetched/a", "_fetched/a_b"):
        (tmp_path / name).mkdir(parents=True)

    own = source_intake.stand_files(DataSource(name="a", kind="local"), tmp_path)

    assert [str(p.relative_to(tmp_path)) for p in own] == ["_fetched/a", "a_0b6a96c6", "a_1234abcd"]


def test_a_folder_mark_holds_its_source_as_the_stands_gold_predicate_reads_it():
    import db

    files = ["roadmap/content/clickhouse/intro.md", "roadmap/content/clickhouse/joins.md"]
    marks = [["roadmap/content/clickhouse"], ["other/file.md"], ["roadmap/content/clickhouse/joins.md"]]
    assert db.count_marking(files, marks) == 2


def test_a_source_is_accepted_from_raw_and_a_bad_verdict_needs_a_reason():
    raw = DataSource(name="a", kind="local", stage="raw", raw={"verdict": "dirty"})
    bad = DataSource(name="b", kind="local", stage="raw", raw={"verdict": "bad"})
    declared = DataSource(name="c", kind="local", stage="declared", raw={})

    assert source_intake.accept_refusal(raw, None) is None
    assert "needs a reason" in source_intake.accept_refusal(bad, None)
    assert source_intake.accept_refusal(bad, "tables lost, prose fine") is None
    assert "accepted from raw" in source_intake.accept_refusal(declared, None)
    kept = source_intake.accepted_raw(bad, "tables lost, prose fine")
    assert kept["verdict"] == "bad" and kept["accepted_despite"] == "tables lost, prose fine" and "accepted_at" in kept


def test_only_an_accepted_source_goes_into_search():
    raw = DataSource(name="a", kind="local", stage="raw")

    assert "only an accepted source" in source_intake.active_refusal(raw, True)
    assert source_intake.active_refusal(raw, False) is None
    assert source_intake.active_refusal(DataSource(name="b", kind="local", stage="accepted"), True) is None


def test_a_source_is_not_accepted_while_its_onboard_waits_or_runs():
    raw = DataSource(name="a", kind="local", stage="raw", raw={"verdict": "ok"})

    assert "onboard job 5" in source_intake.accept_refusal(raw, None, 5)


# a run may name a settings file of its own; a resume converts a piece again when its settings or the route moved
def test_a_piece_is_converted_again_when_its_settings_or_the_route_change(tmp_path, monkeypatch):
    import json

    import config
    from models.corpus import Stage
    from use_cases import route

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "b.pdf").write_bytes(b"%PDF")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)
    monkeypatch.setattr(onboard.route, "seamless_pieces", lambda file, pages, size, rule=None: pieces(*pages, size))
    monkeypatch.setattr(onboard.route, "layer_texts", lambda file, rule=None: ["page words here"] * 10)
    own = {"settings": None}
    one_run = lambda file, rule=None: [route.Run("docling", (1, 10), "text layer", {}, own["settings"])]  # noqa: E731
    monkeypatch.setattr(onboard.route, "route", one_run)
    calls = []

    def fake_piece(file, engine, piece, settings, language, rule=None):
        calls.append(settings[1])
        return {"markdown": "## Page\n\npage words here", "structure": None, "seconds": 1.0, "status": None}

    monkeypatch.setattr(onboard.reading, "convert_piece", fake_piece)
    onboard.onboard_source({"source": "demo"})
    folder = next((tmp_path / "raw").glob("demo_*"))
    row = json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-10"]
    assert row["settings"] == "docling/default"

    calls.clear()
    own["settings"] = reread = config.settings.intake.route.reread_settings
    onboard.onboard_source({"source": "demo"})
    row = json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-10"]
    assert len(calls) == 1 and row["settings"] == reread
    assert source.raw["pages_by_settings"] == {reread: 10}

    calls.clear()
    onboard.onboard_source({"source": "demo"})
    assert calls == []

    record = json.loads((folder / "record.json").read_text())
    record["route_sha256"] = "another"
    (folder / "record.json").write_text(json.dumps(record))
    onboard.onboard_source({"source": "demo"})
    assert len(calls) == 1


# a low piece is read again, the better reading stays unless it loses table cells, and a resume reads it no third time
def test_a_low_piece_is_read_again_and_the_better_reading_stays(tmp_path, monkeypatch):
    import json

    import config
    from models.corpus import Stage
    from use_cases import route

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "b.pdf").write_bytes(b"%PDF")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)
    layer = "the quick brown fox jumps over the lazy dog near the river bank today"
    monkeypatch.setattr(onboard.route, "seamless_pieces", lambda file, pages, size, rule=None: pieces(*pages, size))
    monkeypatch.setattr(onboard.route, "layer_texts", lambda file, rule=None: [layer])
    page_run = [route.Run("docling", (1, 1), "text layer", {})]
    monkeypatch.setattr(onboard.route, "route", lambda file, rule=None: page_run)
    second = {"text": layer, "table": ""}
    calls = []

    def fake_piece(file, engine, piece, settings, language, rule=None):
        calls.append(settings[0]["fields"].get("pdf_backend", "default"))
        if settings[0]["fields"].get("pdf_backend") == "pypdfium2":
            text, table = second["text"], second["table"]
        else:
            text, table = "thequick brownfox jumpsover thelazy dog near the river bank today", "| a | b |\n|---|---|\n"
        return {"markdown": f"## Page\n\n{text}\n\n{table}", "structure": None, "seconds": 1.0, "status": None}

    monkeypatch.setattr(onboard.reading, "convert_piece", fake_piece)
    second["table"] = "| a | b |\n|---|---|\n"
    onboard.onboard_source({"source": "demo"})
    folder = next((tmp_path / "raw").glob("demo_*"))
    row = json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-1"]
    assert calls == ["default", "pypdfium2"]
    assert row["reread"]["taken"] and row["settings"] == config.settings.intake.route.reread_settings
    assert "brown fox" in (folder / "pieces" / f"{onboard._stem('b.pdf#1-1')}.md").read_text()

    calls.clear()
    onboard.onboard_source({"source": "demo"})
    assert calls == []

    # a second reading that loses the table does not replace the first
    calls.clear()
    second["table"] = ""
    record = json.loads((folder / "record.json").read_text())
    record["route_sha256"] = "another"
    (folder / "record.json").write_text(json.dumps(record))
    onboard.onboard_source({"source": "demo"})
    row = json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-1"]
    assert calls == ["default", "pypdfium2"]
    assert not row["reread"]["taken"] and row["settings"] == "docling/default"


# a source's own knobs lie over the stand's; its file speaks for it over its row, and a wrong value is refused
def test_a_source_s_knobs_lie_over_the_stand_s(monkeypatch):
    from types import SimpleNamespace

    import config

    rule, names = source_intake.intake_rule({"mono_spread": 0.12, "settings": {"docling": "docling/pypdfium2"}})
    assert rule.mono_spread == 0.12 and rule.seam_window == config.settings.intake.route.seam_window
    assert names["docling"] == "docling/pypdfium2" and names["mineru"] == config.settings.intake.settings["mineru"]
    with pytest.raises(ValueError):
        source_intake.intake_rule({"seam_margin": 0.9})

    row = {"folder": "inbox/x", "intake": {"mono_spread": 0.2}}
    monkeypatch.setattr(source_intake.files, "source_files", lambda: {})
    assert source_intake.intake_block("x", row) == {"mono_spread": 0.2}
    from sources.declaration import IntakeOverride

    filed = SimpleNamespace(intake=IntakeOverride(seam_window=1))
    monkeypatch.setattr(source_intake.files, "source_files", lambda: {"x": filed})
    assert source_intake.intake_block("x", row) == {"settings": {}, "seam_window": 1}
    assert "source file" in source_intake.intake_refusal(DataSource(name="x"), None)
    assert source_intake.with_intake(row, {}) == {"folder": "inbox/x"}


# a source that names its own settings file is read with it, and the report says which knobs it set
def test_onboarding_reads_a_source_with_its_own_settings(tmp_path, monkeypatch):
    import json

    from models.corpus import Stage
    from use_cases import route

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "b.pdf").write_bytes(b"%PDF")
    origin = {"folder": "inbox/demo", "intake": {"settings": {"docling": "docling/pypdfium2"}}}
    source = DataSource(id=1, name="demo", kind="local", language="en", stage=Stage.declared, origin=origin)
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)
    monkeypatch.setattr(onboard.source_intake.files, "source_files", lambda: {})
    monkeypatch.setattr(onboard.route, "layer_texts", lambda file, rule=None: ["page words here"])
    page_run = [route.Run("docling", (1, 1), "text layer", {})]
    monkeypatch.setattr(onboard.route, "route", lambda file, rule=None: page_run)
    monkeypatch.setattr(onboard.route, "seamless_pieces", lambda file, pages, size, rule=None: pieces(*pages, size))
    read = {"markdown": "## P\n\npage words here", "structure": None, "seconds": 1.0, "status": None}
    monkeypatch.setattr(onboard.reading, "convert_piece", lambda *args, **kwargs: read)

    onboard.onboard_source({"source": "demo"})

    folder = next((tmp_path / "raw").glob("demo_*"))
    assert json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-1"]["settings"] == "docling/pypdfium2"
    assert source.raw["intake"] == {"settings": {"docling": "docling/pypdfium2"}}


# the route's fingerprint moves with what shapes a piece and with the reread file's content, not a skipped page type
def test_the_route_fingerprint_reads_what_shapes_a_piece(monkeypatch):
    import config
    from job_handlers import reading

    content = {"sha": "one"}
    monkeypatch.setattr(reading, "load_settings", lambda name: ({}, content["sha"]))
    stand = config.settings.intake.route
    before = reading.route_sha(stand)
    assert reading.route_sha(stand.model_copy(update={"epub_skip": ["toc"], "suspect_min_words": 99})) == before
    assert reading.route_sha(stand.model_copy(update={"seam_window": stand.seam_window + 1})) != before
    content["sha"] = "two"
    assert reading.route_sha(stand) != before


# a piece end the seam rule moved counts once against a plain cut of the same run
def test_seams_moved_count_the_ends_the_seam_rule_shifted():
    from pathlib import Path

    from use_cases import route

    run = route.Run("docling", (1, 30), "text layer", {})
    loaded = {"docling/default": ({"pages_per_chunk": 10}, "sha")}
    file = Path("b.pdf")
    plain = [(file, "b.pdf", run, p, "docling/default") for p in [(1, 10), (11, 20), (21, 30)]]
    moved = [(file, "b.pdf", run, p, "docling/default") for p in [(1, 11), (12, 20), (21, 30)]]
    assert onboard._seams_moved(plain, loaded) == 0
    assert onboard._seams_moved(moved, loaded) == 1
