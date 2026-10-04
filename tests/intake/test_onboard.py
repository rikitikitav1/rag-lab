from pathlib import Path

import pytest
from config import settings
from corpus_keys import file_stem
from job_handlers import onboard
from job_handlers.base import Final
from models.corpus import DataSource
from use_cases import intake_fetch, source_intake
from use_cases.converting import pieces

# the stand's own rules, named where a test reads with them rather than taken as a silent default
STAND = settings.intake.route


# the stand's switch is its own choice; these tests walk the raw path unless one sets the switch itself
@pytest.fixture(autouse=True)
def _no_auto_accept(monkeypatch):
    import config

    monkeypatch.setattr(config.settings.intake.quality, "auto_accept_ok", False)
    monkeypatch.setattr(source_intake, "index_waiting", lambda name: None)


# a PDF run is cut by the seam rule at the settings' piece size, a whole file is one unit
def test_a_run_is_cut_into_pieces_of_the_settings_size(tmp_path, monkeypatch):
    from use_cases import route

    runs = {".pdf": [route.Run("docling", (1, 120), "text layer")], ".md": [route.Run(None, None, "markdown")]}
    monkeypatch.setattr(onboard.route, "route", lambda file, rule=None: runs[file.suffix])
    monkeypatch.setattr(onboard.route, "seamless_pieces", lambda file, pages, size, rule=None: pieces(*pages, size))
    names, loaded = {"docling": "docling/default"}, {}

    planned = onboard.reading.plan(tmp_path / "a.pdf", names, loaded, STAND)
    assert [(piece, name) for _, piece, name in planned] == [
        ((1, 50), "docling/default"),
        ((51, 100), "docling/default"),
        ((101, 120), "docling/default"),
    ]
    assert [piece for _, piece, _ in onboard.reading.plan(tmp_path / "a.md", names, loaded, STAND)] == [None]


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
    source = DataSource(name="a", kind="local", declaration={"folder": "../../etc"})

    with pytest.raises(Final, match="not a folder of the stand"):
        intake_fetch.gather(source, tmp_path / "fetched", tmp_path)


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
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
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
        failed, partial = piece == (51, 100), piece == (101, 120)
        status = "failure" if failed else "partial_success" if partial else "success"
        return {
            "markdown": f"## Pages {piece[0]}\n\n{text}",
            "structure": None,
            "seconds": 1.0,
            "status": {"status": status, "errors": ["boom"] if failed else []},
        }

    monkeypatch.setattr(onboard.reading, "convert_piece", fake_piece)

    onboard.onboard_source({"source": "demo"})

    folder = next((tmp_path / "raw").glob("demo@*"))
    record = json.loads((folder / "record.json").read_text())
    assert sorted(record["units"]) == ["a.md", "b.pdf#1-50", "b.pdf#101-120", "b.pdf#51-100"]
    assert "conversion.failed" in record["units"]["b.pdf#51-100"]["breached"]
    assert "conversion.partial" in record["units"]["b.pdf#101-120"]["breached"], "a partial piece is kept and said"
    assert record["units"]["b.pdf#1-50"]["engine"] == "docling"
    assert record["units"]["b.pdf#1-50"]["layer_f1"] > 0.95
    whole = (folder / "files" / f"{file_stem('b.pdf')}.md").read_text()
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
    (folder / "pieces" / f"{file_stem('b.pdf#1-50')}.md").unlink()
    onboard.onboard_source({"source": "demo"})
    assert calls == [("b.pdf", (1, 50)), ("b.pdf", (51, 100))]

    # a changed file is converted again, and a file that left takes its rows with it
    calls.clear()
    (inbox / "a.md").write_text("## Notes\n\nAnother note.")
    (inbox / "b.pdf").unlink()
    onboard.onboard_source({"source": "demo"})
    assert calls == [("a.md", None)]
    assert sorted(json.loads((folder / "record.json").read_text())["units"]) == ["a.md"]


def test_an_accepted_source_is_read_again_beside_the_accepted_run(tmp_path, monkeypatch):
    from models.corpus import Stage
    from use_cases import route

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note on snapshots and branches.")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
    )
    for module in (onboard, source_intake):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)
    monkeypatch.setattr(onboard.route, "route", lambda file, rule=None: [route.Run(None, None, "markdown")])
    calls = []

    def fake_piece(file, engine, piece, settings, language, rule=None):
        calls.append(file.name)
        return {"markdown": file.read_text(), "seconds": 0.0, "status": None, "structure": None}

    monkeypatch.setattr(onboard.reading, "convert_piece", fake_piece)

    onboard.onboard_source({"source": "demo"})
    source_intake.promote(source, None)
    accepted = source.raw["folder"]

    calls.clear()
    assert onboard.onboard_source({"source": "demo"}) == {"unchanged": True, "refetched": True}
    assert calls == [] and source.raw["folder"] == accepted

    (inbox / "a.md").write_text("## Notes\n\nAnother note.")
    onboard.onboard_source({"source": "demo"})
    assert source.stage == Stage.accepted and source.raw["folder"] == accepted, "search keeps the accepted run"
    new = source.raw["candidate"]["folder"]
    assert new != accepted and (tmp_path / accepted).is_dir()

    # a second run before the accept replaces the candidate, and the first candidate's folder goes
    (inbox / "a.md").write_text("## Notes\n\nA third note.")
    onboard.onboard_source({"source": "demo"})
    newer = source.raw["candidate"]["folder"]
    assert newer != new and not (tmp_path / new).exists() and (tmp_path / accepted).is_dir()
    owned = {p.name for p in source_intake.stand_files(source, tmp_path / "raw")}
    assert owned >= {Path(accepted).name, Path(newer).name}

    replaced = source_intake.promote(source, None)
    assert source.raw["folder"] == newer and "candidate" not in source.raw and replaced == accepted
    assert (tmp_path / accepted).is_dir(), "the folder goes only after the row is committed"
    source_intake.drop_folder(replaced)
    assert not (tmp_path / accepted).exists()


def test_an_accepted_source_is_accepted_again_only_with_a_new_run_waiting():
    plain = DataSource(name="a", kind="local", stage="accepted", raw={"verdict": "ok"})
    run = {"verdict": "ok", "candidate": {"verdict": "bad"}}
    waiting = DataSource(name="b", kind="local", stage="accepted", raw=run)

    assert "new run waiting" in source_intake.accept_refusal(plain, None)
    assert "needs a reason" in source_intake.accept_refusal(waiting, None)
    assert source_intake.accept_refusal(waiting, "tables lost") is None
    # the accepted run's folder goes with the accept, so an index reading it refuses the accept
    assert "index job 9" in source_intake.accept_refusal(waiting, "tables lost", None, 9)


def test_two_keys_never_share_a_file():
    assert file_stem("docs/a.pdf#1-10") != file_stem("docs_a.pdf#1-10")


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

    monkeypatch.setattr(intake_fetch.fetch.requests, "get", lambda url, **kw: _Got(url))
    a, fresh_a = intake_fetch._download("https://x.org/intro/index.html", tmp_path)
    b, _ = intake_fetch._download("https://x.org/api/index.html", tmp_path)
    again, fresh_again = intake_fetch._download("https://x.org/intro/index.html", tmp_path)
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
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
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
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
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
        intake_fetch._clone({"repo": "https://x.org/r.git", "path": "../../"}, tmp_path / "repo")


def test_a_cancel_stops_before_the_next_piece(tmp_path, monkeypatch):
    from models.corpus import Stage

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note.")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
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
    names = ("a@0b6a96c6", "a@1234abcd@9f8e7d6c", "a_b@0b6a96c6", "a-x@0b6a96c6", "_fetched/a", "_fetched/a_b")
    for name in names:
        (tmp_path / name).mkdir(parents=True)

    own = source_intake.stand_files(DataSource(name="a", kind="local"), tmp_path)

    assert [str(p.relative_to(tmp_path)) for p in own] == ["_fetched/a", "a@0b6a96c6", "a@1234abcd@9f8e7d6c"]


# a name may end in what reads as a hash: `foo_deadbeef`'s run is never `foo`'s, and an older folder goes by its row
def test_a_source_named_like_a_hash_keeps_its_folders_from_a_shorter_names_removal(tmp_path):
    arm, fingerprint = "1234abcd", "9f8e7d6c" * 8
    theirs = source_intake.raw_folder(tmp_path, "foo_deadbeef", arm)
    mine = source_intake.raw_folder(tmp_path, "foo", "deadbeef", fingerprint)
    older = tmp_path / "foo_0b6a96c6"
    for folder in (theirs, mine, older, tmp_path / "foo_deadbeef_1234abcd"):
        folder.mkdir()

    row = DataSource(name="foo", kind="local", raw={"folder": f"datasets/raw_sources/{older.name}"})
    own = {p.name for p in source_intake.stand_files(row, tmp_path)}

    assert own == {mine.name, older.name}
    assert source_intake.raw_folder(tmp_path, "foo", "0b6a96c6") == older, "a run begun under the older mark resumes"


def test_a_folder_mark_holds_its_source_as_the_stands_gold_predicate_reads_it():
    import db

    files = ["roadmap/content/clickhouse/intro.md", "roadmap/content/clickhouse/joins.md"]
    from corpus_keys import Gold

    marks = [["roadmap/content/clickhouse"], ["other/file.md"], ["roadmap/content/clickhouse/joins.md"]]
    assert db.count_marking(files, marks) == 2
    exact = [Gold(("roadmap/content/clickhouse/intro.md",), "Intro"), Gold(("roadmap/content/clickhouse",), "Intro")]
    assert db.count_marking(files, exact) == 1, "an exact gold names its file whole, never a folder"


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
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
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
    folder = next((tmp_path / "raw").glob("demo@*"))
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
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
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
    folder = next((tmp_path / "raw").glob("demo@*"))
    row = json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-1"]
    assert calls == ["default", "pypdfium2"]
    assert row["reread"]["taken"] and row["settings"] == config.settings.intake.route.reread_settings
    assert "brown fox" in (folder / "pieces" / f"{file_stem('b.pdf#1-1')}.md").read_text()

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
def test_a_source_s_knobs_lie_over_the_stand_s():
    import config

    rule, names = source_intake.intake_rule({"mono_spread": 0.12, "settings": {"docling": "docling/pypdfium2"}})
    assert rule.mono_spread == 0.12 and rule.seam_window == config.settings.intake.route.seam_window
    assert names["docling"] == "docling/pypdfium2" and names["mineru"] == config.settings.intake.settings["mineru"]
    with pytest.raises(ValueError):
        source_intake.intake_rule({"seam_margin": 0.9})

    row = {"folder": "inbox/x", "intake": {"mono_spread": 0.2}}
    assert source_intake.intake_block(row) == {"mono_spread": 0.2}
    assert source_intake.intake_block(None) == {}
    # a row the seed writes from a file refuses the door; the file's `intake:` block speaks for it
    assert "source file" in source_intake.intake_refusal(DataSource(name="x", seeded=True), None)
    assert source_intake.intake_refusal(DataSource(name="x", seeded=False), None) is None
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
    source = DataSource(id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration=origin)
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

    folder = next((tmp_path / "raw").glob("demo@*"))
    assert json.loads((folder / "record.json").read_text())["units"]["b.pdf#1-1"]["settings"] == "docling/pypdfium2"
    assert source.raw["intake"] == {"settings": {"docling": "docling/pypdfium2"}}


# the route's fingerprint moves with what shapes a piece and with the reread file's content, not a skipped page type
def test_the_route_fingerprint_reads_what_shapes_a_piece(monkeypatch):
    import config
    from use_cases import reading

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


# the language is the declared one, else read from the source's own text before any piece, and a scan must declare it
def test_a_source_s_language_is_declared_or_read_from_its_text_and_a_bare_scan_must_declare_it(tmp_path, monkeypatch):
    from use_cases import route

    note = tmp_path / "a.md"
    note.write_text("Процесс создаётся вызовом fork, и потомок исполняет ту же программу.")
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF")
    monkeypatch.setattr(route, "layer_texts", lambda path: ["", ""])
    language = intake_fetch.source_language
    assert language(DataSource(name="x", declaration={"language": "en"}), [note], {}) == "en"
    assert language(DataSource(name="x", declaration={}), [note], {}) == "ru"
    # what an earlier run found is on the row, and a new run reads the text again rather than taking it as declared
    assert language(DataSource(name="x", declaration={}, language="en"), [note], {}) == "ru"
    with pytest.raises(Final, match="declare its language"):
        language(DataSource(name="x", declaration={}), [scan], {})


def test_a_markdown_only_source_is_read_from_its_tree_and_a_converted_one_from_its_folder():
    from types import SimpleNamespace

    from job_handlers import onboard
    from paths import ROOT

    md = SimpleNamespace(engine=None)
    pdf = SimpleNamespace(engine="docling")
    tree, folder = ROOT / "datasets" / "t", ROOT / "datasets" / "raw_sources" / "x_1"
    assert onboard._index_root([(None, "a.md", md, None, None)], tree, folder) == ("datasets/t", "tree")
    both = [(None, "a.md", md, None, None), (None, "b.pdf", pdf, None, None)]
    assert onboard._index_root(both, tree, folder) == ("datasets/raw_sources/x_1", "converted")


# a tree the stand fetched is copied into the run: the next fetch resets the clone, not the accepted run's text
def test_a_fetched_tree_is_read_from_its_runs_own_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from job_handlers import onboard

    fetched = tmp_path / "_fetched"
    clone = fetched / "docs" / "repo"
    (clone / ".git").mkdir(parents=True)
    (clone / "a.md").write_text("# A\n")
    monkeypatch.setattr(onboard, "FETCHED", fetched)
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    run = tmp_path / "raw" / "docs_1"
    root, kind = onboard._index_root([(None, "a.md", SimpleNamespace(engine=None), None, None)], clone, run)
    assert (kind, root) == ("tree", "raw/docs_1/tree")
    (clone / "a.md").write_text("# moved upstream\n")
    assert (run / "tree" / "a.md").read_text() == "# A\n" and not (run / "tree" / ".git").exists()


# a family's row is one repository of the family, named by the row, cloned with the family's include
def test_a_family_row_is_gathered_as_its_own_repository(tmp_path, monkeypatch):
    declared = {"name": "repos", "git_family": {"base_url": "https://github.com/x", "repos": ["a-repo"]}}
    source = DataSource(name="a-repo", kind="git", declaration=declared)
    seen = {}

    def clone(git, folder):
        seen.update(git)
        folder.mkdir(parents=True)
        (folder / "README.md").write_text("# A\n")
        return folder, {}

    monkeypatch.setattr(intake_fetch, "_clone", clone)
    root, files, *_ = intake_fetch.gather(source, tmp_path / "inbox", tmp_path)
    assert seen["repo"] == "https://github.com/x/a-repo" and [f.name for f in files] == ["README.md"]


# a converted source read under a settings file, a choice or a route the stand no longer has is named, a tree is not
def test_a_converted_source_says_what_moved_since_it_was_read(tmp_path, monkeypatch):
    import json

    folder = tmp_path / "raw" / "book_1"
    folder.mkdir(parents=True)
    record = {"settings": {"docling": "docling/default"}, "settings_sha256": {"docling": "old"}, "route_sha256": "r1"}
    (folder / "record.json").write_text(json.dumps(record))
    monkeypatch.setattr(source_intake, "ROOT", tmp_path)
    monkeypatch.setattr("use_cases.converting.load_settings", lambda name: ({}, "old"))
    monkeypatch.setattr("use_cases.reading.route_sha", lambda rule: "r1")
    book = DataSource(name="book", declaration={"name": "book"}, raw={"folder": "raw/book_1"})
    assert source_intake.conversion_drift(book) == {}
    monkeypatch.setattr("use_cases.converting.load_settings", lambda name: ({}, "new"))
    assert source_intake.conversion_drift(book)["edited"] == ["docling"]
    monkeypatch.setattr("use_cases.reading.route_sha", lambda rule: "r2")
    assert source_intake.conversion_drift(book)["route"] is True
    tree = DataSource(name="md", raw={"folder": "raw/md_1", "root_kind": "tree"})
    assert source_intake.conversion_drift(tree) is None

    # a settings file the job itself chose is the run's own choice, not a move of the source
    monkeypatch.setattr("use_cases.reading.route_sha", lambda rule: "r1")
    monkeypatch.setattr("use_cases.converting.load_settings", lambda name: ({}, "old"))
    chosen = {"docling": "docling/pypdfium2"}
    record = {**record, "settings": chosen, "settings_override": chosen}
    (folder / "record.json").write_text(json.dumps(record))
    assert source_intake.conversion_drift(book) == {}


# an ok verdict walks straight to accepted when the stand says so, and waits at raw when it does not
@pytest.mark.parametrize("switch", [True, False])
def test_an_ok_raw_source_is_accepted_with_no_person_only_when_the_switch_is_on(tmp_path, monkeypatch, switch):
    import config
    from models.corpus import Stage

    inbox = tmp_path / "inbox" / "demo"
    inbox.mkdir(parents=True)
    (inbox / "a.md").write_text("## Notes\n\nA short note on snapshots and branches.")
    source = DataSource(
        id=1, name="demo", kind="local", language="en", stage=Stage.declared, declaration={"folder": "inbox/demo"}
    )
    monkeypatch.setattr(onboard, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)
    monkeypatch.setattr(onboard, "_verdict", lambda rows, sections: ("ok", {}, {"pieces": 0, "sections": 0}))
    quality = config.settings.intake.quality.model_copy(update={"auto_accept_ok": switch})
    monkeypatch.setattr(config.settings.intake, "quality", quality)

    onboard.onboard_source({"source": "demo"})

    assert source.raw["root_kind"] == "tree" and source.raw["root"] == "inbox/demo"
    if switch:
        assert source.stage == Stage.accepted and source.raw["accepted_by"] == "auto"
    else:
        assert source.stage == Stage.raw and "accepted_at" not in source.raw


# a new run accepted under an indexed source leaves its variants cut from the old run, and drift says so
def test_accepting_a_new_run_marks_the_variants_cut_from_the_old_one_as_moved():
    from sources import files
    from sources.declaration import Declaration

    declaration = Declaration(name="book", urls=["https://example.org/b.pdf"]).model_dump(mode="json")
    digest = files.digest(Declaration.model_validate(declaration))
    raw = {"folder": "raw/book@1", "verdict": "ok", "candidate": {"folder": "raw/book@2", "verdict": "ok"}}
    row = DataSource(
        name="book", kind="urls", stage="accepted", raw=raw, declaration=declaration, indexed_with={"v1": digest}
    )
    assert files.drift(row.name, row.indexed_with, row.declaration)["moved"] == []

    source_intake.promote(row, None)

    assert files.drift(row.name, row.indexed_with, row.declaration)["moved"] == ["v1"]


# an ok run auto-accepted over an earlier waiting one names both folders it leaves: the waiting run's and the accepted's
def test_an_auto_accepted_run_drops_the_run_it_replaced_and_the_one_that_waited(monkeypatch):
    import config

    monkeypatch.setattr(config.settings.intake.quality, "auto_accept_ok", True)
    raw = {"folder": "raw/book@1", "verdict": "ok", "candidate": {"folder": "raw/book@2", "verdict": "bad"}}
    row = DataSource(name="book", kind="urls", stage="accepted", raw=raw)

    gone = source_intake.take_run(row, {"folder": "raw/book@3", "verdict": "ok", "language": "en"})

    assert sorted(gone) == ["raw/book@1", "raw/book@2"] and row.raw["folder"] == "raw/book@3"


# an index claimed between an accept and its commit reads the old folder, so the folder stays until it is done
def test_a_replaced_folder_stays_while_an_index_reads_the_source(tmp_path, monkeypatch):
    monkeypatch.setattr(source_intake, "ROOT", tmp_path)
    monkeypatch.setattr(source_intake, "RAW", tmp_path / "raw")
    (tmp_path / "raw" / "book@1").mkdir(parents=True)
    monkeypatch.setattr(source_intake, "index_waiting", lambda name: 12)

    source_intake.drop_folder("raw/book@1", "book")
    assert (tmp_path / "raw" / "book@1").is_dir()

    monkeypatch.setattr(source_intake, "index_waiting", lambda name: None)
    source_intake.drop_folder("raw/book@1", "book")
    assert not (tmp_path / "raw" / "book@1").exists()


# the reason kept beside a bad verdict is bounded in the refusal both doors call, not in one door's model
def test_an_accept_reason_is_bounded_at_every_door():
    row = DataSource(name="book", kind="urls", stage="raw", raw={"verdict": "bad"})

    assert "3 to 500" in source_intake.accept_refusal(row, "ok")
    assert "3 to 500" in source_intake.accept_refusal(row, "x" * 501)
    assert source_intake.accept_refusal(row, "tables read by hand") is None


# a url source's inbox is kept between runs, so an unchanged answer says it read no new copy of the upstream
def test_an_unchanged_url_source_says_its_inbox_was_not_fetched_again(tmp_path, monkeypatch):
    from models.corpus import Stage
    from use_cases import route

    source = DataSource(
        id=1, name="demo", kind="urls", stage=Stage.declared,
        declaration={"urls": ["https://example.org/a.md"], "language": "en"},
    )
    for module in (onboard, source_intake):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "RAW", tmp_path / "raw")
    monkeypatch.setattr(onboard, "FETCHED", tmp_path / "raw" / "_fetched")
    monkeypatch.setattr(onboard, "Session", lambda: _Session(source))
    monkeypatch.setattr(onboard.measurements, "record", lambda kind, name, payload, bulk=(): str(tmp_path / "m.json"))
    monkeypatch.setattr(onboard.measurements, "ROOT", tmp_path)
    monkeypatch.setattr(onboard.route, "route", lambda file, rule=None: [route.Run(None, None, "markdown")])

    def download(url, target):
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("## Notes\n\nA short note on snapshots and branches.")
        return True

    monkeypatch.setattr(intake_fetch.fetch, "download", download)
    monkeypatch.setattr(
        onboard.reading, "convert_piece",
        lambda file, *a, **kw: {"markdown": file.read_text(), "seconds": 0.0, "status": None, "structure": None},
    )
    onboard.onboard_source({"source": "demo"})
    source_intake.promote(source, None)

    assert onboard.onboard_source({"source": "demo"}) == {"unchanged": True, "refetched": False}


# a site whose declared release moved fetches its pages anew; the same release keeps what it fetched
def test_a_site_whose_release_moved_fetches_its_pages_anew(tmp_path, monkeypatch):
    from use_cases import fetch

    fetched = []

    def download(url, target):
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<div id='content'><p>page</p></div>")
        fetched.append(url)
        return True

    monkeypatch.setattr(fetch, "download", download)

    def source(release):
        declared = {"name": "site", "pages": ["https://x/a.html"], "site": {"main": "div#content", "release": release}}
        return DataSource(name="site", kind="pages", declaration=declared)

    releases = (None, "1.0", "1.0", "1.1")
    states = [intake_fetch.gather(source(r), tmp_path / "inbox", tmp_path).fetched for r in releases]

    assert fetched == ["https://x/a.html"] * 3, "the first release declared over kept pages fetches them too"
    assert ["fetched_at" in state for state in states] == [True, True, False, True], "what the record calls refetched"
    assert (tmp_path / "inbox" / "release").read_text() == "1.1"


# a fresh run reads every piece by its tool: the kept reading is passed over, and the flag is gone after the job
def test_a_fresh_reading_passes_the_kept_one_over(monkeypatch, tmp_path):
    from use_cases import converting

    monkeypatch.setattr(converting, "READINGS", tmp_path)
    monkeypatch.setattr(converting, "reading_key", lambda *a: "ab" + "0" * 62)
    kept = tmp_path / "ab" / ("ab" + "0" * 62 + ".json")
    kept.parent.mkdir()
    kept.write_text('{"markdown": "old", "status": "success", "errors": [], "seconds": 3}')
    read = []
    fresh = {"markdown": "new", "status": "success", "errors": [], "seconds": 1}
    monkeypatch.setitem(converting.PIECE, converting.Tool("docling"), lambda *a: read.append(1) or fresh)
    assert converting.convert(None, "docling", tmp_path / "f.pdf", [], None, 10)["markdown"] == "old"
    hold = converting.CardHold(lambda spec: None, lambda spec: None)
    with converting.reading_fresh(True), converting.card_hold(hold):
        assert converting.convert(None, "docling", tmp_path / "f.pdf", [], None, 10)["markdown"] == "new"
    assert read == [1] and converting._FRESH.get() is False


# a file the run no longer reads (its origin moved from a folder to a link) leaves the folder with its markdown
def test_a_run_drops_the_markdown_of_a_file_it_no_longer_reads(tmp_path):
    import config
    from job_handlers import onboard

    (tmp_path / "files").mkdir()
    (tmp_path / "pieces").mkdir()
    rel = "b8472833/notes.md"
    key = f"{rel}#1-1"
    (tmp_path / "pieces" / f"{file_stem(key)}.md").write_text("## A\n\ntext")
    stale = tmp_path / "files" / f"{file_stem('notes.md')}.md"
    stale.write_text("## A\n\nold")
    record = {"units": {key: {"key": key, "file": rel, "pages": None, "route": None}}}

    onboard._assemble([Path(rel)], {Path(rel): rel}, record, tmp_path, set(), config.settings.intake.route, {})

    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == [f"{file_stem(rel)}.md"]


# the board says who moves a source next: a person for a run that is not ok, a door for the plain next step
def test_the_next_step_of_a_source_names_who_moves_it():
    from types import SimpleNamespace

    from models.corpus import Stage
    from use_cases.intake_board import next_step

    def row(stage, raw=None, active=False):
        return SimpleNamespace(stage=stage, raw=raw or {}, active=active)

    assert next_step(row(Stage.declared), 0, []) == ("a door", "onboard it (onboard_source)")
    who, step = next_step(row(Stage.raw, {"verdict": "dirty"}), 0, [])
    assert who == "a person" and step.startswith("a person decides on a dirty run")
    assert next_step(row(Stage.raw, {"verdict": "ok"}), 0, ["onboard_source"]) == (
        "the queue", "waits for its queued onboard_source")
    waiting = row(Stage.accepted, {"verdict": "ok", "candidate": {"verdict": "ok"}})
    assert next_step(waiting, 10, [])[0] == "a person"
    assert next_step(row(Stage.accepted, {"verdict": "ok"}), 0, [])[1] == "index it into a variant (index_data)"
    assert next_step(row(Stage.accepted, {"verdict": "ok"}, active=True), 5, []) == (None, "nothing waits")
    # a source whose every chunk another, more trusted source keeps waits for nothing, not for an index
    copied = row(Stage.accepted, {"verdict": "ok", "copies_kept_by": {"v": {"pg-docs": 3}}})
    assert next_step(copied, 0, [], "v")[0] is None


# one page the sitemap lists and the site removed is left out with its answer; a server error still fails the fetch
def test_a_page_the_site_no_longer_serves_is_left_out_not_the_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import requests
    from use_cases import fetch

    def download(url, target):
        status = {"https://x/gone.html": 404, "https://x/broken.html": 503}.get(url)
        if status:
            raise requests.HTTPError(response=SimpleNamespace(status_code=status))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<div id='content'><p>page</p></div>")
        return True

    monkeypatch.setattr(fetch, "download", download)

    def source(pages):
        declared = {"name": "site", "pages": pages, "site": {"main": "div#content"}}
        return DataSource(name="site", kind="pages", declaration=declared)

    got = intake_fetch.gather(source(["https://x/a.html", "https://x/gone.html"]), tmp_path / "inbox", tmp_path)
    assert len(got.files) == 1 and list(got.gone) == ["https://x/gone.html"] and "404" in got.gone["https://x/gone.html"]
    with pytest.raises(requests.HTTPError):
        intake_fetch.gather(source(["https://x/broken.html"]), tmp_path / "inbox2", tmp_path)


# a source's skipped paths change on its row in place: deleting and declaring it again would drop its chunks
def test_skip_paths_are_set_on_the_row_and_refused_for_a_seeded_or_busy_source(monkeypatch):
    monkeypatch.setattr(source_intake, "_onboard_waiting", lambda name: None)
    row = DataSource(name="docs", seeded=False, declaration={"name": "docs", "folder": "inbox/docs", "licence": "MIT"})
    source_intake.set_skip_paths(row, ["release-notes/*"])
    assert row.declaration["skip_paths"] == ["release-notes/*"] and row.declaration["folder"] == "inbox/docs"
    source_intake.set_skip_paths(row, [])
    assert "skip_paths" not in row.declaration
    with pytest.raises(Final, match="source file"):
        source_intake.set_skip_paths(DataSource(name="x", seeded=True, declaration={"name": "x"}), ["a/*"])
    monkeypatch.setattr(source_intake, "_onboard_waiting", lambda name: 42)
    with pytest.raises(Final, match="42"):
        source_intake.set_skip_paths(row, ["a/*"])
