import pytest
from job_handlers import onboard
from job_handlers.base import Final
from models.corpus import DataSource
from use_cases import source_intake


def test_a_run_is_cut_into_pieces_of_the_settings_size():
    assert onboard._pieces((1, 120), 50) == [(1, 50), (51, 100), (101, 120)]
    assert onboard._pieces(None, 50) == [None]


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

    def fake_route(file):
        if file.suffix == ".pdf":
            return [route.Run("docling", (1, 120), "text layer", {"absorbed": []})]
        return [route.Run(None, None, "markdown")]

    monkeypatch.setattr(onboard.route, "route", fake_route)
    monkeypatch.setattr(onboard.route, "layer_texts", lambda file: [f"page {n} words here" for n in range(1, 121)])
    calls = []

    def fake_piece(file, engine, piece, settings, language):
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

    monkeypatch.setattr(onboard, "_convert_piece", fake_piece)

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
