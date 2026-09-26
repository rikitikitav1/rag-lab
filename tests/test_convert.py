import json

import pytest
from engines import converter_tools
from job_handlers import convert, converting


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
    called = []
    monkeypatch.setattr(convert, "convert", lambda *args: called.append(args))

    convert.convert_source(
        {"settings": "docling/default", "language": "en", "inputs": ["books/a.pdf"], "out": "run", "_job_id": 7}
    )

    assert called == []
    assert json.loads((gold / "runs" / "run" / "record.json").read_text())["cancelled_before"] == "books/a.pdf 1-50"


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

    result = converting.convert("spec", "docling", None, [], None, 1)

    assert restarted == ["spec"] and result["errors"][-1] == "child restarted"


def test_a_resume_under_another_language_is_refused():
    record = {"converted": {"a": {}}, "tool": "docling", "settings_sha256": "s", "build": None, "language": "en"}
    with pytest.raises(convert.Final, match="language"):
        convert._refuse_another_configuration(
            record, {"tool": "docling", "settings_sha256": "s", "build": None, "language": "ru"}
        )
