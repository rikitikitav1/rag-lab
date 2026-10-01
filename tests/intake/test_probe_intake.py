import pytest
from job_handlers import probe
from job_handlers.base import Final
from job_specs import Refused, check
from models.corpus import DataSource, Stage
from use_cases import route


def test_a_probe_names_real_knobs_and_pages_in_order():
    check("probe_intake", {"source": "a", "pages": [3, 5], "knobs": {"numbered_levels": True}})
    with pytest.raises(Refused):
        check("probe_intake", {"source": "a", "pages": [3, 5], "knobs": {"no_such_knob": True}})
    with pytest.raises(Refused):
        check("probe_intake", {"source": "a", "pages": [5, 3], "knobs": {}})


class _Session:
    def __init__(self, source):
        self.source = source

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def scalar(self, stmt):
        return self.source

    def expunge(self, obj):
        pass


def _stub(monkeypatch, tmp_path, source, read):
    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF")
    monkeypatch.setattr(probe, "Session", lambda: _Session(source))
    monkeypatch.setattr(probe, "_file", lambda source, wanted: book)
    layers = [f"page {n} words here" for n in range(1, 11)]
    monkeypatch.setattr(probe.route, "layer_texts", lambda file, rule=None: layers)
    run = route.Run("docling", (1, 10), "layer", {})
    planned = [(run, (2, 4), None)]
    monkeypatch.setattr(probe.reading, "plan", lambda file, names, loaded, rule, pages: planned)
    monkeypatch.setattr(probe.reading, "read_piece", read)


# the same pages read twice, the knobs over the source's own on the second, each side scored on its own
def test_a_probe_reads_the_pages_twice_and_scores_each_side(tmp_path, monkeypatch):
    source = DataSource(name="book", kind="local", language="en", stage=Stage.raw, declaration={"folder": "x"})
    seen = []

    def read(file, engine, piece, name, loaded, language, layer, rule):
        seen.append(rule.numbered_levels)
        text = " ".join(f"page {n} words here" for n in range(piece[0], piece[1] + 1))
        return {"markdown": text if rule.numbered_levels else text.replace("words", "wo rds")}, name, None

    _stub(monkeypatch, tmp_path, source, read)
    got = probe.probe_intake({"source": "book", "pages": [2, 4], "knobs": {"numbered_levels": True}})

    assert seen[0] != seen[-1] and seen[-1] is True
    assert got["pages"] == [2, 4] and got["knobs"] == {"numbered_levels": True}
    assert got["defects"]["before"] > got["defects"]["after"] == 0


# the language is read as onboarding reads it: declared, else from the layer; a scan with neither is refused
def test_a_probe_reads_the_language_as_onboarding_does(tmp_path, monkeypatch):
    source = DataSource(name="book", kind="local", stage=Stage.declared, declaration={"folder": "x"})
    seen = []

    def read(file, engine, piece, name, loaded, language, layer, rule):
        seen.append(language)
        return {"markdown": layer or ""}, name, None

    _stub(monkeypatch, tmp_path, source, read)
    probe.probe_intake({"source": "book", "pages": [2, 4], "knobs": {}})
    assert set(seen) == {"en"}

    monkeypatch.setattr(probe.route, "layer_texts", lambda file, rule=None: [""] * 10)
    with pytest.raises(Final, match="declare its language"):
        probe.probe_intake({"source": "book", "pages": [2, 4], "knobs": {}})
