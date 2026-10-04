# an agent reads what a dirty verdict was read on, a chapter at a time, from the run's own folder only
import pytest
from errors import Refusal
from use_cases import raw_text


@pytest.fixture
def run(tmp_path, monkeypatch):
    files = tmp_path / "datasets" / "raw_sources" / "book@1" / "files"
    files.mkdir(parents=True)
    (files / "book.md").write_text("# One\nfirst\n## One a\nsub\n# Two\nsecond\n", encoding="utf-8")
    monkeypatch.setattr(raw_text, "ROOT", tmp_path)
    monkeypatch.setattr(raw_text, "RAW", tmp_path / "datasets" / "raw_sources")
    monkeypatch.setattr(raw_text, "_files_dir", lambda name: ("datasets/raw_sources/book@1", files.resolve()))
    return files


def test_the_run_lists_its_files_and_reads_one_chapter_to_the_next_heading_of_its_level(run):
    assert raw_text.read("book")["files"] == [{"file": "book.md", "chars": 38}]
    assert raw_text.read("book", "book.md", "one")["text"] == "# One\nfirst\n## One a\nsub\n"
    part = raw_text.read("book", "book.md", offset=6, limit=6)
    assert part["text"] == "first\n" and part["more"]


def test_a_path_outside_the_run_and_a_missing_heading_are_refused(run):
    with pytest.raises(Refusal, match="not a file"):
        raw_text.read("book", "../../../../etc/passwd")
    with pytest.raises(Refusal, match="no heading"):
        raw_text.read("book", "book.md", "three")
