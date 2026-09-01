from pathlib import Path

import config
import pytest
from sources import base
from sources.base import Doc
from sources.cheatsheets import CheatsheetsSource
from sources.interview import InterviewSource
from sources.notes import NotesSource


# built through the model production loads, so an impossible fixture cannot pass here
def _policy(**kw) -> dict:
    from config import PolicyCfg

    return PolicyCfg(**kw).model_dump()


DROPPING = _policy(chunker="rooted", max_chunk_size=1024)
KEEPING = _policy(chunker="legacy", max_chunk_size=1024)


def doc(content, i=0, body=None):
    return Doc(
        content=content, source="s/f.md", category="c", language="eng",
        chunk_index=i, title="t", links=[], tags=[], body=body,
    )


def write(base: Path, rel: str, text: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_versioned_cheatsheet_goes_and_the_plain_one_stays(tmp_path):
    for name in ("react.md", "react@0.14.md", "vainglory.md", "figlet.md", "101.md"):
        write(tmp_path, name, "---\ntitle: x\n---\n\n## S\n\nbody\n")
    kept = {f.name for f in CheatsheetsSource(tmp_path).discover(DROPPING)}
    assert kept == {"react.md", "101.md"}


def test_the_same_cheatsheets_stay_when_the_policy_keeps_them(tmp_path):
    for name in ("react.md", "react@0.14.md", "vainglory.md"):
        write(tmp_path, name, "---\ntitle: x\n---\n\n## S\n\nbody\n")
    kept = {f.name for f in CheatsheetsSource(tmp_path).discover(KEEPING)}
    assert kept == {"react.md", "react@0.14.md", "vainglory.md"}


def test_the_interview_badge_goes_and_the_answers_stay(tmp_path):
    source = InterviewSource(tmp_path, name="ruby-interview-questions")
    docs = [doc("a badge. You can also find all 100 answers here", 0), doc("a real answer", 1)]
    kept = source.postprocess(docs, DROPPING)
    assert [d.content for d in kept] == ["a real answer"]
    assert [d.chunk_index for d in kept] == [0]


def test_the_notes_hub_is_skipped_on_ingest_not_on_search(tmp_path):
    write(tmp_path, "index.md", "# Hub\n")
    write(tmp_path, "real.md", "# Real\n")
    source = NotesSource(tmp_path)
    assert {f.name for f in source.discover(DROPPING)} == {"real.md"}
    # and a variant that declares the old cut still sees the file it always saw
    assert {f.name for f in source.discover(KEEPING)} == {"index.md", "real.md"}


def test_a_symlink_out_of_the_corpus_is_not_discovered(tmp_path):
    outside = tmp_path.parent / "secret.md"
    outside.write_text("# not ours\n", encoding="utf-8")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "real.md").write_text("# Real\n", encoding="utf-8")
    (root / "creds.md").symlink_to(outside)
    assert {f.name for f in NotesSource(root).discover(DROPPING)} == {"real.md"}


def test_a_symlink_inside_the_corpus_is_fine(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "real.md").write_text("# Real\n", encoding="utf-8")
    (root / "link.md").symlink_to(root / "sub" / "real.md")
    assert {f.name for f in NotesSource(root).discover(DROPPING)} == {"real.md", "link.md"}


def test_the_index_shim_never_fires_on_a_variant_that_was_cut_with_the_rule():
    from use_cases import chat

    for variant in config.settings.corpus.variants:
        policy = config.settings.corpus.policy(variant)
        hidden = chat._hidden_by_cut("notes/index.md", variant)
        assert hidden != base.hygienic(policy), variant
    # every declared variant makes the two exact complements, so the assertion needs more
    assert base.hygienic({"chunker": "structured"}) is True
    assert base.hygienic({"chunker": "legacy"}) is False


def test_a_missing_index_is_queued_rather_than_built_while_the_stack_waits(monkeypatch):
    # an hnsw build takes tens of minutes and the whole stack waits on bootstrap
    import bootstrap

    queued = []
    monkeypatch.setattr(
        "db.corpus_variants", lambda: [{"variant": "a"}, {"variant": "b"}]
    )
    monkeypatch.setattr("use_cases.index.has_vector_index", lambda v: v == "b")
    monkeypatch.setattr(
        "use_cases.index.ensure_vector_index",
        lambda v: pytest.fail("bootstrap must not build an index inline"),
    )
    monkeypatch.setattr(bootstrap.job_queue, "pending_of_type", lambda t, **kw: False)
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda t, o: queued.append((t, o)))
    bootstrap._ensure_vector_indexes()
    assert queued == [("build_vector_index", {"variant": "a"})]
