from pathlib import Path

import pytest
from sources.base import Base, Doc
from sources.declaration import SourceFile

# a class built here reads its rules from its file in `sources/`: these tests pin the corpus's current lists

# a source read the way its file declares it, with no reader class of its own
def _declared(file, root, **kw):
    from sources import files
    from sources.base import Base

    return Base(root, settings=files.source_files()[file], **kw)



# built through the model production loads, so an impossible fixture cannot pass here

# a plain markdown folder source, as a folder source file declares one
def _plain(root):
    rules = {"index": "a hub of links answers nothing"}
    return Base(root, SourceFile(name="book", language="ru", licence="x", folder=str(root), skip_when_hygienic=rules))

def _policy(**kw) -> dict:
    from config import PolicyCfg

    return PolicyCfg(**kw).model_dump()


DROPPING = _policy(chunker="rooted", max_chunk_size=1024)


def doc(content, i=0, body=None):
    return Doc(
        content=content,
        source="s/f.md",
        category="c",
        language="en",
        chunk_index=i,
        title="t",
        links=[],
        tags=[],
        body=body,
    )


def write(base: Path, rel: str, text: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_versioned_cheatsheet_goes_and_the_plain_one_stays(tmp_path):
    for name in ("react.md", "react@0.14.md", "vainglory.md", "figlet.md", "101.md"):
        write(tmp_path, name, "---\ntitle: x\n---\n\n## S\n\nbody\n")
    kept = {f.name for f in _declared("cheatsheets", tmp_path).discover(DROPPING)}
    assert kept == {"react.md", "101.md"}


def test_the_interview_badge_goes_and_the_answers_stay(tmp_path):
    source = _declared("interview", tmp_path, name="ruby-interview-questions")
    docs = [doc("a badge. You can also find all 100 answers here", 0), doc("a real answer", 1)]
    kept = source.postprocess(docs, DROPPING)
    assert [d.content for d in kept] == ["a real answer"]
    assert [d.chunk_index for d in kept] == [0]


def test_a_hub_of_links_is_skipped_on_ingest_not_on_search(tmp_path):
    write(tmp_path, "index.md", "# Hub\n")
    write(tmp_path, "real.md", "# Real\n")
    source = _plain(tmp_path)
    assert {f.name for f in source.discover(DROPPING)} == {"real.md"}


def test_a_symlink_out_of_the_corpus_is_not_discovered(tmp_path):
    outside = tmp_path.parent / "secret.md"
    outside.write_text("# not ours\n", encoding="utf-8")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "real.md").write_text("# Real\n", encoding="utf-8")
    (root / "creds.md").symlink_to(outside)
    assert {f.name for f in _plain(root).discover(DROPPING)} == {"real.md"}


def test_a_symlink_inside_the_corpus_is_fine(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "real.md").write_text("# Real\n", encoding="utf-8")
    (root / "link.md").symlink_to(root / "sub" / "real.md")
    assert {f.name for f in _plain(root).discover(DROPPING)} == {"real.md", "link.md"}


def test_a_missing_index_is_queued_rather_than_built_while_the_stack_waits(monkeypatch):
    # an hnsw build takes tens of minutes and the whole stack waits on bootstrap
    import bootstrap

    queued = []
    monkeypatch.setattr("db.corpus_variants", lambda: [{"variant": "a"}, {"variant": "b"}])
    monkeypatch.setattr("use_cases.index.has_vector_index", lambda v: v == "b")
    monkeypatch.setattr(
        "use_cases.index.ensure_vector_index",
        lambda v: pytest.fail("bootstrap must not build an index inline"),
    )
    monkeypatch.setattr(bootstrap.job_queue, "pending_of_type", lambda t, **kw: False)
    monkeypatch.setattr(bootstrap.job_queue, "enqueue", lambda t, o: queued.append((t, o)))
    bootstrap._ensure_vector_indexes()
    assert queued == [("build_vector_index", {"variant": "a"})]


# a path pattern leaves a folder out at the index too, where the stems inside it say nothing
def test_a_declared_path_is_not_discovered(tmp_path):
    for rel in ("blog/2025/post.md", "learn/state.md"):
        (tmp_path / rel).parent.mkdir(parents=True)
        (tmp_path / rel).write_text("# x\n\ntext\n")
    declared = SourceFile(name="book", language="en", licence="x", folder=str(tmp_path), skip_paths=["blog/*"])
    source = Base(tmp_path, declared)

    assert [source.rel_of(f) for f in source.discover()] == ["learn/state.md"]


# a page's metadata goes, its title and description stay as words under the page's own heading
def test_frontmatter_is_read_away_and_its_description_kept():
    from use_cases.markdown_cleanup import without_frontmatter

    page = (
        "---\ntitle: \"USE\"\ndescription: Changes the database context.\nms.date: 07/15/2025\n"
        "f1_keywords:\n  - USE\n---\n# USE\n\nBody.\n"
    )
    assert without_frontmatter(page) == "# USE\n\nChanges the database context.\n\nBody.\n"
    toml = "+++\ntitle = \"Pods\"\nweight = 3\n+++\nA pod is a group.\n"
    assert without_frontmatter(toml) == "# Pods\n\nA pod is a group.\n"
    # a page whose body has only sub-headings keeps its title as the root, its lead under the title
    hashes = "---\ntitle: Redis hashes\ndescription: Intro\n---\nMaps.\n\n## Basic commands\n\nHSET.\n"
    assert without_frontmatter(hashes) == "# Redis hashes\n\nIntro\n\nMaps.\n\n## Basic commands\n\nHSET.\n"
    # a body that repeats the title at another level is not titled twice
    echo = "---\ntitle: Pods\n---\n## Pods\n\nA pod.\n"
    assert without_frontmatter(echo) == "## Pods\n\nA pod.\n"
    # an own top heading after an import keeps the page as it is; a `#` comment in code is no heading
    mdx = "---\ntitle: Agents\n---\nimport X from 'y'\n\n# Agents\n\nText.\n"
    assert without_frontmatter(mdx) == "import X from 'y'\n\n# Agents\n\nText.\n"
    shell = "---\ntitle: Setup\n---\n```bash\n# install\n```\n\n## Steps\n"
    assert without_frontmatter(shell).startswith("# Setup\n\n```bash")


# a rule between two lines of prose, or a fence of text that is not a mapping, is the page's own and stays
def test_a_thematic_break_is_not_frontmatter():
    from use_cases.markdown_cleanup import without_frontmatter

    for page in ("---\nJust a line.\n---\nMore.\n", "# T\n\n---\nkey: value\n---\n", "---\n\n# T\n"):
        assert without_frontmatter(page) == page


# a table row's padding goes, its cells and a code span inside them stay word for word; a fenced table is code
def test_table_padding_is_squeezed_outside_code():
    from use_cases.markdown_cleanup import without_table_padding

    page = "| a        | `x  y`   |\n|----------|----------|\n```\n| keep     | this |\n```\nText  with  spaces.\n"
    assert without_table_padding(page) == (
        "| a | `x  y` |\n|----------|----------|\n```\n| keep     | this |\n```\nText  with  spaces.\n"
    )
    tilde = "~~~\n| keep     | this |\n```\n| still    | code |\n~~~\n| a      | b |\n"
    assert without_table_padding(tilde) == tilde.replace("| a      | b |", "| a | b |")
