from pathlib import Path

import pytest
from sources.base import Base
from sources.declaration import SourceFile

PRIMER = """*[English](README.md) ∙ [日本語](README-ja.md) ∙ [简体中文](README-zh-Hans.md)

**Help [translate](TRANSLATIONS.md) this guide!**

# The System Design Primer

## Motivation
"""

REDIS_DOC = """---
title: "Redis Streams"
weight: 60
---

Introduction to Redis streams.

## Consumer groups
"""

REDIS_COMMAND = """Get the value of `key`.
If the key does not exist the special value `nil` is returned.
"""

CHEATSHEET = """---
title: React
category: React
---

## Components
"""

INTERVIEW = """# 100 Core Ruby Interview Questions in 2026

## 1. What is _Ruby_?
"""

NOTE = """# Notes, база знаний

Хаб, отсюда расходятся темы.
"""

# a source read the way its file declares it, with no reader class of its own
def _declared(file, root, **kw):
    from sources import files
    from sources.base import Base

    return Base(root, settings=files.source_files()[file], **kw)



# the declared root is a rule of the hygienic cut, so these ask under that policy

# a plain markdown folder source, as a folder source file declares one
def _plain(root):
    rules = {"index": "a hub of links answers nothing"}
    return Base(root, SourceFile(name="book", language="ru", licence="x", folder=str(root), skip_when_hygienic=rules))

def _policy(**kw) -> dict:
    from config import PolicyCfg

    return PolicyCfg(**kw).model_dump()


HYGIENIC = _policy(chunker="rooted", max_chunk_size=1024)


def root_of(source, file):
    rel = str(file.relative_to(source.root))
    return source.section_root_for(file, source.read(file, rel, HYGIENIC))


def write(base: Path, rel: str, text: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_interview_root_is_the_h1_of_the_readme(tmp_path):
    source = _declared("interview", tmp_path, name="ruby-interview-questions")
    file = write(tmp_path, "README.md", INTERVIEW)
    assert root_of(source, file) == "100 Core Ruby Interview Questions in 2026"


def test_redis_docs_root_comes_from_frontmatter(tmp_path):
    source = _declared("redis-doc", tmp_path)
    file = write(tmp_path, "docs/data-types/streams.md", REDIS_DOC)
    assert root_of(source, file) == "Redis Streams"


def test_redis_command_root_is_the_file_name_because_there_is_no_heading(tmp_path):
    source = _declared("redis-doc", tmp_path)
    file = write(tmp_path, "commands/acl-cat.md", REDIS_COMMAND)
    assert root_of(source, file) == "ACL CAT"


def test_cheatsheet_root_is_its_frontmatter_title(tmp_path):
    source = _declared("cheatsheets", tmp_path)
    file = write(tmp_path, "react.md", CHEATSHEET)
    assert root_of(source, file) == "React"


def test_primer_root_skips_the_translation_banner(tmp_path):
    source = _declared("system-design-primer", tmp_path)
    file = write(tmp_path, "README.md", PRIMER)
    assert root_of(source, file) == "The System Design Primer"


def test_a_plain_file_s_root_is_its_markdown_heading(tmp_path):
    source = _plain(tmp_path)
    file = write(tmp_path, "index.md", NOTE)
    assert root_of(source, file) == "Notes, база знаний"


def test_a_byte_order_mark_does_not_hide_the_frontmatter(tmp_path):
    source = _declared("redis-doc", tmp_path)
    file = write(tmp_path, "docs/get-started/_index.md", "\ufeff" + REDIS_DOC)
    assert root_of(source, file) == "Redis Streams"


@pytest.mark.parametrize("text", ["", "   \n  ", "no heading at all\njust body"])
def test_a_file_without_a_heading_has_no_root(tmp_path, text):
    source = _plain(tmp_path)
    file = write(tmp_path, "empty.md", text)
    assert root_of(source, file) is None


def test_a_numeric_frontmatter_title_is_still_text(tmp_path):
    source = _declared("cheatsheets", tmp_path)
    file = write(tmp_path, "101.md", "---\ntitle: 101\ncategory: JavaScript libraries\n---\n\n## Usage\n")
    assert root_of(source, file) == "101"


def test_the_question_builder_and_the_cut_agree_on_what_a_heading_is():
    # the cut asks the parser and skips fenced lines; a regexp made `## 2.` a question
    from evals import build_questions

    text = (
        "## 1. Real question\n\nanswer one\n\n"
        "```bash\n## 2. Not a question\n```\n\n"
        "## 3. Another real\n\nanswer three\n"
    )
    assert [q for q, _ in build_questions._qa_pairs(text)] == [
        "Real question",
        "Another real",
    ]


def test_the_question_builder_falls_back_when_a_fence_never_closes():
    # the cut reads such a file fence-blind and says so; a builder that trusts it drops questions
    from evals import build_questions

    text = (
        "## 1. First question\n\nanswer one\n\n"
        "```bash\nnever closed\n\n"
        "## 2. Second question\n\nanswer two\n\n"
        "## 3. Third question\n\nanswer three\n"
    )
    got = [q for q, _ in build_questions._qa_pairs(text, "some/README.md")]
    assert got == ["First question", "Second question", "Third question"]


def test_a_chunk_takes_the_language_of_its_own_text(tmp_path):
    source = _plain(tmp_path)
    text = (
        "# Книга\n\n## Глава\n\nЗдесь идёт русский текст о снимках данных.\n\n"
        "## Listing\n\nSELECT id FROM accounts WHERE amount > 100;\n"
    )
    file = write(tmp_path, "book.md", text)
    languages = {doc.section: doc.language for doc in source.to_documents(file, HYGIENIC)}
    assert "ru" in languages.values() and "en" in languages.values(), languages


# a converter reads cover text as a book's first heading; a declared root by file glob stands in for it
def test_declared_root_by_path_replaces_the_cover_heading(tmp_path):
    roots = {"rtl.pdf*": "Run-Time Library", "*": "The Book"}
    declared = SourceFile(name="book", language="en", licence="x", folder=str(tmp_path), section_root_by_path=roots)
    source = Base(tmp_path, declared)
    cover = "## MdLAHeH 97 1588380\n\ntitle page\n\n## Chapter\n\ntext\n"
    assert root_of(source, write(tmp_path, "rtl.pdf.md", cover)) == "Run-Time Library"
    assert root_of(source, write(tmp_path, "other.md", cover)) == "The Book"
    assert root_of(_plain(tmp_path), write(tmp_path, "plain.md", cover)) == "MdLAHeH 97 1588380"
