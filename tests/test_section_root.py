from pathlib import Path

import pytest
from sources.cheatsheets import CheatsheetsSource
from sources.interview import InterviewSource
from sources.notes import NotesSource
from sources.redis_docs import RedisDocsSource
from sources.system_design_primer import SystemDesignPrimerSource

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


# the declared root is a rule of the hygienic cut, so the tests ask for it under the
# policy that turns it on, built through the model production loads: `header_prefix` is
# derived and cannot be written by hand
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
    source = InterviewSource(tmp_path, name="ruby-interview-questions")
    file = write(tmp_path, "README.md", INTERVIEW)
    assert root_of(source, file) == "100 Core Ruby Interview Questions in 2026"


def test_redis_docs_root_comes_from_frontmatter(tmp_path):
    source = RedisDocsSource(tmp_path)
    file = write(tmp_path, "docs/data-types/streams.md", REDIS_DOC)
    assert root_of(source, file) == "Redis Streams"


def test_redis_command_root_is_the_file_name_because_there_is_no_heading(tmp_path):
    source = RedisDocsSource(tmp_path)
    file = write(tmp_path, "commands/acl-cat.md", REDIS_COMMAND)
    assert root_of(source, file) == "ACL CAT"


def test_cheatsheet_root_is_its_frontmatter_title(tmp_path):
    source = CheatsheetsSource(tmp_path)
    file = write(tmp_path, "react.md", CHEATSHEET)
    assert root_of(source, file) == "React"


def test_primer_root_skips_the_translation_banner(tmp_path):
    source = SystemDesignPrimerSource(tmp_path)
    file = write(tmp_path, "README.md", PRIMER)
    assert root_of(source, file) == "The System Design Primer"


def test_notes_root_is_the_markdown_heading(tmp_path):
    source = NotesSource(tmp_path)
    file = write(tmp_path, "index.md", NOTE)
    assert root_of(source, file) == "Notes, база знаний"


def test_a_byte_order_mark_does_not_hide_the_frontmatter(tmp_path):
    source = RedisDocsSource(tmp_path)
    file = write(tmp_path, "docs/get-started/_index.md", "\ufeff" + REDIS_DOC)
    assert root_of(source, file) == "Redis Streams"


@pytest.mark.parametrize("text", ["", "   \n  ", "no heading at all\njust body"])
def test_a_file_without_a_heading_has_no_root(tmp_path, text):
    source = NotesSource(tmp_path)
    file = write(tmp_path, "empty.md", text)
    assert root_of(source, file) is None


def test_a_numeric_frontmatter_title_is_still_text(tmp_path):
    source = CheatsheetsSource(tmp_path)
    file = write(tmp_path, "101.md", "---\ntitle: 101\ncategory: JavaScript libraries\n---\n\n## Usage\n")
    assert root_of(source, file) == "101"


def test_the_question_builder_and_the_cut_agree_on_what_a_heading_is():
    # the cut asks the parser and skips fenced lines; this asked a regexp, so `## 2.`
    # inside a code block became a question no chunk could carry as its section
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
    # the cut reads such a file fence-blind and says so; a builder that trusts the scan
    # drops every question after the stray opener, and four files of 1010 open one
    from evals import build_questions

    text = (
        "## 1. First question\n\nanswer one\n\n"
        "```bash\nnever closed\n\n"
        "## 2. Second question\n\nanswer two\n\n"
        "## 3. Third question\n\nanswer three\n"
    )
    got = [q for q, _ in build_questions._qa_pairs(text, "some/README.md")]
    assert got == ["First question", "Second question", "Third question"]
