from book_matter import is_book_matter


# front and back matter is left out by its leaf, a contents line by its dot leader; a section named like it stays
def test_front_and_back_matter_is_read_by_the_leaf_alone():
    left_out = [
        "Think Python > Contents", "Think Python > Index", "Redis in Action > brief contents",
        "IN ACTION > preface",
        "IN ACTION > foreword", "IN ACTION > about this book", "IN ACTION > about the cover illustration",
        "Jest > References", "Книга > Предметный указатель", "Книга > Предисловие", "Книга > Список литературы",
        "IN ACTION > PART 1 GETTING STARTED ..................................................1",
    ]
    kept = ["Postgres > Index ranges", "Think Python > Lists", "Book > Introduction", "Redis > References to keys",
            "Handbook > 05: arrays and records", "Анализ данных > 04 Глава 3. Описательный анализ"]
    kept.append(None)
    assert [s for s in left_out if not is_book_matter(s)] == []
    assert [s for s in kept if is_book_matter(s)] == []


# the door every reader of a source shares leaves the matter out, so the index and the generator never see it
def test_a_sources_documents_leave_the_matter_out(tmp_path):
    import config
    from sources.base import Base
    from sources.declaration import SourceFile

    root = tmp_path / "book"
    root.mkdir()
    (root / "a.md").write_text("# Book\n\n## Preface\n\nthanks to all\n\n## Locks\n\ntext about locks\n")
    settings = SourceFile(name="book", language="en", licence="CC BY", folder=str(root), categories=["x"])
    source = Base(root, settings, name="book")
    docs = source.documents(config.settings.corpus.policy("clean_1024"))
    assert [d.section for d in docs] == ["Book > Locks"]
    assert source.left_out_as_matter == [("book/a.md", "Book > Preface")]


# a book's title page is the section headed by the title its file is named after; a markdown project keeps its first
def test_a_books_title_page_is_left_out_and_a_projects_first_section_kept():
    from book_matter import is_title_page

    assert is_title_page("redis-in-action/redis-in-action.pdf", "IN ACTION > Redis in Action")
    assert is_title_page("redis-in-action/redis-in-action.pdf", "IN ACTION")
    assert is_title_page("think-python-2e/think-python-2e.pdf", "Think Python")
    assert not is_title_page("think-python-2e/think-python-2e.pdf", "Think Python > Functions")
    assert not is_title_page("redis-doc/docs/redis.md", "Redis")
    assert not is_title_page("redis-in-action/redis-in-action.pdf", "IN ACTION > Redis")


def _book(index_level: str, body_lines: int = 30) -> str:
    text = ["# Kafka", "## Preface", "words"] + [f"## Chapter {n}\nline" for n in range(body_lines)]
    letters = [f"{index_level} Index", "###### A", "acks, 12", "## D", "durability, 40", "## Symbols", "@, 3"]
    return "\n".join(text + letters + ["## About the Author", "she wrote it"])


# a back index goes whole, its letters at any level, to the next heading that is neither a letter nor matter
def test_a_back_index_is_cut_by_position_whatever_level_its_letters_stand_at():
    import ingest

    for level in ("##", "###"):
        cut = ingest.without_index(_book(level), "kafka.pdf-bb367b09.md")
        assert "durability" not in cut and "@, 3" not in cut and "acks" not in cut
        assert "## Chapter 29" in cut and "she wrote it" not in cut


# only a book has a back index, only in its last quarter, and never as its root
def test_an_index_heading_outside_a_books_back_is_text():
    import ingest

    assert ingest.without_index(_book("##"), "docs/cli.md") == _book("##")
    early = "# Manual\n## Index\nhow to build an index\n" + "\n".join(f"## Part {n}\ntext" for n in range(20))
    assert ingest.without_index(early, "manual.pdf") == early
    root = "# Index\n" + "\n".join(f"## Unit {n}\ntext" for n in range(5))
    assert ingest.without_index(root, "rtl.pdf") == root


# the checker reads the stand's own matter: the back index by position and every `##` section that is matter
def test_matter_lines_are_the_index_span_and_the_matter_sections():
    import ingest

    text = _book("###")
    lines = text.split("\n")
    spans = ingest.matter_lines(text, "kafka.pdf")
    assert (1, 3) in spans
    assert (lines.index("### Index"), len(lines)) in spans


# the back index leaves before the sections exist, so the report names it beside the matter sections
def test_a_cut_back_index_is_named_in_the_matter_left_out(tmp_path):
    import config
    from sources.base import Base
    from sources.declaration import SourceFile

    (tmp_path / "kafka.pdf.md").write_text(_book("##"))
    settings = SourceFile(name="book", language="en", licence="CC BY", folder=str(tmp_path), categories=["x"])
    source = Base(tmp_path, settings, name="book")
    docs = source.documents(config.settings.corpus.policy("clean_1024"))
    assert not any("durability" in d.content for d in docs)
    assert [s for _, s in source.left_out_as_matter if s.startswith("Index (lines")]


def test_a_run_of_leader_lines_is_cut_whatever_its_heading_says():
    from book_matter import leader_spans

    contents = [f"{n}.{k} Section title {'.' * 12} {n * 10 + k}" for n in range(1, 4) for k in range(1, 4)]
    lines = ["# Book", "Some prose before.", "", "## Part II", *contents, "", "## Chapter 1", "Prose again."]
    assert leader_spans(lines, 8) == [(4, 13)], "a contents row the converter made a heading is bridged, not an end"
    interleaved = contents[:5] + ["A preface paragraph a reader needs.", "It goes on."] + contents[5:]
    assert leader_spans(interleaved, 8) == [(0, 5), (7, 11)], "prose between contents rows stays"
    assert leader_spans(lines, 0) == [], "0 is off"
    assert leader_spans(lines, 10) == [], "nine leader lines are not a run of ten"


def test_a_contents_table_is_a_run_and_one_leader_line_in_prose_is_not():
    from book_matter import leader_spans

    table = ["| Глава | Стр. |", "|---|---|"] + [f"| Глава {n} {'. ' * 6}{n * 7} |" for n in range(1, 10)]
    prose = ["See the table ........ 12 for details.", "A paragraph.", "Another one.", "And more.", "Still prose."]
    assert leader_spans(table, 8) == [(2, 11)]
    assert leader_spans(prose * 3, 8) == [], "leader lines far apart in prose never make a run"


def test_the_cut_drops_contents_runs_only_when_the_variant_asks():
    import ingest

    contents = "\n".join(f"Chapter {n} {'.' * 10} {n}" for n in range(1, 10))
    text = f"# Book\n\n{contents}\n\n## One\n\nThe text of chapter one."
    assert "Chapter 3" in ingest.without_index(text, "book.md")
    kept = ingest.without_index(text, "book.md", contents_runs_from=8)
    assert "Chapter 3" not in kept and "The text of chapter one." in kept
