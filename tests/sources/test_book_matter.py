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
