# a PDF with no bookmarks takes its printed contents as the outline, the printed pages moved onto the file's
from pathlib import Path

from use_cases import route


def test_printed_contents_become_an_outline_shifted_by_the_front_matter(monkeypatch):
    contents = "\n".join([
        "Часть I. Основы .......................... 15",
        "Глава 1. Большие данные (Big Data)........ 15",
        "Зрелость данных .......................... 18",
        "Глава 2. Типы архитектур данных .......... 25",
        "Итоги .................................... 30",
        "Глава 3. Дизайн-сессия ................... 31",
    ])
    pages = ["обложка", contents] + ["текст"] * 14 + ["Глава 1. Большие данные (Big Data) текст"] + ["текст"] * 9 \
        + ["Глава 2. Типы архитектур данных"] + ["текст"] * 5 + ["Глава 3. Дизайн-сессия"] + ["текст"] * 5
    monkeypatch.setattr(route, "layer_texts", lambda path: pages)
    outline = route.contents_outline(Path("book.pdf"))
    assert outline[:3] == [(1, "Часть I. Основы", 17), (2, "Глава 1. Большие данные (Big Data)", 17),
                           (3, "Зрелость данных", 20)]
    assert (2, "Глава 2. Типы архитектур данных", 27) in outline


def test_a_book_with_too_few_contents_lines_gets_no_outline(monkeypatch):
    monkeypatch.setattr(route, "layer_texts", lambda path: ["Глава 1. Одно .......... 5", "текст"])
    assert route.contents_outline(Path("book.pdf")) == []
