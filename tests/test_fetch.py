import zipfile

from use_cases import fetch


# an archive of an added source is read as the files inside it, not sent whole to a converter
def test_an_archive_is_its_files(tmp_path):
    archive = tmp_path / "book.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("book/a.md", "# A")
        z.writestr("book/b.md", "# B")

    assert [p.name for p in fetch.unzip(archive)] == ["a.md", "b.md"]


def test_a_path_stays_inside_its_folder(tmp_path):
    assert fetch.inside(tmp_path, "docs") == (tmp_path / "docs").resolve()
    assert fetch.inside(tmp_path, None) == tmp_path.resolve()
    assert fetch.inside(tmp_path, "../x") is None
    assert fetch.inside(tmp_path, "/etc") is None


def _epub(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        z.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><metadata><title>Book (for A Buyer)</title></metadata>'
            '<manifest><item id="a" href="intro.html" media-type="application/xhtml+xml"/>'
            '<item id="b" href="ch01.html" media-type="application/xhtml+xml"/>'
            '<item id="ix" href="ix01.html" media-type="application/xhtml+xml"/>'
            '<item id="css" href="style.css" media-type="text/css"/></manifest>'
            '<spine><itemref idref="b"/><itemref idref="a"/><itemref idref="ix"/></spine></package>',
        )
        z.writestr("OEBPS/intro.html", '<body data-type="book"><p>intro</p></body>')
        z.writestr("OEBPS/ix01.html", '<body><section data-type="index"><p>api, 12</p></section></body>')
        z.writestr("OEBPS/ch01.html", "<p>one</p>")
        z.writestr("OEBPS/style.css", "body {}")


# the spine sets the order, and nothing but the chapters leaves the archive
def test_an_epub_is_its_chapters_in_reading_order(tmp_path):
    _epub(tmp_path / "book.epub")

    chapters, skipped = fetch.epub_chapters(tmp_path / "book.epub", tmp_path / "out")

    assert [p.name for p in chapters] == ["001_ch01.html", "002_intro.html", "003_ix01.html"] and not skipped
    assert chapters[0].read_text() == "<p>one</p>"
    assert "Buyer" not in "".join(p.read_text() for p in (tmp_path / "out").iterdir())


def test_an_epub_in_a_folder_is_named_by_its_chapters(tmp_path):
    from use_cases import source_intake

    root = tmp_path / "book"
    root.mkdir()
    _epub(root / "book.epub")
    (root / "notes.md").write_text("## Notes")

    names, skipped = source_intake.named_files(root, sorted(root.iterdir()), tmp_path / "inbox")
    assert list(names.values()) == ["notes.md"] and "epub_chapters" in skipped["book.epub"]

    names, _ = source_intake.named_files(root, sorted(root.iterdir()), tmp_path / "inbox", epub=True)

    assert sorted(names.values()) == [
        "book.epub/001_ch01.html",
        "book.epub/002_intro.html",
        "book.epub/003_ix01.html",
        "notes.md",
    ]


def test_an_epub_page_of_a_skipped_type_is_left_out_and_named(tmp_path):
    _epub(tmp_path / "book.epub")

    chapters, skipped = fetch.epub_chapters(tmp_path / "book.epub", tmp_path / "out", frozenset({"index"}))

    assert [p.name for p in chapters] == ["001_ch01.html", "002_intro.html"]
    assert skipped == {"003_ix01.html": "epub page type index"}


# a type on a figure or inside a typed body counts, single quotes too; a chapter's typed notes further down do not
def test_a_page_type_is_read_from_any_element_near_the_top():
    assert "cover" in fetch.page_types('<body><figure data-type="cover"><img/></figure></body>')
    assert "titlepage" in fetch.page_types("<body epub:type='frontmatter'><section epub:type='titlepage'>")
    assert "index" not in fetch.page_types('<body data-type="chapter">' + "x" * 5000 + '<div data-type="index">')
