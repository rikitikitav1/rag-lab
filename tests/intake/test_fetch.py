import zipfile

import pytest
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
    from use_cases import intake_fetch

    root = tmp_path / "book"
    root.mkdir()
    _epub(root / "book.epub")
    (root / "notes.md").write_text("## Notes")

    names, skipped = intake_fetch.named_files(root, sorted(root.iterdir()), tmp_path / "inbox")
    assert list(names.values()) == ["notes.md"] and "epub_chapters" in skipped["book.epub"]

    names, _ = intake_fetch.named_files(root, sorted(root.iterdir()), tmp_path / "inbox", epub=True)

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


# a page the site builds itself is left out with its pattern named, as the gold leaves it out
def test_a_page_the_site_builds_is_left_out_by_its_pattern(tmp_path):
    from use_cases import intake_fetch

    root = tmp_path / "site"
    root.mkdir()
    (root / "api.html").write_text("<p>auto-generated from the Go source</p>")
    (root / "guide.html").write_text("<p>a guide</p>")

    files = sorted(root.iterdir())
    names, skipped = intake_fetch.named_files(root, files, tmp_path / "inbox", generated=["^.*auto-generated"])

    assert list(names.values()) == ["guide.html"]
    assert skipped == {"api.html": "the site builds this page: ^.*auto-generated"}


# a declared path pattern leaves a whole folder out with the pattern named, where every page is called `index`
def test_a_declared_path_leaves_its_folder_out(tmp_path):
    from use_cases import intake_fetch

    root = tmp_path / "site"
    for rel in ("releasenotes/26.3/index.html", "guide/index.html"):
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_text("<p>text</p>")

    files = sorted(root.rglob("*.html"))
    names, skipped = intake_fetch.named_files(root, files, tmp_path / "inbox", skip_paths=["releasenotes/*"])

    assert list(names.values()) == ["guide/index.html"]
    assert skipped == {"releasenotes/26.3/index.html": "the declaration skips releasenotes/*"}


# highlighting is flat in any HTML, a page from `urls` too, and the fetched file is left as it came
def test_any_html_reaches_the_tool_with_its_highlighting_flat(tmp_path):
    from use_cases import intake_fetch

    root = tmp_path / "book"
    root.mkdir()
    page = root / "ch1.html"
    page.write_text('<pre><span class="k">def</span> <span class="n">f</span>():</pre>')
    (root / "plain.html").write_text("<p>no code</p>")

    names, _ = intake_fetch.named_files(root, sorted(root.iterdir()), tmp_path / "inbox")

    flat = next(f for f, rel in names.items() if rel == "ch1.html")
    assert flat.read_text() == "<pre><code>def f():</code></pre>" and "<span" in page.read_text()
    assert root / "plain.html" in names


# a repeated intake of a git source reads the upstream's new tip; without the update the kept clone stays as it came
def test_a_kept_clone_takes_the_upstreams_new_tip_only_when_asked(tmp_path):
    import subprocess

    from use_cases import fetch

    upstream = tmp_path / "up"
    upstream.mkdir()

    def git(*args):
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", *args], cwd=upstream, check=True)

    git("init", "-q", "-b", "main")
    (upstream / "a.md").write_text("one")
    git("add", ".")
    git("commit", "-qm", "one")
    folder = tmp_path / "clone"
    first = fetch.clone(f"file://{upstream}", folder)
    (upstream / "a.md").write_text("two")
    git("commit", "-qam", "two")

    kept = fetch.clone(f"file://{upstream}", folder)
    assert (folder / "a.md").read_text() == "one" and kept["revision"] == first["revision"]
    moved = fetch.clone(f"file://{upstream}", folder, update=True)
    assert (folder / "a.md").read_text() == "two" and moved["revision"] != first["revision"]

    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=other, check=True)
    (other / "b.md").write_text("elsewhere")
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "add", "."], cwd=other, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "b"], cwd=other, check=True)
    fetch.clone(f"file://{other}", folder, update=True)
    assert (folder / "b.md").read_text() == "elsewhere", "a moved repo is cloned anew"
    assert not (folder / "a.md").exists()


class _Got:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield b"<html>page</html>"


# a stalled request is asked again, and only the last stall of the tries fails the page
def test_a_stalled_page_is_asked_again(monkeypatch, tmp_path):
    import requests

    calls = []

    def get(url, stream, timeout):
        calls.append(url)
        if len(calls) < fetch.TRIES:
            raise requests.ReadTimeout("stalled")
        return _Got()

    monkeypatch.setattr(fetch.requests, "get", get)
    assert fetch.download("https://a/p.html", tmp_path / "p.html") is True
    assert (tmp_path / "p.html").read_bytes() == b"<html>page</html>" and len(calls) == fetch.TRIES

    monkeypatch.setattr(fetch.requests, "get", lambda url, stream, timeout: (_ for _ in ()).throw(
        requests.ConnectTimeout("down")))
    with pytest.raises(requests.ConnectTimeout):
        fetch.download("https://a/q.html", tmp_path / "q.html")
    assert not (tmp_path / "q.html").exists()


# a folder of documents reads them and names the screenshots beside them as left out; a scan folder reads its images
def test_a_docs_folder_leaves_its_pictures_and_a_scan_folder_keeps_them(tmp_path):
    from use_cases.intake_fetch import named_files, stand_folder

    docs = tmp_path / "docs"
    (docs / "media").mkdir(parents=True)
    (docs / "page.md").write_text("# Page\n")
    (docs / "media" / "shot.png").write_bytes(b"\x89PNG")
    scans = tmp_path / "scans"
    scans.mkdir()
    (scans / "p1.png").write_bytes(b"\x89PNG")

    root, files = stand_folder("docs", tmp_path)
    names, skipped = named_files(root, files, tmp_path / "inbox")
    assert list(names.values()) == ["page.md"] and list(skipped) == ["media/shot.png"]
    root, files = stand_folder("scans", tmp_path)
    assert list(named_files(root, files, tmp_path / "inbox")[0].values()) == ["p1.png"]
