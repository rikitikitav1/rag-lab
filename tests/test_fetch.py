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
