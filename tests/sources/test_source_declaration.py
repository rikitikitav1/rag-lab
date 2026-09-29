import pytest
from pydantic import ValidationError
from sources.declaration import Declaration

BASE = {"name": "progit-ru", "language": "ru", "licence": "CC BY-NC-SA 3.0"}


def test_each_origin_is_a_valid_declaration():
    Declaration(**BASE, urls=["https://example.org/book.pdf"])
    Declaration(**BASE, folder="datasets/inbox/progit-ru")
    Declaration(**BASE, git={"repo": "https://github.com/progit/progit2-ru", "path": "book"})
    Declaration(**BASE, pages=["https://kubernetes.io/docs/concepts/"], site={"main": "div.td-content"})


def test_a_source_comes_from_exactly_one_origin():
    with pytest.raises(ValidationError, match="exactly one"):
        Declaration(**BASE)
    with pytest.raises(ValidationError, match="exactly one"):
        Declaration(**BASE, urls=["https://example.org/a.pdf"], folder="datasets/inbox/a")


# the route reads the engine from the file; a declaration that names one is refused, not obeyed
def test_a_declaration_does_not_name_an_engine():
    with pytest.raises(ValidationError):
        Declaration(**BASE, urls=["https://example.org/a.pdf"], engine="mineru")


def test_site_settings_belong_to_pages():
    with pytest.raises(ValidationError, match="pages of a site"):
        Declaration(**BASE, urls=["https://example.org/a.pdf"], site={"main": "article"})


def test_a_git_repo_is_a_url_not_an_option():
    with pytest.raises(ValidationError):
        Declaration(**BASE, git={"repo": "--upload-pack=touch /tmp/x"})


# a source's knobs name settings files of their own tool; a misspelt one is refused at the door
def test_intake_knobs_name_settings_files_of_their_tool():
    from sources.declaration import IntakeOverride

    assert IntakeOverride(settings={"docling": "docling/pypdfium2_cells"}, reread_settings="docling/pypdfium2")
    wrong = [{"settings": {"docling": "mineru/ocr"}}, {"settings": {"docling": "pypdfium2"}}]
    for bad in [*wrong, {"reread_settings": "mineru/ocr"}]:
        with pytest.raises(ValidationError, match="not a settings file"):
            IntakeOverride(**bad)


# a knob lives in the stand's rule, in a source's override and in the door's description; the four lists agree
def test_every_source_knob_is_said_in_the_docs_that_list_them():
    import re
    from pathlib import Path

    from config import SOURCE_KNOBS

    docs = Path(__file__).resolve().parents[2] / "docs"
    api = (docs / "api.md").read_text()
    door = api[api.index("`PUT /v1/source/{id}/intake`") :].split("\n", 1)[0]
    assert set(SOURCE_KNOBS) <= set(re.findall(r"`(\w+)`", door))
    assert set(SOURCE_KNOBS) <= set(re.findall(r"`(\w+)`", (docs / "intake.md").read_text()))


def test_a_source_knob_keeps_the_stand_s_bounds():
    from pydantic import ValidationError
    from sources.declaration import IntakeOverride

    assert IntakeOverride(reread_cells_slack=0.05).reread_cells_slack == 0.05
    for bad in ({"reread_cells_slack": 2}, {"mono_faces": []}, {"seam_margin": 0.5}, {"no_such_knob": True}):
        try:
            IntakeOverride(**bad)
        except ValidationError:
            continue
        raise AssertionError(bad)


# one model for the door and the seed: the rules and the owner's questions ride on it, the language and licence may wait
def test_a_declaration_carries_its_rules_and_questions_and_may_leave_the_language_to_the_stand():
    from pydantic import ValidationError
    from sources.declaration import Declaration
    from use_cases import source_intake

    declaration = Declaration(
        name="book",
        folder="inbox/book",
        reader="converted",
        categories=["postgresql"],
        questions=[{"text": "What does VACUUM reclaim?", "file": "book/ch9.pdf"}],
    )
    assert declaration.language is None and declaration.licence is None
    row = source_intake.declared_row(declaration)
    assert row.declaration["reader"] == "converted" and row.declaration["questions"][0]["file"] == "book/ch9.pdf"
    assert row.kind == "local" and row.origin is None
    family = Declaration(name="repos", git_family={"base_url": "https://github.com/x", "repos": ["a"]})
    assert source_intake.declared_row(family).kind == "git"
    for bad in ({"text": "why"}, {"text": "What?", "gold": "x"}):
        try:
            Declaration(name="book", folder="inbox/book", questions=[bad])
        except ValidationError:
            continue
        raise AssertionError(bad)


# a door's row and a seeded row of one declaration carry the same derived columns
def test_a_door_row_has_the_path_and_the_family_url_a_seeded_row_has():
    from sources import files
    from use_cases import source_intake

    family = {"base_url": "https://github.com/x", "repos": ["go-questions"]}
    book, bank = Declaration(name="book", folder="inbox/book"), Declaration(name="go-questions", git_family=family)
    for declaration in (book, bank):
        row = source_intake.declared_row(declaration)
        derived = files.row_of(declaration, declaration.name)
        assert (row.kind, row.path, row.git_url) == (derived["kind"], derived["path"], derived["git_url"])


# a site's pages come from its sitemap or from a written list, never both, and the sitemap is filtered by address
def test_a_site_reads_its_pages_from_its_sitemap():
    from use_cases.site_page import sitemap_urls

    site = {"main": "div#content", "sitemap": "https://nginx.org/sitemap.xml",
            "include": ["https://nginx.org/en/docs/*"], "exclude": ["*/dirindex.html"]}
    Declaration(name="n", licence="x", site=site)
    with pytest.raises(ValidationError, match="not both"):
        Declaration(name="n", licence="x", site=site, pages=["https://nginx.org/en/docs/a.html"])
    xml = ("<urlset><url><loc>https://nginx.org/en/docs/b.html</loc></url>"
           "<url><loc> https://nginx.org/en/docs/dirindex.html </loc></url>"
           "<url><loc>https://nginx.org/ru/docs/b.html</loc></url>"
           "<url><loc>https://nginx.org/en/docs/a.html</loc></url></urlset>")
    assert sitemap_urls(xml, site["include"], site["exclude"]) == [
        "https://nginx.org/en/docs/a.html", "https://nginx.org/en/docs/b.html"]


# a site's release is read from its own page before any page is fetched; a release no category lists is refused
def test_a_site_reads_its_release_first_and_refuses_one_its_category_does_not_list(monkeypatch, tmp_path):
    from errors import Final
    from models.corpus import DataSource
    from use_cases import source_intake

    with pytest.raises(ValidationError, match="both or neither"):
        Declaration(name="n", licence="x", pages=["https://a/b"], site={"main": "m", "release_page": "https://a/d"})
    site = {"main": "div#content", "release": "1.31.6", "release_page": "https://nginx.org/en/download.html",
            "release_pattern": r"nginx-(\d+\.\d+\.\d+)"}
    written = {"live": "1.31.6"}

    def download(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"<a>nginx-{written['live']}</a>" if url.endswith("download.html") else "<p>page</p>")
        return True

    monkeypatch.setattr(source_intake.fetch, "download", download)
    monkeypatch.setattr(source_intake.site_page, "prepared", lambda html, main, drop: None)
    declared = Declaration(name="nginx-org-en", licence="x", pages=["https://nginx.org/en/docs/a.html"], site=site,
                           categories=["nginx"]).model_dump(exclude_none=True)
    source = DataSource(name="nginx-org-en", declaration=declared)

    _, _, state = source_intake.gather(source, tmp_path, tmp_path)
    assert "release" not in state
    written["live"] = "9.9.9"
    with pytest.raises(Final, match="documents 9.9.9 now"):
        source_intake.gather(source, tmp_path, tmp_path)
    import config

    nginx = config.settings.categories["nginx"]
    monkeypatch.setattr(nginx, "versions", ["9.9.9", *nginx.versions])
    _, _, state = source_intake.gather(source, tmp_path, tmp_path)
    assert state["release"] == "9.9.9" and (tmp_path / "release").read_text() == "9.9.9"
