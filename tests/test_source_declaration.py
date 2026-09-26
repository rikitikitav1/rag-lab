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
