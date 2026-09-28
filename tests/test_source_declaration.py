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
def test_every_route_knob_is_a_source_knob_and_said_at_the_door():
    import re
    from pathlib import Path

    from config import RouteCfg
    from job_handlers import reading
    from sources.declaration import IntakeOverride

    own = set(IntakeOverride.model_fields) - {"settings"}
    stand = set(RouteCfg.model_fields) - {"min_layer_chars", "min_raster_run", "suspect_min_words"}
    assert own == stand
    assert set(reading._SHAPES) <= set(RouteCfg.model_fields)
    source = (Path(__file__).resolve().parent.parent / "app" / "mcp_ops.py").read_text()
    said = source[source.index('name="set_source_intake"') : source.index("def set_source_intake(")]
    assert own <= set(re.findall(r"`(\w+)`", said))
