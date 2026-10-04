import config
import pytest
from sources import files
from sources.declaration import SourceFile


@pytest.fixture(scope="module")
def found():
    return files.source_files()


def test_every_reader_class_has_exactly_one_file(found):
    from sources import factory  # noqa: F401
    from sources.base import Base

    for reader, cls in Base._registry.items():
        if cls.one_file:
            assert files.of_reader(reader).reader == reader
    assert {s.reader for s in found.values()} - {None} <= set(Base._registry)


def test_the_interview_family_holds_its_173_banks(found):
    family = found["interview"].git_family
    assert family.base_url == "https://github.com/Devinterview-io" and len(family.repos) == 173


def test_the_veto_families_are_the_ones_the_quotas_count(found):
    named = files.veto_families()
    assert set(named) == set(config.settings.evals.veto.quotas)
    # the redis family is a prefix: its command pages stay out
    assert named["redis-doc/docs"] == "redis-doc/docs/"


def test_the_format_files_hold_the_whole_vocabulary():
    import formats

    docbook = formats.format_of("docbook")
    assert {"refentry", "simplesect", "reference", "preface", "bibliography", "sect5", "refsect3"} <= set(
        docbook.sections
    )
    assert formats.format_of("latex").levels["subsubsection"] == 4


def test_a_source_file_takes_one_origin():
    with pytest.raises(ValueError, match="exactly one of"):
        SourceFile(name="x", language="en", licence="MIT")
    with pytest.raises(ValueError, match="exactly one of"):
        SourceFile(
            name="x", language="en", licence="MIT", folder="a", git_family={"base_url": "https://a", "repos": ["b"]}
        )


def test_the_seed_rows_unroll_a_family_and_keep_a_folder(found):
    import seed

    rows = {r["name"]: r for r in seed._source_rows(found)}
    assert len(rows) == 8 + 173
    assert rows["arangodb-docs"]["kind"] == "local" and rows["arangodb-docs"]["path"].endswith("arangodb/3.12")
    assert rows["nginx-org-ru"]["kind"] == "pages" and rows["nginx-org-ru"]["language"] == "ru"
    assert rows["eloquent-javascript"]["kind"] == "urls" and rows["eloquent-javascript"]["git_url"] is None
    assert rows["ado-net-interview-questions"]["git_url"].endswith("/ado-net-interview-questions")
    assert rows["redis-doc"]["licence"] == "CC BY-SA 4.0" and rows["redis-doc"]["language"] == "en"


def test_a_source_file_refuses_a_ref_the_index_would_not_read():
    with pytest.raises(ValueError, match="clones a branch whole; a ref is named per version"):
        SourceFile(name="x", language="en", licence="MIT", git={"repo": "https://a/b", "ref": "v1"})


def _declared(source) -> dict:
    return source.model_dump(mode="json", exclude_defaults=True)


def test_a_row_answers_to_its_declaration_and_names_the_variants_cut_by_another_version(found):
    now = files.digest(found["interview"])
    indexed = {"clean_1024": now, "baseline": "old"}
    said = files.drift("ado-net-interview-questions", indexed, _declared(found["interview"]))
    assert said["source"] == "interview" and said["moved"] == ["baseline"]
    assert files.drift("a-source-declared-by-hand", {}, None) is None


def test_a_row_of_a_family_and_of_a_folder(found):
    book = SourceFile(name="book", language="ru", licence="CC BY", folder="datasets/book")
    assert files.row_of(book, "book") == {
        "name": "book",
        "language": "ru",
        "licence": "CC BY",
        "kind": "local",
        "git_url": None,
        "path": "datasets/book",
    }
    repo = files.row_of(found["interview"], "aws-interview-questions")
    assert repo["kind"] == "git" and repo["git_url"] == "https://github.com/Devinterview-io/aws-interview-questions"


def test_the_digest_follows_the_rules_and_not_the_reasons_or_the_comments(found):
    sheets = found["cheatsheets"]
    reworded = sheets.model_copy(
        update={"skip_when_hygienic": {k: "another reason" for k in sheets.skip_when_hygienic}}
    )
    assert files.digest(reworded) == files.digest(sheets)
    assert files.digest(sheets.model_copy(update={"skip": [*sheets.skip, "new"]})) != files.digest(sheets)
    for metadata in ({"licence": "CC0"}, {"veto_families": []}, {"drifts": True}):
        assert files.digest(sheets.model_copy(update=metadata)) == files.digest(sheets)
    assert files.digest(sheets.model_copy(update={"language": "ru"})) != files.digest(sheets)


# a book the stand downloads on a cold start is a file too; one origin still, never two
def test_a_source_file_may_name_urls_and_still_takes_one_origin():
    SourceFile(name="x", language="en", licence="MIT", urls=["https://a/b.pdf"])
    with pytest.raises(ValueError, match="exactly one of"):
        SourceFile(name="x", language="en", licence="MIT", folder="a", urls=["https://a/b.pdf"])


def test_no_source_files_refuses_rather_than_cutting_an_empty_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.yaml"))
    with pytest.raises(FileNotFoundError, match="the corpus would be empty"):
        files.source_files()


def test_one_row_declared_by_two_files_refuses(tmp_path, monkeypatch):
    (tmp_path / "sources").mkdir()
    for name in ("a", "b"):
        (tmp_path / "sources" / f"{name}.yaml").write_text(
            f"name: {name}\nlanguage: en\nlicence: MIT\ngit_family: {{base_url: https://x, repos: [same]}}\n"
        )
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.yaml"))
    with pytest.raises(ValueError, match="already declared by a"):
        files.source_files()


def test_the_report_names_moved_files_with_their_variants_and_orphans(found, preflight):
    now = files.digest(found["redis-doc"])
    report = files.drift_report(
        [
            ("redis-doc", {"clean_1024": "old", "baseline": now}, _declared(found["redis-doc"])),
            ("renamed-away", {"clean_1024": "x"}, None),
            ("cheatsheets", {}, _declared(found["cheatsheets"])),
        ]
    )
    assert report == {
        "moved": {"redis-doc": ["clean_1024"]},
        "fields": {},
        "orphaned": ["renamed-away"],
        "unrecorded": 1,
    }
    ok, said = preflight.source_files_verdict(report)
    assert not ok and "redis-doc: clean_1024" in said and "renamed-away" in said


# a field left at its default is not in the digest, and a moved digest names the fields it was cut by otherwise
def test_a_new_default_field_moves_no_digest_and_a_moved_one_names_its_fields(found, preflight):
    sheets = found["cheatsheets"]
    assert files.digest(sheets.model_copy(update={"markup": None, "skip_paths": []})) == files.digest(sheets)
    was = files.cut_rules(sheets)
    moved = sheets.model_copy(update={"skip_paths": ["old/*"]})
    report = files.drift_report(
        [("cheatsheets", {"v": files.digest(sheets)}, _declared(moved), {"v": was})]
    )
    assert report["moved"] == {"cheatsheets": ["v"]} and report["fields"] == {"cheatsheets": ["skip_paths"]}
    ok, said = preflight.source_files_verdict(report)
    assert not ok and "(by skip_paths)" in said


def test_a_row_whose_declaration_drifts_is_named_by_the_base(monkeypatch):
    import orm.sync_db

    rows = [("live", {"name": "live", "folder": "/live", "drifts": True}), ("book", {"name": "book"}), ("old", None)]

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt):
            return type("R", (), {"all": lambda self: rows})()

    monkeypatch.setattr(orm.sync_db, "Session", _Session)
    from use_cases import source_intake

    assert source_intake.drifting_rows() == ["live"]


def test_the_map_names_the_rows_a_source_may_cover(found):
    assert config.settings.categories["postgresql"].versions == ["18", "17"]
    assert found["redis-doc"].categories == ["redis"]
    empty = files.empty_rows(found)
    assert "redis" not in empty and "system-design" not in empty and "kafka" in empty


def test_a_category_outside_the_map_refuses(found):
    bad = found["redis-doc"].model_copy(update={"categories": ["cobol"]})
    with pytest.raises(ValueError, match="not rows of config/categories.yaml"):
        files._refuse_unmapped({"redis-doc": bad})


# a site's release its category does not list would key chunks a search with no version never reads
def test_a_site_release_outside_its_category_refuses(found):
    assert found["nginx-org-en"].site.release in config.settings.categories["nginx"].versions
    bad = found["nginx-org-en"].model_copy(
        update={"site": found["nginx-org-en"].site.model_copy(update={"release": "0.1"})}
    )
    with pytest.raises(ValueError, match="release 0.1 is not listed for nginx"):
        files._refuse_unmapped({"nginx-org-en": bad})


# a source without its category's newest version would answer no search that names none, so it is refused
def test_versions_without_the_newest_refuse():
    from sources.declaration import SourceFile

    older = SourceFile(
        name="pg17",
        licence="x",
        categories=["postgresql"],
        git={"repo": "https://a/b"},
        versions={"17": {"ref": "REL_17"}},
    )
    with pytest.raises(ValueError, match="lack postgresql's newest 18"):
        files._refuse_unmapped({"pg17": older})


# the hand-written notes declare their trust: with the readers gone nothing else names them notes
def test_the_notes_sources_declare_their_trust(found):
    from use_cases.dedup import source_trust

    for name in ("interview", "cheatsheets", "system-design-primer"):
        assert source_trust(found[name].model_dump(mode="json", exclude_none=True)) == "notes", name
