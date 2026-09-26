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

    for reader in Base._registry:
        assert files.of_reader(reader).reader == reader
    assert {s.reader for s in found.values()} <= set(Base._registry)


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
    assert len(rows) == 4 + 173
    assert rows["notes"]["kind"] == "local" and rows["notes"]["path"] == "/notes"
    assert rows["ado-net-interview-questions"]["git_url"].endswith("/ado-net-interview-questions")
    assert rows["redis-doc"]["licence"] == "CC BY-SA 4.0" and rows["redis-doc"]["language"] == "en"


def test_a_source_file_refuses_a_ref_the_index_would_not_read():
    with pytest.raises(ValueError, match="clones the default branch whole"):
        SourceFile(name="x", language="en", licence="MIT", git={"repo": "https://a/b", "ref": "v1"})


def test_a_row_answers_to_its_file_and_names_the_variants_cut_by_another_version(found):
    now = files.digest(found["interview"])
    said = files.drift("ado-net-interview-questions", {"clean_1024": now, "baseline": "old"})
    assert said["file"] == "sources/interview.yaml" and said["moved"] == ["baseline"]
    assert files.drift("a-source-declared-by-hand", {}) is None


def test_a_row_of_a_family_and_of_a_folder(found):
    assert files.row_of(found["notes"], "notes") == {
        "name": "notes",
        "language": "ru",
        "licence": "the owner's own",
        "kind": "local",
        "git_url": None,
        "path": "/notes",
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


def test_a_source_file_refuses_what_the_index_does_not_read():
    with pytest.raises(ValueError, match="onboarded through the door"):
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
        [("redis-doc", {"clean_1024": "old", "baseline": now}), ("renamed-away", {"clean_1024": "x"}), ("notes", {})]
    )
    assert report == {
        "moved": {"sources/redis-doc.yaml": ["clean_1024"]},
        "orphaned": ["renamed-away"],
        "unrecorded": 1,
    }
    ok, said = preflight.source_files_verdict(report)
    assert not ok and "sources/redis-doc.yaml: clean_1024" in said and "renamed-away" in said


def test_a_family_that_drifts_drifts_by_its_rows(found):
    assert files.drifting_rows() == ["notes"]


def test_the_map_names_the_rows_a_source_may_cover(found):
    assert config.settings.technologies["postgresql"].versions == ["18", "17"]
    assert found["redis-doc"].technologies == ["redis"]
    empty = files.empty_rows(found)
    assert "redis" not in empty and "system-design" not in empty and "kafka" in empty


def test_a_technology_outside_the_map_refuses(found):
    bad = found["redis-doc"].model_copy(update={"technologies": ["cobol"]})
    with pytest.raises(ValueError, match="not rows of config/technologies.yaml"):
        files._refuse_unmapped({"redis-doc": bad})
