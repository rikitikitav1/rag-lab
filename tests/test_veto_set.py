import pytest
from evals import build_veto
from use_cases.retrieval_compare import clean_gold, heading_text


def test_a_family_is_named_by_prefix_and_commands_are_not_one():
    assert build_veto._family_of("cheatsheets/vim.md") == "cheatsheets"
    assert build_veto._family_of("redis-doc/docs/about/_index.md") == "redis-doc/docs"
    # no heading that is a question: a question made from the file stem is a label
    assert build_veto._family_of("redis-doc/commands/get.md") is None
    assert build_veto._family_of("java-interview-questions/README.md") is None


def test_the_stored_heading_is_the_one_the_matcher_will_look_for():
    # `heading_text` strips a numeric prefix and `clean_gold` does not
    section = "Redis license > 12. Licenses of dependencies"
    stored = build_veto._leaf(section)
    assert stored == "Licenses of dependencies"
    assert clean_gold(stored) == heading_text(section)


def test_a_plan_is_the_same_list_twice_and_a_different_one_under_another_seed(monkeypatch):
    rows = [
        {"family": "notes", "source": f"notes/{i}.md", "heading": f"Heading number {i}",
         "language": "rus"}
        for i in range(40)
    ]
    monkeypatch.setattr(build_veto, "candidates", lambda *_: rows)
    first = build_veto.plan("hygiene_v1", [], "", {"notes": 10})
    assert first == build_veto.plan("hygiene_v1", [], "", {"notes": 10})
    assert first != build_veto.plan("other_seed", [], "", {"notes": 10})
    assert len(first) == 10


def test_a_quota_larger_than_the_family_takes_what_there_is(monkeypatch):
    rows = [{"family": "notes", "source": "notes/a.md", "heading": "Heading number one",
             "language": "rus"}]
    monkeypatch.setattr(build_veto, "candidates", lambda *_: rows)
    assert len(build_veto.plan("seed", [], "", {"notes": 5})) == 1


def test_a_set_without_a_seed_cannot_be_rebuilt():
    with pytest.raises(ValueError, match="seed"):
        build_veto.build(seed="")


def test_the_families_the_veto_reads_are_the_ones_the_criterion_cannot_see():
    # devinterview is the criterion's own population; a veto over it would veto nothing
    assert "devinterview" not in build_veto.FAMILIES
    assert set(build_veto.QUOTAS) == set(build_veto.FAMILIES)
