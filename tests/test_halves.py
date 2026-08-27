import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "retrieval_report", Path(__file__).resolve().parent.parent / "scripts" / "retrieval_report.py"
)
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def rows(spec_):
    return [{"id": qid, "repo": repo} for repo, ids in spec_.items() for qid in ids]


def test_a_question_keeps_its_side_when_the_set_grows():
    # the property the pre-registration rests on: adding questions must not move the ones
    # already there. A rule drawn inside a repository balances better and fails this
    small = rr.halves(rows({"ruby": [1, 2], "go": [7, 8]}))
    grown = rr.halves(rows({"ruby": [1, 2, 3, 4, 5], "go": [7, 8, 9]}))
    assert all(grown[qid] == side for qid, side in small.items())


def test_the_side_depends_on_nothing_but_the_id():
    assert rr.halves(rows({"ruby": [1, 2, 3]})) == rr.halves(rows({"go": [3, 2, 1]}))


def test_the_split_is_stable_across_calls():
    spec_ = {"ruby": [1, 2, 3, 4], "go": [9, 8, 7]}
    assert rr.halves(rows(spec_)) == rr.halves(rows(spec_))


def test_every_question_lands_on_one_of_two_sides():
    assigned = rr.halves(rows({"ruby": list(range(50))}))
    assert set(assigned.values()) == {"A", "B"}
    assert len(assigned) == 50


def test_repo_coverage_counts_the_topics_a_half_actually_saw():
    got = rows({"ruby": [1, 2], "go": [3, 4]})
    assert rr.repo_coverage(got, {1, 2}) == 1
    assert rr.repo_coverage(got, {1, 3}) == 2




def report(ids, questions_hash):
    return {"questions_hash": questions_hash, "rows": [{"id": i, "repo": "r"} for i in ids]}


def test_a_grown_set_is_not_a_different_procedure():
    before = report([1, 2, 3], "aaa")
    after = report([1, 2, 3, 4, 5], "bbb")
    differ = [("questions_hash", "aaa", "bbb")]
    assert rr.set_grew(before, after, differ) is True


def test_a_shrunk_or_shifted_set_is_still_refused():
    assert rr.set_grew(report([1, 2, 3], "a"), report([1, 2], "b"),
                       [("questions_hash", "a", "b")]) is False
    assert rr.set_grew(report([1, 2], "a"), report([2, 3], "b"),
                       [("questions_hash", "a", "b")]) is False


def test_growth_does_not_excuse_a_changed_procedure():
    before = report([1, 2], "a")
    after = report([1, 2, 3], "b")
    differ = [("questions_hash", "a", "b"), ("search", "exact", "hnsw")]
    assert rr.set_grew(before, after, differ) is False
