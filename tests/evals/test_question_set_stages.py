import json
from types import SimpleNamespace

from evals import pair_judge, question_sets
from models.eval import ACCEPTED, CANDIDATE, REFUSED


def _q(pair, lang, status, why=None):
    return SimpleNamespace(pair_id=pair, language=lang, status=status, acceptance_why=why)


# a generated set reads its way from the generator's reports and its rows: who settled what, by language
def test_a_generated_set_tells_its_stages(monkeypatch, tmp_path):
    (tmp_path / "question_set_gen_x_src_20260930.json").write_text(json.dumps(
        {"set_name": "gen_x", "pairs_asked": 10, "pairs_kept": 6, "pairs_lost_by_reason": {"cut": 4}}))
    (tmp_path / "question_set_gen_x_v2_src_20260930.json").write_text(json.dumps(
        {"set_name": "gen_x_v2", "pairs_asked": 99, "pairs_kept": 99}))
    (tmp_path / "question_reparse_gen_x_src_20260930.json").write_text(json.dumps(
        {"set_name": "gen_x", "pairs_kept_now": 2}))
    monkeypatch.setattr(question_sets.measurements, "FOLDER", tmp_path)
    rows = [
        _q("a", "en", ACCEPTED), _q("a", "ru", ACCEPTED),
        _q("b", "en", REFUSED, pair_judge.NOT_ANSWERED), _q("b", "ru", REFUSED, pair_judge.NOT_ANSWERED),
        _q("c", "en", REFUSED, "the reader found no answer in the section"),
        _q("c", "ru", REFUSED, "the reader found no answer in the section"),
        _q("d", "en", CANDIDATE, pair_judge.SPLIT), _q("d", "ru", CANDIDATE, pair_judge.SPLIT),
    ]
    way = question_sets.stages("gen_x", rows)
    assert way["generated"] == {"pairs_asked": 10, "pairs_kept": 6, "pairs_lost_by_reason": {"cut": 4}}
    assert way["reparsed_pairs"] == 2 and way["pairs_in_set"] == 4 and way["accepted_pairs"] == 1
    assert way["by_language"]["en"] == {"status": {"accepted": 1, "refused": 2, "candidate": 1},
                                        "refused_by_judge": 1, "refused_by_sieve": 1,
                                        "anchored": 0, "anchors_unread": 4, "shares_heading_word": 0}
    assert way["under_the_floor"] is True
    assert question_sets.stages("hand_written", [_q(None, "en", ACCEPTED)]) is None
