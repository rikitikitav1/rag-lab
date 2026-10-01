from types import SimpleNamespace

import pytest
from evals import pair_judge
from job_handlers import question_reading as reading


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    # the pass's stamp is read where it is built, beside the other job helpers
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: None)

TEXT = "Intro words here. Setting the *`size`* to 0 disables checking of the client_max_body_size. More text follows."


# the passage is found past the section's markup and carries the evidence with its neighbours
def test_the_passage_is_found_past_markup_and_holds_the_evidence():
    passage = pair_judge.around(TEXT, "Setting the size to 0 disables checking")
    assert passage.startswith("Intro words here.") and "client_max_body_size" in passage
    assert pair_judge.around(TEXT, "Nothing like this is written") is None


def test_the_reply_is_read_as_one_word_and_the_pair_whole():
    assert [pair_judge.verdict(r) for r in ("YES", "no.", "**Yes**", "Maybe", None)] == [True, False, True, None, None]
    assert pair_judge.pair_outcome([True, True]) == ("accepted", None)
    assert pair_judge.pair_outcome([False, False]) == ("refused", pair_judge.NOT_ANSWERED)
    assert pair_judge.pair_outcome([True, False]) == ("undecided", pair_judge.SPLIT)
    assert pair_judge.pair_outcome([True, None]) == ("undecided", pair_judge.UNREAD)


def _question(qid, language, text, evidence="Setting the size to 0 disables checking"):
    return SimpleNamespace(id=qid, language=language, original_text=text, evidence=evidence,
                           gold={"file": "n/core.md", "section": "Core"})


# a judged pair moves its status; a half whose evidence is gone from the section leaves the pair to a person
def test_a_judged_pair_is_settled_whole_and_a_lost_evidence_leaves_it(monkeypatch):
    settled = []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": [
        {"file": "n/core.md", "section": "Core", "text": TEXT}]})
    monkeypatch.setattr(reading, "_candidate_pairs", lambda set_name, files, again, every, **kw: {
        "p1": [_question(1, "en", "Can I turn the size check off?"), _question(2, "ru", "Как выключить проверку?")],
        "p2": [_question(3, "en", "What is X?", evidence="Not in the text at all"), _question(4, "ru", "Что такое X?")],
    })
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("judge it", 1))
    judge = SimpleNamespace(name="qwen", engine=SimpleNamespace(name="vllm"))
    monkeypatch.setattr(reading.llm, "resolve_for", lambda role, model=None: judge)
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {})
    asked_by = []
    monkeypatch.setattr(reading.llm, "ask", lambda system, user, role, model=None: asked_by.append(model)
                        or SimpleNamespace(text="YES"))
    monkeypatch.setattr(reading, "_settle_judged", lambda members, outcome, why: settled.append((
        [m.id for m in members], outcome, why)))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 1})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: "report.json")

    out = reading.judge_questions({"source": "nginx-org-en", "set_name": "smoke"})

    assert settled == [([1, 2], "accepted", None), ([3, 4], "undecided", pair_judge.NOT_FOUND)]
    assert out["pairs"] == {"accepted": 1, "undecided": 1} and out["second_judge"] is False
    assert set(asked_by) == {None}


# on a page built from a template the quote's first words repeat; the passage is the place that holds the rest
def test_the_passage_is_the_place_that_holds_the_whole_quote():
    block = "Default: none; Context: http, server, location. Sets the {} for the {} directive."
    text = block.format("buffer size", "first") + " filler" * 200 + " " + block.format("cache zone", "second")
    passage = pair_judge.around(text, block.format("cache zone", "second"))
    assert "cache zone" in passage.split("filler")[-1]


# a pair keeps where its evidence sits in the block the generator read; the judge opens that place while the block holds
def test_the_judge_opens_the_place_the_generator_read():
    import hashlib

    from evals import section_questions

    block = "Default: none; Context: http. Sets the {} for the {} directive."
    first, second = block.format("buffer size", "first"), block.format("cache zone", "second")
    text = first + " filler" * 200 + " " + second
    evidence = "Default: none; Context: http."
    at = section_questions.placed_in(1, second)({"evidence": evidence})
    assert at == {"block": 1, "block_sha": hashlib.sha256(second.encode()).hexdigest()[:12], "char": 0}
    question = SimpleNamespace(evidence=evidence, evidence_at=at)
    assert "cache zone" in reading._passage(question, text, [first, second])
    assert "buffer size" in reading._passage(question, text, [first, "moved since"])


# the judge reads only pairs the sieve has read; a gone section is left to the sieve, a refused call to a person
def test_the_judge_waits_for_the_sieve_and_leaves_a_gone_section_alone(monkeypatch):
    settled, seen = [], []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": [
        {"file": "n/core.md", "section": "Core", "text": TEXT}]})
    gone = _question(5, "en", "Gone?")
    gone.gold = {"file": "n/core.md", "section": "Elsewhere"}
    monkeypatch.setattr(reading, "_candidate_pairs", lambda set_name, files, again, every, sieved=False: seen.append(
        sieved) or {"p1": [_question(1, "en", "Turn it off?"), _question(2, "ru", "Выключить?")], "p9": [gone]})
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("judge it", 1))
    monkeypatch.setattr(reading.llm, "resolve_for", lambda role, model=None: SimpleNamespace(
        name="qwen", engine=SimpleNamespace(name="vllm")))
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {})

    def ask(system, user, role, model=None):
        raise reading.llm.RequestRefused("400")

    monkeypatch.setattr(reading.llm, "ask", ask)
    monkeypatch.setattr(reading, "_settle_judged", lambda members, outcome, why: settled.append((outcome, why)))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 0})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: "report.json")

    out = reading.judge_questions({"source": "nginx-org-en", "set_name": "smoke"})

    assert seen == [True]
    assert settled == [("undecided", pair_judge.REFUSED_CALL)]
    assert out["pairs"] == {"undecided": 1, "section_gone": 1}
