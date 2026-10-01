from types import SimpleNamespace

import pytest
from evals import question_acceptance as qa
from job_handlers import question_reading as reading


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    # the pass's stamp is read where it is built, beside the other job helpers
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: None)

EVIDENCE = "The NX option sets the expiry only when the key has none."


# a quote naming the evidence's place holds its words; one from elsewhere in the section does not
def test_a_quote_is_read_against_the_evidence_by_its_words():
    assert qa.overlap(EVIDENCE, "Use NX: " + EVIDENCE)["held"] == 1.0
    assert qa.overlap(EVIDENCE, "XX sets it only when a timeout already exists")["held"] < qa.EVIDENCE_HELD
    assert qa.overlap(EVIDENCE, "")["held"] == 0.0


# a limit's number is its evidence, however short
def test_a_number_counts_as_a_word_at_any_length():
    assert qa.overlap("a key holds up to 64 fields", "Each key holds up to 64 fields.")["held"] == 1.0
    assert qa.overlap("a key holds up to 64 fields", "Each key holds up to 32 fields.")["held"] == 0.75


# the marker is the whole reply; a quote that holds its words is a quote
def test_not_in_section_is_read_only_as_the_whole_reply():
    assert qa.verdict("Not in section.", "stop", EVIDENCE)["answerable"] is False
    assert qa.verdict("Options not in section headers are ignored", "stop", EVIDENCE)["answerable"] is True


def test_a_reply_is_one_of_answered_here_answered_elsewhere_none_or_unread():
    assert qa.verdict("NX option sets the expiry only when the key has none.", "stop", EVIDENCE)["why"] is None
    assert qa.verdict("The GT option sets expiry when greater.", "stop", EVIDENCE)["why"] == qa.ELSEWHERE
    assert qa.verdict("not in section", "stop", EVIDENCE) == {
        "answerable": False, "why": qa.NO_ANSWER, "held": None, "f1": None}
    assert qa.verdict(EVIDENCE, "length", EVIDENCE)["why"] == qa.UNREAD
    assert qa.verdict("  ", "stop", EVIDENCE)["answerable"] is None


# both halves without an answer refuse the pair; both on the evidence accept it; anything else waits for the judge
def test_a_pair_is_settled_whole():
    here = {"answerable": True, "why": None}
    elsewhere = {"answerable": True, "why": qa.ELSEWHERE}
    none = {"answerable": False, "why": qa.NO_ANSWER}
    assert qa.pair_outcome([here, here]) == ("accepted", None)
    assert qa.pair_outcome([none, none]) == ("refused", qa.NO_ANSWER)
    # the reader answered one language and not the other: the judge reads it, the pair is not refused
    assert qa.pair_outcome([here, none]) == ("undecided", qa.NO_ANSWER)
    assert qa.pair_outcome([elsewhere, none]) == ("undecided", qa.ELSEWHERE)
    assert qa.pair_outcome([here, elsewhere]) == ("undecided", qa.ELSEWHERE)


def _question(id, language, text):
    gold = {"file": "r/expire.md", "section": "EXPIRE > Options", "version": None}
    return SimpleNamespace(id=id, pair_id=f"p{(id + 1) // 2}", language=language, original_text=text,
                           evidence=EVIDENCE, gold=gold)


# each question is asked on its own with its section, and the pair's two rows are settled by one outcome
def test_a_sets_candidates_are_read_pair_by_pair(monkeypatch):
    pairs = {"p1": [_question(1, "en", "keep the timeout?"), _question(2, "ru", "сохранить срок?")],
             "p2": [_question(3, "en", "what does GT do?"), _question(4, "ru", "что делает GT?")]}
    replies = {1: EVIDENCE, 2: EVIDENCE, 3: EVIDENCE, 4: "NOT IN SECTION"}
    asked, settled = [], []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "text": EVIDENCE}]})
    monkeypatch.setattr(reading, "_candidate_pairs", lambda set_name, files, again, every, **kw: pairs)
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("quote it", 1))
    engine = SimpleNamespace(name="ollama")
    monkeypatch.setattr(reading.llm, "resolve", lambda role: SimpleNamespace(name="llama3.1:8b", engine=engine))
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {"temperature": 0})
    monkeypatch.setattr(reading, "_settle", lambda members, verdicts, outcome, why: settled.append(
        ([m.id for m in members], outcome, why)))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 1})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: "report.json")

    def ask(system, user, role):
        question = next(q for members in pairs.values() for q in members if q.original_text in user)
        asked.append((question.id, role))
        return SimpleNamespace(text=replies[question.id], finish_reason="stop")

    monkeypatch.setattr(reading.llm, "ask", ask)

    out = reading.accept_questions({"source": "redis-doc", "set_name": "smoke"})

    assert asked == [(1, "accepting"), (2, "accepting"), (3, "accepting"), (4, "accepting")]
    assert settled == [([1, 2], "accepted", None), ([3, 4], "undecided", qa.NO_ANSWER)]
    assert out["pairs"] == {"accepted": 1, "undecided": 1} and out["pairs_read"] == 2
    assert out["by_language"]["ru"] == {"asked": 2, "answered": 1, "no_answer": 1, "evidence_held": 1,
                                        "undecided_halves": 1}
    assert out["source_under_the_floor"] and out["why"] == {qa.NO_ANSWER: 1}


# a pass that does not settle reads the same pairs and writes nothing, so two of them can be compared
def test_a_pass_that_does_not_settle_writes_nothing(monkeypatch):
    settled = []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "text": EVIDENCE}]})
    seen_every = []
    pair = {"p1": [_question(1, "en", "keep it?"), _question(2, "ru", "сохранить?")]}
    monkeypatch.setattr(reading, "_candidate_pairs",
                        lambda set_name, files, again, every, **kw: seen_every.append(every) or pair)
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("quote it", 1))
    monkeypatch.setattr(reading.llm, "resolve", lambda role: SimpleNamespace(
        name="llama3.1:8b", engine=SimpleNamespace(name="ollama")))
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr(reading.llm, "ask", lambda system, user, role: SimpleNamespace(
        text=EVIDENCE, finish_reason="stop"))
    monkeypatch.setattr(reading, "_settle", lambda *a: settled.append(a))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 0})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: "report.json")

    out = reading.accept_questions({"source": "redis-doc", "set_name": "smoke", "settle": False})

    assert settled == [] and out["pairs"] == {"accepted": 1} and out["settled"] is False
    assert seen_every == [True]

    out = reading.accept_questions({"source": "redis-doc", "set_name": "smoke", "every": True})

    assert len(settled) == 1 and out["settled"] is True
    assert seen_every == [True, True], "every reads the settled pairs and settles them again"


# a section past the reader's window leaves its pair to the judge and the pass goes on
def test_a_section_past_the_window_is_a_reason_not_a_failure(monkeypatch):
    settled = []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "text": EVIDENCE}]})
    monkeypatch.setattr(reading, "_candidate_pairs", lambda set_name, files, again, every, **kw: {
        "p1": [_question(1, "en", "keep it?"), _question(2, "ru", "сохранить?")]})
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("quote it", 1))
    monkeypatch.setattr(reading.llm, "resolve", lambda role: SimpleNamespace(
        name="llama3.1:8b-w16384", engine=SimpleNamespace(name="ollama")))
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {})

    def ask(system, user, role):
        raise reading.llm.InputOverWindow("the input is at least 20000 tokens")

    monkeypatch.setattr(reading.llm, "ask", ask)
    monkeypatch.setattr(reading, "_settle", lambda members, verdicts, outcome, why: settled.append((outcome, why)))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 0})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: "report.json")

    out = reading.accept_questions({"source": "redis-doc", "set_name": "smoke"})

    assert settled == [("undecided", qa.TOO_LONG)] and out["why"] == {qa.TOO_LONG: 2}


# a pair whose section the export refuses now is settled once, so a rerun does not read it again
def test_a_pair_whose_section_is_gone_is_refused_once(monkeypatch):
    settled = []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": []})
    monkeypatch.setattr(reading, "_candidate_pairs", lambda set_name, files, again, every, **kw: {
        "p1": [_question(1, "en", "keep it?"), _question(2, "ru", "сохранить?")]})
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("quote it", 1))
    monkeypatch.setattr(reading.llm, "resolve", lambda role: SimpleNamespace(
        name="llama3.1:8b-w12288", engine=SimpleNamespace(name="ollama")))
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr(reading, "_settle", lambda members, verdicts, outcome, why: settled.append((outcome, why)))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 0})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: "report.json")

    out = reading.accept_questions({"source": "redis-doc", "set_name": "smoke"})

    assert settled == [("refused", qa.SECTION_GONE)] and out["pairs"] == {"refused": 1}


# a long section is read in overlapping windows; a short one whole
def test_a_long_section_is_read_in_overlapping_windows(monkeypatch):
    monkeypatch.setattr(qa, "WINDOW_WORDS", 10)
    monkeypatch.setattr(qa, "WINDOW_OVERLAP", 2)
    text = " ".join(f"w{i}" for i in range(25))
    parts = qa.windows(text)
    assert [p.split()[0] for p in parts] == ["w0", "w8", "w16"] and parts[-1].split()[-1] == "w24"
    assert qa.windows("short text") == ["short text"]


# the reader goes window by window and stops at the first that answers
def test_the_reader_stops_at_the_first_window_that_answers(monkeypatch):
    monkeypatch.setattr(qa, "WINDOW_WORDS", 10)
    monkeypatch.setattr(qa, "WINDOW_OVERLAP", 2)
    text = " ".join(f"w{i}" for i in range(25))
    asked = []

    def ask(system, user, role):
        asked.append(user)
        answer = EVIDENCE if len(asked) == 2 else "NOT IN SECTION"
        return SimpleNamespace(text=answer, finish_reason="stop")

    monkeypatch.setattr(reading.llm, "ask", ask)
    reply, said, read = reading._read(_question(1, "en", "keep it?"), text, "quote it")
    assert read == 2 and said["why"] is None and len(asked) == 2


# a window long in characters though short in words is cut again at spaces; a padded table is left whole
def test_a_window_long_in_characters_is_cut_again(monkeypatch):
    from evals import question_acceptance as qa

    monkeypatch.setattr(qa, "WINDOW_CHARS", 50)
    links = " ".join(["[a](x/y/z/long/path.html#anchor)"] * 5)
    cut = qa.windows(links)
    assert len(cut) > 1 and all(len(w) <= 50 for w in cut) and " ".join(cut) == links
    table = "| a |          b |\n|---|------------|\n| c |          d |"
    assert qa.windows(table) == [table]


# a request the reader's engine refuses leaves that pair undecided with its reason, and the pass reads on
def test_a_refused_request_is_a_reason_and_a_broken_pass_keeps_its_report(monkeypatch):
    settled, records = [], []
    monkeypatch.setattr(reading, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(reading.section_export, "of_source", lambda name: {"sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "text": EVIDENCE}]})
    monkeypatch.setattr(reading, "_candidate_pairs", lambda set_name, files, again, every, **kw: {
        "p1": [_question(1, "en", "keep it?"), _question(2, "ru", "сохранить?")],
        "p2": [_question(3, "en", "other?"), _question(4, "ru", "другое?")]})
    monkeypatch.setattr(reading.prompt_repo, "active", lambda purpose: ("quote it", 1))
    monkeypatch.setattr(reading.llm, "resolve", lambda role: SimpleNamespace(
        name="llama3.1:8b", engine=SimpleNamespace(name="ollama")))
    monkeypatch.setattr(reading.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr(reading, "_settle", lambda members, verdicts, outcome, why: settled.append((outcome, why)))
    monkeypatch.setattr(reading, "_set_counts", lambda set_name, files: {"accepted_pairs": 0})
    monkeypatch.setattr(reading.measurements, "record", lambda *a, **kw: records.append(a[2]) or "report.json")

    def ask(system, user, role):
        if "other" in user or "другое" in user:
            raise reading.llm.ServerFailed("502")
        raise reading.llm.RequestRefused("400")

    monkeypatch.setattr(reading.llm, "ask", ask)

    with pytest.raises(reading.llm.ServerFailed):
        reading.accept_questions({"source": "redis-doc", "set_name": "smoke"})
    assert settled == [("undecided", qa.REFUSED_CALL)]
    assert records[0]["pairs"] == {"undecided": 1} and "502" in records[0]["stopped_by"]
