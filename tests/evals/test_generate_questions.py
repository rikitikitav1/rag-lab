import json
from types import SimpleNamespace

import pytest
from job_handlers import questions


@pytest.fixture(autouse=True)
def _an_empty_set(monkeypatch):
    monkeypatch.setattr(questions, "_held_by_section", lambda set_name, language: {})


def _export(words=200):
    facts = "The NX option sets the expiry only when the key has none. The GT option sets it when greater. "
    text = facts * (words // 19)
    return {
        "source": "redis-doc",
        "sections": [
            {"file": "r/expire.md", "section": "EXPIRE > Options", "chapter": "EXPIRE > Options",
             "versions": [], "words": words, "text": text},
            {"file": "r/stub.md", "section": "STUB > One", "chapter": "STUB > One", "versions": [], "words": 10,
             "text": "tiny"},
        ],
        "refused": [],
        "quota": {"EXPIRE > Options": 2, "STUB > One": 1},
    }


def _pair(n):
    return {"en": f"How do I keep the timeout only when none is set, case {n}?",
            "ru": f"Как задать срок, только если его нет, случай {n}?",
            "answer": "Use NX.", "evidence": ["The NX option sets the expiry only when the key has none.",
                                              "The GT option sets it when greater."][(n - 1) % 2]}


# a source's sections are asked in one call each, a stub never, and a kept pair is written as two rows of one gold
def test_a_source_is_asked_section_by_section_and_its_pairs_written(monkeypatch):
    asked, written = [], []
    monkeypatch.setattr(questions, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: _export())
    monkeypatch.setattr(questions.prompt_repo, "active", lambda purpose: ("{source}: write {pairs} pair(s)", 1))
    engine = SimpleNamespace(name="neuraldeep", env_prefix="NEURALDEEP")
    monkeypatch.setattr(questions.llm, "resolve", lambda role: SimpleNamespace(name="gemma-4-31b", engine=engine))
    monkeypatch.setattr(questions.llm, "sampler_of", lambda role, picked: {"temperature": 0.3, "max_tokens": 4096})
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: "abc123")
    queued = []
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: queued.append(kind))

    def ask(system, user, role):
        asked.append((system, user.splitlines()[0], role))
        reply = json.dumps({"pairs": [_pair(1), _pair(2)]})
        return SimpleNamespace(text=reply, prompt_tokens=100, completion_tokens=50, finish_reason="stop", parsed=None)

    monkeypatch.setattr(questions.llm, "ask", ask)
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (written.extend(rows) or len(rows), 0))
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "report.json")

    out = questions.generate_questions({"source": "redis-doc", "set_name": "smoke"})

    assert asked == [("redis-doc: write 2 pair(s)", "Section path: EXPIRE > Options", "questioning")]
    assert out["pairs_asked"] == 2 and out["pairs_kept"] == 2 and out["questions_written"] == 4
    assert out["kept_under_the_floor"] and out["prompt"]["version"] == 1 and out["stubs"] == 1
    assert (out["engine"], out["key_fingerprint"], out["sampler"]["max_tokens"]) == ("neuraldeep", "abc123", 4096)
    assert queued == ["embed_questions"]
    assert {r["pair_id"] for r in written} and len({r["pair_id"] for r in written}) == 2
    assert all(r["gold"] == {"file": "r/expire.md", "section": "EXPIRE > Options", "version": None} for r in written)


# a probe's cap keeps the spread: the biggest asks go first, one pair at a time
def test_a_probe_cap_takes_pairs_from_the_largest_asks_first():
    wanted = {("a", "x"): 3, ("a", "y"): 1, ("a", "z"): 2}
    assert questions._capped(wanted, 3) == {("a", "x"): 2, ("a", "z"): 1}
    assert questions._capped(wanted, 10) == wanted


# the pairs each refusal cost add up with the kept ones; a reply past its ask cost none
def test_the_pairs_lost_are_counted_by_the_pairs_not_the_replies():
    assert questions._lost([
        {"why": "the reply was cut at max_tokens", "count": 2},
        {"why": "a question repeats the heading"},
        {"why": "fewer pairs than asked", "count": 1},
        {"why": "pairs past the number asked", "count": 3},
    ]) == {"the reply was cut at max_tokens": 2, "a question repeats the heading": 1, "fewer pairs than asked": 1}


# stored replies are read again by today's checks; a pair on evidence the set holds is a repeat, a gone section counted
def test_a_report_is_reparsed_into_the_set_without_asking_anyone(monkeypatch, tmp_path):
    import json

    report = tmp_path / "question_set_smoke_redis_doc_20260930.json"
    report.write_text(json.dumps({"source": "redis-doc", "set_name": "smoke", "sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "calls": [
            {"asked": 2, "reply": json.dumps({"pairs": [_pair(1), _pair(2)]})}]},
        {"file": "gone.md", "section": "Gone", "calls": [{"asked": 1, "reply": "{}"}]},
    ]}))
    monkeypatch.setattr(questions.measurements, "FOLDER", tmp_path)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: _export())
    held = {("r/expire.md", "EXPIRE > Options", None): [
        {"en": "held?", "evidence": "The NX option sets the expiry only when the key has none."}]}
    monkeypatch.setattr(questions, "_held_by_section", lambda set_name, language: held)
    written, queued = [], []
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (written.extend(rows) or len(rows), 0))
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: queued.append(kind))
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "record.json")

    out = questions.reparse_questions({"source": "redis-doc", "set_name": "smoke", "report": report.name})

    assert out["pairs_kept_now"] == 1 and len(written) == 2 and out["sections_gone"] == 1
    assert queued == ["embed_questions"]


# a report from before the version field still finds a versioned section, by file and heading, while one stream has it
def test_a_report_without_versions_is_reparsed_on_a_versioned_source(monkeypatch, tmp_path):
    import json

    report = tmp_path / "question_set_smoke_redis_doc_20260930.json"
    report.write_text(json.dumps({"source": "redis-doc", "set_name": "smoke", "sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "calls": [
            {"asked": 1, "reply": json.dumps({"pairs": [_pair(2)]})}]}]}))
    exported = _export()
    exported["sections"][0] |= {"stream": "7.4", "versions": ["7.4"]}
    monkeypatch.setattr(questions.measurements, "FOLDER", tmp_path)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: exported)
    monkeypatch.setattr(questions, "_held_by_section", lambda set_name, language: {})
    written = []
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (written.extend(rows) or len(rows), 0))
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "record.json")

    out = questions.reparse_questions({"source": "redis-doc", "set_name": "smoke", "report": report.name})
    assert out["sections_gone"] == 0 and {r["gold"]["version"] for r in written} == {"7.4"}


# a reply is read against the block it was written from; a block the intake has moved since is counted, not read
def test_a_reparse_skips_a_section_whose_block_moved(monkeypatch, tmp_path):
    import json

    report = tmp_path / "question_set_smoke_redis_doc_20260930.json"
    report.write_text(json.dumps({"source": "redis-doc", "set_name": "smoke", "sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "calls": [
            {"asked": 1, "block": 0, "block_sha": "not-the-same", "reply": json.dumps({"pairs": [_pair(2)]})}]},
    ]}))
    monkeypatch.setattr(questions.measurements, "FOLDER", tmp_path)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: _export())
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (len(rows), 0))
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "record.json")

    out = questions.reparse_questions({"source": "redis-doc", "set_name": "smoke", "report": report.name})

    assert out["sections_moved"] == 1 and out["pairs_kept_now"] == 0


# a smoke goes section by section until enough pairs survive, and asks no section past that
def test_a_smoke_stops_once_enough_pairs_are_kept(monkeypatch):
    export = _export()
    export["sections"].append({"file": "r/ttl.md", "section": "TTL > Notes", "chapter": "TTL > Notes", "versions": [],
                               "words": 200, "text": export["sections"][0]["text"]})
    export["quota"]["TTL > Notes"] = 2
    asked = []
    monkeypatch.setattr(questions, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: export)
    monkeypatch.setattr(questions.prompt_repo, "active", lambda purpose: ("{source}: write {pairs} pair(s)", 1))
    engine = SimpleNamespace(name="neuraldeep", env_prefix="NEURALDEEP")
    monkeypatch.setattr(questions.llm, "resolve", lambda role: SimpleNamespace(name="gemma", engine=engine))
    monkeypatch.setattr(questions.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: "abc123")
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)

    def ask(system, user, role):
        asked.append(user.splitlines()[0])
        return SimpleNamespace(text=json.dumps({"pairs": [_pair(1), _pair(2)]}), prompt_tokens=1,
                               completion_tokens=1, finish_reason="stop", parsed=None)

    monkeypatch.setattr(questions.llm, "ask", ask)
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (len(rows), 0))
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "report.json")

    out = questions.generate_questions({"source": "redis-doc", "set_name": "smoke", "kept_at_least": 2})

    assert asked == ["Section path: EXPIRE > Options"] and out["pairs_kept"] == 2 and out["pairs_asked"] == 2


# a section the model refuses is lost alone with its reason, and the next section is still asked
def test_a_refused_section_is_lost_alone(monkeypatch):
    export = _export()
    export["sections"].append({"file": "r/ttl.md", "section": "TTL > Notes", "chapter": "TTL > Notes", "versions": [],
                               "words": 200, "text": export["sections"][0]["text"]})
    export["quota"]["TTL > Notes"] = 2
    engine = SimpleNamespace(name="neuraldeep", env_prefix="NEURALDEEP")
    monkeypatch.setattr(questions, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: export)
    monkeypatch.setattr(questions.prompt_repo, "active", lambda purpose: ("{source}: write {pairs} pair(s)", 1))
    monkeypatch.setattr(questions.llm, "resolve", lambda role: SimpleNamespace(name="gemma", engine=engine))
    monkeypatch.setattr(questions.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: "abc123")
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)

    def ask(system, user, role):
        if "EXPIRE" in user.splitlines()[0]:
            raise questions.llm.RequestRefused("http 400")
        return SimpleNamespace(text=json.dumps({"pairs": [_pair(1), _pair(2)]}), prompt_tokens=1,
                               completion_tokens=1, finish_reason="stop", parsed=None)

    monkeypatch.setattr(questions.llm, "ask", ask)
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (len(rows), 0))
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "report.json")

    out = questions.generate_questions({"source": "redis-doc", "set_name": "smoke"})

    assert out["pairs_kept"] == 2
    assert out["pairs_lost_by_reason"] == {questions.section_questions.REFUSED_CALL: 2}


# the generator reads a section's blocks, each with its share of pairs, and the gold stays the section
def test_a_section_is_asked_block_by_block(monkeypatch):
    export = _export()
    text = export["sections"][0]["text"]
    export["sections"][0]["blocks"] = [text[: len(text) // 2], text[len(text) // 2:]]
    engine = SimpleNamespace(name="neuraldeep", env_prefix="NEURALDEEP")
    seen, written = [], []
    monkeypatch.setattr(questions, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: export)
    monkeypatch.setattr(questions.prompt_repo, "active", lambda purpose: ("{source}: write {pairs} pair(s)", 1))
    monkeypatch.setattr(questions.llm, "resolve", lambda role: SimpleNamespace(name="gemma", engine=engine))
    monkeypatch.setattr(questions.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: "abc123")
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)

    def ask(system, user, role):
        seen.append(len(user))
        return SimpleNamespace(text=json.dumps({"pairs": [_pair(len(seen))]}), prompt_tokens=1,
                               completion_tokens=1, finish_reason="stop", parsed=None)

    monkeypatch.setattr(questions.llm, "ask", ask)
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (written.extend(rows) or len(rows), 0))
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "report.json")

    out = questions.generate_questions({"source": "redis-doc", "set_name": "smoke"})

    assert len(seen) == 2 and max(seen) < len(text), "each call reads one block, not the section"
    assert {row["gold"]["section"] for row in written} == {"EXPIRE > Options"} and out["pairs_kept"] == 2


# a saved set comes back as candidates with today's anchors; a pair whose section is gone is counted, not written
def test_a_saved_set_is_poured_back_without_generation(monkeypatch, tmp_path):
    import json

    from job_handlers import question_files as files

    monkeypatch.setattr(files, "SAVED", tmp_path)
    kept = {"file": "r/expire.md", "section": "EXPIRE > Options", "version": None}
    lines = [{"original_text": "How do I keep NX?", "language": "en", "gold": kept, "reference_answer": "NX",
              "evidence": "The NX option", "pair_id": "p1", "kind": "in_corpus",
              "evidence_at": {"block": 0, "block_sha": "abc", "char": 0}},
             {"original_text": "Gone?", "language": "en", "gold": {**kept, "section": "GONE"},
              "reference_answer": "x", "evidence": "x", "pair_id": "p2", "kind": "in_corpus"}]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    written = []
    monkeypatch.setattr(files.section_export, "of_source", lambda name: _export())
    monkeypatch.setattr(files.question_sets, "write_pairs", lambda rows: (written.extend(rows) or len(rows), 0))
    monkeypatch.setattr(files.job_queue, "enqueue", lambda kind, options: None)

    out = files.load_questions({"source": "redis-doc", "set_name": "s"})

    assert out["questions_written"] == 1 and out["pairs_of_gone_sections"] == 1
    assert written[0]["status"] == files.CANDIDATE and written[0]["set_name"] == "s"
    assert written[0]["evidence_at"]["block_sha"] == "abc", "the judge still opens the block the generator read"


def _stub_generation(monkeypatch, export, ask):
    engine = SimpleNamespace(name="neuraldeep", env_prefix="NEURALDEEP")
    monkeypatch.setattr(questions, "require_role_ready", lambda role, take_card=True: None)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: export)
    monkeypatch.setattr(questions.prompt_repo, "active", lambda purpose: ("{source}: write {pairs} pair(s)", 1))
    monkeypatch.setattr(questions.llm, "resolve", lambda role: SimpleNamespace(name="gemma", engine=engine))
    monkeypatch.setattr(questions.llm, "sampler_of", lambda role, picked: {})
    monkeypatch.setattr("job_handlers.base.key_fingerprint", lambda spec: None)
    monkeypatch.setattr(questions.llm, "ask", ask)
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (len(rows), 0))


# a retried pass asks a section only for the pairs its set lacks, and the prompt hears the ones it holds
def test_a_retried_generation_does_not_ask_again_for_what_the_set_holds(monkeypatch):
    asked = []

    def ask(system, user, role):
        asked.append((system, user))
        return SimpleNamespace(text=json.dumps({"pairs": [_pair(2)]}), prompt_tokens=1, completion_tokens=1,
                               finish_reason="stop", parsed=None)

    _stub_generation(monkeypatch, _export(), ask)
    held = {("r/expire.md", "EXPIRE > Options", None): [
        {"en": "Held question?", "evidence": "The NX option sets the expiry only when the key has none."}]}
    monkeypatch.setattr(questions, "_held_by_section", lambda set_name, language: held)
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)
    records = []
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: records.append(a[2]) or "r.json")

    out = questions.generate_questions({"source": "redis-doc", "set_name": "smoke"})

    assert [system for system, _ in asked] == ["redis-doc: write 1 pair(s)"] and "Held question?" in asked[0][1]
    assert out["pairs_already_held"] == 1 and out["pairs_kept"] == 1

    asked.clear()
    held[("r/expire.md", "EXPIRE > Options", None)].append({"en": "Second?", "evidence": "The GT option"})
    out = questions.generate_questions({"source": "redis-doc", "set_name": "smoke"})
    assert asked == [] and out["pairs_asked"] == 0 and out["pairs_already_held"] == 2


# a pass that breaks writes its report before the failure goes on to the worker, and embeds what it wrote
def test_a_broken_generation_writes_its_report_and_fails(monkeypatch):
    def ask(system, user, role):
        raise questions.llm.ServerFailed("502 after the client's retry")

    _stub_generation(monkeypatch, _export(), ask)
    queued, records = [], []
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: queued.append(kind))
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: records.append(a[2]) or "r.json")

    with pytest.raises(questions.llm.ServerFailed):
        questions.generate_questions({"source": "redis-doc", "set_name": "smoke"})
    assert len(records) == 1 and "502" in records[0]["stopped_by"] and queued == ["embed_questions"]


# a reparsed reply is checked against the block it was written from, as the generation checked it
def test_a_reparsed_reply_is_checked_against_its_own_block(monkeypatch, tmp_path):
    export = _export()
    row = export["sections"][0]
    row["blocks"] = ["The NX option sets the expiry only when the key has none.", "The GT option sets it when greater."]
    sha = questions.section_questions.block_sha(row["blocks"][1])
    report = tmp_path / "question_set_smoke_redis_doc_20260930.json"
    report.write_text(json.dumps({"source": "redis-doc", "set_name": "smoke", "sections": [
        {"file": "r/expire.md", "section": "EXPIRE > Options", "calls": [
            {"asked": 2, "block": 1, "block_sha": sha, "reply": json.dumps({"pairs": [_pair(1), _pair(2)]})}]},
    ]}))
    monkeypatch.setattr(questions.measurements, "FOLDER", tmp_path)
    monkeypatch.setattr(questions.section_export, "of_source", lambda name: export)
    written = []
    monkeypatch.setattr(questions.question_sets, "write_pairs", lambda rows: (written.extend(rows) or len(rows), 0))
    monkeypatch.setattr(questions.job_queue, "enqueue", lambda kind, options: None)
    monkeypatch.setattr(questions.measurements, "record", lambda *a, **kw: "record.json")

    out = questions.reparse_questions({"source": "redis-doc", "set_name": "smoke", "report": report.name})

    assert out["pairs_kept_now"] == 1 and {r["evidence"] for r in written} == {"The GT option sets it when greater."}
    assert all(r["evidence_at"]["block"] == 1 and r["evidence_at"]["char"] is not None for r in written)
