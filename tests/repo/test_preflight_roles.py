def _seen(declared: dict, served: dict) -> dict:
    # the shape `stand_health.roles()` answers the preflight with, drift named by its one rule
    from use_cases.stand_health import drifting_roles

    return {"declared": declared, "served": served, "drift": drifting_roles(declared, served)}


def test_a_role_the_stand_serves_and_the_file_never_declares_is_drift(preflight):
    # gemma3:4b served as the generator while the file said llama
    declared = {"generation": "llama3.1:8b"}
    served = {"generation": "llama3.1:8b", "reranking": "bge-reranker"}

    assert preflight.role_drift(_seen(declared, served)) == [
        "reranking: the stand serves bge-reranker, the config declares no such role"
    ]


def test_an_asleep_vllm_is_not_printed_with_a_spill_s_words(preflight, monkeypatch):
    # an asleep judge passes the check, and its line must not read as a spill
    import json

    seen = {
        "judging": {"model": "Qwen/Q", "engine": "vllm", "on_card": False, "spilled": False},
        "generation": {"model": "llama", "engine": "ollama", "on_card": True, "spilled": False},
    }
    monkeypatch.setattr(preflight, "_in_worker", lambda code: json.dumps(seen))
    monkeypatch.setattr(preflight, "_card", lambda: "card free 600 of 7805 MiB")
    ok, line = preflight.models_are_on_the_card()
    assert ok and "judging=Qwen/Q@vllm not on the card now" in line and "off the card" not in line
    seen["generation"].update(on_card=False, spilled=True)
    seen["embedding"] = {"model": "bge-m3", "engine": "ollama", "on_card": True, "spilled": False}
    ok, line = preflight.models_are_on_the_card()
    assert not ok and "generation=llama@ollama spilled to the cpu" in line


def test_a_name_that_differs_is_drift_and_a_matching_pair_is_not(preflight):
    declared = {"generation": "llama3.1:8b", "judging": "qwen2.5:7b"}

    assert preflight.role_drift(_seen(declared, dict(declared))) == []
    assert preflight.role_drift(_seen(declared, {**declared, "generation": "gemma3:4b"})) == [
        "generation: config says llama3.1:8b, the stand serves gemma3:4b"
    ]
    assert preflight.role_drift(_seen(declared, {"judging": "qwen2.5:7b"})) == [
        "generation: config says llama3.1:8b, the stand serves nothing"
    ]


def test_a_prompt_activated_past_the_file_is_drift(preflight):
    declared = {"judge_faithfulness": 3, "grade_chunk": 1}
    assert preflight.prompt_drift(declared, dict(declared)) == []
    assert preflight.prompt_drift(declared, {"judge_faithfulness": 4}) == [
        "grade_chunk: config says v1, the stand serves vnone",
        "judge_faithfulness: config says v3, the stand serves v4",
    ]


def test_the_source_files_check_is_among_the_checks(preflight):
    assert preflight.sources_match_their_files in preflight.CHECKS


# after a guest pass the resident model was the ragas one, and its 16384 read as the generator's window
def test_the_window_check_does_not_compare_another_model(preflight, monkeypatch):
    seen = {
        "engine": "ollama",
        "generator": "llama3.1:8b",
        "declared": 8192,
        "asked": "qwen2.5:7b-w16384",
        "served": 16384,
        "refuses_past_it": False,
    }
    monkeypatch.setattr(preflight, "_in_worker", lambda code: __import__("json").dumps(seen))
    ok, said = preflight.window_matches_config()
    assert ok and "the generator is not resident, qwen2.5:7b-w16384 is" in said
    seen.update(asked="llama3.1:8b", served=4096)
    assert preflight.window_matches_config()[0] is False


def test_a_versioned_category_without_its_newest_in_the_variant_is_refused(preflight):
    ok, said = preflight.newest_versions_verdict({"held": {"postgresql": ["17"]}, "newest": {"postgresql": "18"}})
    assert not ok and "not its newest 18" in said
    ok, _ = preflight.newest_versions_verdict({"held": {"postgresql": ["18", "17"]}, "newest": {"postgresql": "18"}})
    assert ok
    assert preflight.newest_versions_verdict({"held": {}, "newest": {"postgresql": "18"}})[0]
    assert preflight.newest_versions_are_searchable in preflight.CHECKS


# the check reads what a search reads, through the search's own functions, not a copy of the query
def test_the_newest_check_asks_the_rows_search_reads(preflight, monkeypatch):
    asked = []
    monkeypatch.setattr(preflight, "_in_worker", lambda code: asked.append(code) or '{"held": {}, "newest": {}}')
    assert preflight.newest_versions_are_searchable()[0]
    assert "db.versions_held(" in asked[0] and "db.newest()" in asked[0] and "data_chunks" not in asked[0]


def test_a_folder_source_missing_on_the_host_is_refused(preflight):
    here = {"folders": ["datasets/inbox/books/ostep"], "there": [True]}
    assert preflight.source_folders_verdict({"ostep": here})[0]
    gone = {"folders": ["datasets/inbox/books/lost"], "there": [False]}
    ok, said = preflight.source_folders_verdict({"lost": gone})
    assert not ok and "datasets/inbox/books/lost is not on this host" in said
    assert preflight.source_folders_are_there in preflight.CHECKS


# a seeded row whose source file is gone is named, so it is removed rather than indexed on in silence
def test_a_seeded_row_without_its_file_is_named(preflight):
    assert preflight.seeded_rows_verdict([])[0]
    ok, said = preflight.seeded_rows_verdict(["old-book"])
    assert not ok and "old-book" in said
