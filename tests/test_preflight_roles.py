def test_a_role_the_stand_serves_and_the_file_never_declares_is_drift(preflight):
    # gemma3:4b served as the generator while the file said llama
    declared = {"generation": "llama3.1:8b"}
    served = {"generation": "llama3.1:8b", "reranking": "bge-reranker"}

    assert preflight.role_drift(declared, served) == [
        "reranking: the stand serves bge-reranker, the config declares no such role"
    ]


def test_a_name_that_differs_is_drift_and_a_matching_pair_is_not(preflight):
    declared = {"generation": "llama3.1:8b", "judging": "qwen2.5:7b"}

    assert preflight.role_drift(declared, dict(declared)) == []
    assert preflight.role_drift(declared, {**declared, "generation": "gemma3:4b"}) == [
        "generation: config says llama3.1:8b, the stand serves gemma3:4b"
    ]
    assert preflight.role_drift(declared, {"judging": "qwen2.5:7b"}) == [
        "generation: config says llama3.1:8b, the stand serves nothing"
    ]


def test_the_two_homes_of_the_drift_rule_answer_the_same_question(preflight):
    # the predicate is spelled twice on purpose; two green tests that never meet are not
    from use_cases.stand_health import drifting_roles

    cases = [
        ({"generation": "llama3.1:8b"}, {"generation": "llama3.1:8b"}),
        ({"generation": "llama3.1:8b"}, {"generation": "gemma3:4b"}),
        ({"generation": "llama3.1:8b"}, {}),
        ({}, {"judging": "qwen2.5:7b"}),
    ]
    for declared, served in cases:
        by_name = drifting_roles(declared, served)
        as_sentences = preflight.role_drift(declared, served)
        assert bool(by_name) == bool(as_sentences), (declared, served)
        assert len(by_name) == len(as_sentences), (declared, served)
