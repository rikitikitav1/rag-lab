import job_specs
import pytest


def test_every_registered_type_is_named_here():
    # a type with neither a spec nor a place on the free list is a hole nobody declared
    import job_handlers

    unnamed = set(job_handlers.HANDLERS) - set(job_specs.SPECS) - set(job_specs.FREE)
    assert not unnamed, f"these types accept anything and nobody said so: {sorted(unnamed)}"


def test_a_run_with_no_target_is_refused_wherever_it_comes_from():
    # `POST /v1/eval/run {}` validated cleanly and swept all 19645 questions through a typed door
    with pytest.raises(job_specs.Refused):
        job_specs.check("eval_run", {"run_name": "r"})
    job_specs.check("eval_run", {"run_name": "r", "set_name": "interview"})
    job_specs.check("eval_run", {"run_name": "r", "question_ids": [1, 2]})


def test_judging_everything_is_allowed_only_when_the_sweep_says_so():
    # no target judges every unjudged row in the base, which is right for the sweep and nothing else
    with pytest.raises(job_specs.Refused):
        job_specs.check("judge_answers", {})
    job_specs.check("judge_answers", {"sweep": True})
    # the sweep re-enqueues itself with `sweep + 1`, so a flag type refuses its own second round
    job_specs.check("judge_answers", {"run_name": "r", "sweep": 2})
    job_specs.check("judge_answers", {"sweep": 3})
    job_specs.check("judge_answers", {"run_name": "r"})
    job_specs.check("judge_answers", {"log_ids": [1]})


def test_the_key_the_worker_injects_is_not_part_of_anyone_request():
    job_specs.check("eval_run", {"run_name": "r", "set_name": "s", "_job_id": 12})


def test_an_unspecced_type_passes_rather_than_blocking_the_queue():
    job_specs.check("index_data", {"source": "all", "variant": "clean_1024"})


def test_the_universal_door_refuses_with_the_text_the_handler_would_have_used(client, monkeypatch):
    # a door of its own per type meant fourteen doors and four enqueuers each defended once
    bad = client.post("/v1/job", json={"type": "eval_run", "options": {"run_name": "r"}})
    assert bad.status_code == 400
    assert "needs a target" in bad.json()["detail"]

    unknown = client.post("/v1/job", json={"type": "no_such_thing", "options": {}})
    assert unknown.status_code == 400
    assert "no such job type" in unknown.json()["detail"]


def test_a_bound_the_door_carried_is_now_carried_by_the_queue():
    # the door held sixteen fields with bounds and the spec held three: `k` of a million passed
    with pytest.raises(job_specs.Refused):
        job_specs.check("eval_run", {"run_name": "r", "set_name": "s", "k": 1_000_000})
    with pytest.raises(job_specs.Refused):
        job_specs.check("eval_run", {"run_name": "r", "set_name": "s", "max_hops": 99})
    job_specs.check("eval_run", {"run_name": "r", "set_name": "s", "k": 5})


def test_the_lane_belongs_to_the_type_not_to_whoever_asks():
    # a card job placed on the io lane is two processes on one card, and that cost 3239 seconds once
    assert job_specs.lane("pull_llm_model") == "io"
    assert job_specs.lane("eval_run") == "default"


def test_a_misspelt_option_is_refused_rather_than_ignored():
    # `rerank` misspelt used to be accepted and dropped, and the run kept the default silently
    with pytest.raises(job_specs.Refused):
        job_specs.check("eval_run", {"run_name": "r", "set_name": "s", "reranking": True})


def test_the_dictionaries_the_stand_actually_builds_all_pass():
    # forbidding extras before reading the dynamic callers would refuse work that is correct today
    from use_cases import rejudge

    job_specs.check("judge_answers", rejudge.arm_options({"judge_model": "q"}, "a1", 50, 3))
    job_specs.check("judge_answers", {"run_name": "r", "sweep": 2, "judge_prompts": {}})
    # the deferral is the worker's own bookkeeping, and it validates on the worker's side
    job_specs.check("judge_answers", {"run_name": "r", "deferred_seconds": 90},
                    from_the_worker=True)
    job_specs.check("judge_guest_axes", {"run_name": "r", "judge_width": None})


def test_the_cap_lives_where_both_doors_meet_it():
    # the guest cap sat on `/eval/guest-axes`, and the universal door reached the same job past it
    import inspect

    from job_handlers import judging

    assert "MAX_GUEST_ROWS" in inspect.getsource(judging.judge_guest_axes)


def test_the_types_that_had_no_door_are_no_longer_unchecked():
    # the universal door exposed them for the first time, and nothing said what they accept
    assert job_specs.FREE == ()
    for name in ("index_data", "build_vector_index", "build_veto_set", "paraphrase_questions"):
        assert name in job_specs.SPECS
    with pytest.raises(job_specs.Refused):
        job_specs.check("build_vector_index", {"variant": "../etc"})


def test_a_model_name_meets_the_same_rules_at_both_doors():
    # `POST /v1/job` pulled an image by any name at all while `POST /v1/model` refused the same one
    job_specs.check("pull_llm_model", {"name": "llama3.1:8b"})
    for bad in ("evil.example.com/x/y", "a" * 200, "has spaces"):
        with pytest.raises(job_specs.Refused):
            job_specs.check("pull_llm_model", {"name": bad})


def test_deleting_weights_asks_whether_a_role_still_points_at_them():
    # the typed door answers 409 and deletes the record first; the universal one reached the handler
    import inspect

    from job_handlers import model_ops

    body = inspect.getsource(model_ops.delete_llm_model)
    assert "ModelRole" in body and "reassign" in body


def test_the_guest_cap_counts_the_rows_the_pass_will_walk():
    # the door counted the run's whole debt, the handler the drawn subsample: `sample=50` split them
    import inspect

    from api.v1 import eval as eval_mod
    from job_handlers import judging

    door = inspect.getsource(eval_mod.enqueue_guest_axes)
    assert "min(owed, request.sample)" in door, "the door still caps the whole debt"
    assert "MAX_GUEST_ROWS" in inspect.getsource(judging.judge_guest_axes)


def test_a_retried_job_still_validates_with_the_counter_the_worker_wrote():
    # the worker writes `attempts` into options on every retry, and the second end refused it
    import job_specs

    job_specs.check("pull_llm_model", {"name": "qwen2.5:7b", "engine_id": 1, "attempts": 2},
                    from_the_worker=True)


def test_the_bookkeeping_of_a_retry_is_not_accepted_from_a_caller():
    # `attempts` past the cap makes a job give up on its first error, and nothing would say why
    import job_specs
    import pytest

    with pytest.raises(job_specs.Refused, match="attempts"):
        job_specs.check("pull_llm_model", {"name": "qwen2.5:7b", "attempts": 99})
