from use_cases import question_set_removal as removal


def _holds(questions=3, answered=0, drawn_from=0):
    return {"questions": questions, "answered": answered, "drawn_from": drawn_from}


# a set goes only when nothing reads it: the verdict, a job, an answer log, a set drawn from it
def test_a_set_is_refused_while_anything_reads_it():
    assert removal.removal_refusal("smoke", _holds(), False, None) is None
    assert "no question set" in removal.removal_refusal("smoke", _holds(questions=0), False, None)
    assert "config/evals.yaml" in removal.removal_refusal("paraphrased_v2", _holds(), True, None)
    assert "job 7" in removal.removal_refusal("smoke", _holds(), False, 7)
    assert "2 answer logs" in removal.removal_refusal("smoke", _holds(answered=2), False, None)
    assert "4 questions of other sets" in removal.removal_refusal("smoke", _holds(drawn_from=4), False, None)


# a paraphrase job that names no set still reads one, so any job that reads sets holds the door
def test_any_job_that_reads_sets_holds_the_door(monkeypatch):
    import config
    import job_queue
    import pytest
    from errors import Final


    monkeypatch.setattr(removal, "_holds", lambda name: _holds())
    monkeypatch.setattr(job_queue, "pending_of_type", lambda t, **options: 12 if t == "paraphrase_questions" else None)
    monkeypatch.setattr(config.settings.verdict, "criterion_sets", [])
    monkeypatch.setattr(config.settings.verdict, "veto_sets", [])

    with pytest.raises(Final, match="job 12"):
        removal.remove("smoke")
