# a dry run names the model each role of the job would call: two paid jobs died on a broker nobody saw
from types import SimpleNamespace

import job_queue
import llm


def test_a_dry_run_names_the_seated_model_of_each_role_the_job_loads(monkeypatch):
    monkeypatch.setattr(job_queue, "_check", lambda type, options: None)
    monkeypatch.setattr(llm, "resolve", lambda role: SimpleNamespace(
        name="deepseek-v4", engine=SimpleNamespace(name="gonka")))
    said = job_queue.dry_run("generate_questions", {"source": "book", "set_name": "s1"})
    assert said["roles"] == {"questioning": {"model": "deepseek-v4", "engine": "gonka"}}
    assert job_queue.dry_run("load_variants", {"set_name": "s2", "rows": [
        {"source_question_id": 1, "original_text": "x"}]})["roles"] == {}
