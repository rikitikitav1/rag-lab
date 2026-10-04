import job_queue
import pytest
import search_scope
from job_handlers.base import Final
from use_cases import source_intake


# a variant under an eval, a vector index or a veto build is not removed; the whole-corpus index is found as `all` too
def test_every_job_that_reads_a_variant_holds_its_removal(monkeypatch):
    waiting = {}
    monkeypatch.setattr(job_queue, "pending_of_type", lambda type, **options: waiting.get((type, *options.values())))
    monkeypatch.setattr(job_queue, "pending_listing", lambda type, key, value: waiting.get((type, key, value)))
    readers = [("eval_run", "clean_2048"), ("build_vector_index", "clean_2048")]
    for held in [*readers, ("build_veto_set", "variants", "clean_2048")]:
        waiting.clear()
        waiting[held] = 7
        with pytest.raises(Final, match="7"):
            source_intake.remove_variant("clean_2048", "clean_1024")
    waiting.clear()
    waiting[("index_data", "all")] = 9
    assert source_intake.whole_corpus_index() == 9


# a label is read in the case the index wrote tags in
def test_a_scope_label_is_read_lowercased():

    assert search_scope.Scope(label="Redis").label == "redis"
    assert search_scope.as_scope("PostgreSQL").label == "postgresql"


# a veto build that names no variants or no cut_from reads clean_1024 by default, and holds its removal
def test_a_veto_build_on_its_defaults_holds_the_variants_it_reads(monkeypatch):
    waiting = {}
    monkeypatch.setattr(job_queue, "pending_of_type", lambda type, **options: waiting.get((type, *options.items())))
    monkeypatch.setattr(job_queue, "pending_listing", lambda type, key, value: None)
    waiting[("build_veto_set", ("cut_from", None))] = 5
    with pytest.raises(Final, match="5"):
        source_intake.remove_variant("clean_1024", "clean_2048")
    waiting.clear()
    waiting[("build_veto_set", ("variants", None))] = 6
    with pytest.raises(Final, match="6"):
        source_intake.remove_variant("clean_1024", "clean_2048")
    waiting[("build_veto_set", ("cut_from", "clean_512"))] = 8
    with pytest.raises(Final, match="8"):
        source_intake.remove_variant("clean_512", "clean_2048")
    assert source_intake.veto_reading("clean_4096") is None
