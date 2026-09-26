import shutil
from pathlib import Path

import config
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _with(tmp_path, change) -> str:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    # the data file stays where the repository keeps it
    raw["sources"] = {name: str(ROOT / path) for name, path in raw["sources"].items()}
    change(raw)
    # the sections a process owns and the roles sit in `config/` beside the file, as in the tree
    shutil.copytree(ROOT / "config", tmp_path / "config", dirs_exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_a_missing_or_misspelled_key_fails_the_start_rather_than_falling_back(tmp_path):
    # a default in the code stood in for a measured number: `ef_search` fell to "auto", the corpus to baseline
    assert config._load(_with(tmp_path, lambda raw: None)).retrieval.ef_search == 100
    with pytest.raises(Exception, match="ef_search"):
        config._load(_with(tmp_path, lambda raw: raw["retrieval"].pop("ef_search")))
    with pytest.raises(Exception, match="variant"):
        config._load(_with(tmp_path, lambda raw: raw["corpus"].pop("variant")))
    with pytest.raises(Exception, match="reranck"):
        config._load(_with(tmp_path, lambda raw: raw.update(reranck={})))
    with pytest.raises(Exception, match="gate_signl"):
        config._load(_with(tmp_path, lambda raw: raw["agent"]["gate"].update(gate_signl="distance")))


def test_the_moved_keys_hold_the_values_they_held_before_the_move():
    # the layout changed and no number did: the old flat names, read at their new places
    s = config._load(str(ROOT / "config.yaml"))
    assert s.retrieval.keyword.model_dump() == {
        "query": "and",
        "rank": "ts_rank",
        "norm": 0,
        "query_lang": "function_words",
    }
    assert (s.retrieval.ef_search, s.retrieval.distance_threshold, s.retrieval.results_limit) == (100, 0.55, 5)
    assert s.verdict.criterion_sets == ["paraphrased_v2_ru", "paraphrased_v2"] and s.verdict.veto_sets == ["veto_v1"]
    assert s.verdict.search_depth.model_dump() == {
        "ef_ladder": [100, 200, 400],
        "recall_gate": 0.98,
        "max_mrr_loss": 0.01,
        "max_questions_lost": 0,
    }
    assert (s.verdict.index_alive.recall, s.verdict.index_alive.questions) == (0.9, 40)
    assert s.agent.gate.model_dump() == {
        "signal": "distance",
        "weak_distance": 0.39,
        "weak_threshold": 0.5,
        "candidates": 5,
    }
    assert s.agent.topic_threshold == {"ru": 0.4560, "en": 0.4374}
    assert (s.ingestion.batch_size, s.ingestion.commit_size) == (100, 1000)
    assert s.sources.interview.language == "eng" and len(s.sources.interview.repos) == 173


def test_the_record_names_the_keyword_switches_as_before():
    # a run snapshot writes these names, and a changed one reads as a changed search
    assert config.KEYWORD_SWITCHES == ("query", "rank", "norm", "query_lang")
    assert set(config.keyword_switches()) == set(config.KEYWORD_SWITCHES)


def test_the_engines_compose_addresses_are_the_engines_the_config_seeds():
    # seed sowed four, compose addressed six and the readme named five
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["x-engine-env"]
    # a blank one keeps a host value out of the containers and addresses nothing
    addressed = {key.removesuffix("_BASE_URL") for key, value in compose.items() if key.endswith("_BASE_URL") and value}
    seeded = {engine.env_prefix for engine in config._load(str(ROOT / "config.yaml")).engines}
    # the seeded ollama is addressed by `llm.base_url`, every other engine by compose
    assert addressed == seeded - {"OLLAMA"}


def test_the_topic_gate_reads_a_three_letter_code_as_the_language_it_names():
    # the corpus and the questions spell it `eng`, the detector returns `en`, and the gate is keyed by two
    import config

    agent = config.settings.agent
    assert agent.topic_threshold_for("eng") == agent.topic_threshold_for("en")
    assert agent.topic_threshold_for("rus") == agent.topic_threshold_for("ru")
    # a language nobody measured still gets the most permissive threshold, not an error
    assert agent.topic_threshold_for("tl") == max(agent.topic_threshold.values())


def test_the_gate_says_whether_the_language_it_answered_for_was_ever_measured():
    import config

    agent = config.settings.agent
    assert agent.topic_threshold_is_measured("eng") and agent.topic_threshold_is_measured("ru")
    assert not agent.topic_threshold_is_measured("tl")
    assert not agent.topic_threshold_is_measured(None)


# the route may send a file to either tool, so a tool without its engine or settings is refused at load
def test_every_converter_tool_has_its_engine_and_settings():
    from pydantic import ValidationError

    raw = config.settings.intake.model_dump()
    raw["settings"].pop("mineru")
    with pytest.raises(ValidationError, match="no entry for"):
        config.IntakeCfg(**raw)


# twenty numbers left the code for config/: each holds what the code held before
def test_the_values_moved_out_of_the_code_hold_what_the_code_held():
    s = config._load(str(ROOT / "config.yaml"))
    e = s.evals
    assert (e.stats.bootstrap_n, e.stats.alpha, e.stats.seed) == (10_000, 0.05, 42)
    assert e.judge_correlation.model_dump() == {
        "rho_with_overlap": 0.3,
        "partial_gap": 0.1,
        "stratum_gap": 0.1,
        "min_rows": 100,
        "code_stratum": 0.2,
    }
    assert (e.judge_language.control_floor, e.judge_language.moves_allowed) == (0.90, 0.10)
    assert e.veto.quotas == {"cheatsheets": 80, "redis-doc/docs": 80, "notes": 80, "system-design-primer": 30}
    assert e.veto.min_heading == 12
    assert e.grade_curve.cuts == [None, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    assert (e.retrieval_compare.candidates, e.retrieval_compare.depth, e.retrieval_compare.rrf_k) == (100, 20, 60)
    assert e.retrieval_compare.cutoffs == [1, 3, 5, 10]
    assert (s.ingest_quality.measure.soup_alnum_ratio, s.ingest_quality.measure.prose_word_letters) == (0.55, 4)
    assert (s.llm.token_estimate.latin_divisor, s.llm.token_estimate.cyrillic_extra) == (1.02, 0.65)
    assert s.llm.measured_repeat_penalty == {"0.32.0": 1.1}


# a section is owned by one file: the same one in two would leave which wins to the order of a listing
def test_a_section_in_two_files_refuses_to_load(tmp_path):
    with pytest.raises(ValueError, match="already come from another file"):
        config._load(_with(tmp_path, lambda raw: raw.update(intake={})))


# a roles block left in the base would be dropped without a word, since the roles file replaces it
def test_roles_written_in_the_base_refuse_to_load(tmp_path):
    with pytest.raises(ValueError, match="the roles live in"):
        config._load(_with(tmp_path, lambda raw: raw["llm"].update(roles={})))


# the seed reads one version per purpose; two roles naming one purpose would let the last one win
def test_a_prompt_two_roles_name_refuses(tmp_path, monkeypatch):
    path = _with(tmp_path, lambda raw: None)
    roles = tmp_path / "config" / "roles.yaml"
    layer = yaml.safe_load(roles.read_text())
    layer["llm"]["roles"]["grading"]["prompts"] = {"judge_faithfulness": 3}
    roles.write_text(yaml.safe_dump(layer))
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    with pytest.raises(ValueError, match="named by two roles"):
        config.declared_prompts()


# a source's policy file decides what is indexed, so it counts as config for the stamp
def test_a_source_data_file_is_among_the_loaded_files():
    assert any(name.endswith("datasets/sources/interview.yaml") for name in config.loaded_files())
    assert not any("datasets" in name for name in config.loaded_files(with_data=False))
