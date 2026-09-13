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
    assert s.retrieval.keyword.model_dump() == {"query": "and", "rank": "ts_rank", "norm": 0,
                                                "query_lang": "function_words"}
    assert (s.retrieval.ef_search, s.retrieval.distance_threshold, s.retrieval.results_limit) == (100, 0.55, 5)
    assert s.verdict.criterion_sets == ["paraphrased_v2_ru", "paraphrased_v2"] and s.verdict.veto_sets == ["veto_v1"]
    assert s.verdict.search_depth.model_dump() == {"ef_ladder": [100, 200, 400], "recall_gate": 0.98,
                                                   "max_mrr_loss": 0.01, "max_questions_lost": 0}
    assert (s.verdict.index_alive.recall, s.verdict.index_alive.questions) == (0.9, 40)
    assert s.agent.gate.model_dump() == {"signal": "distance", "weak_distance": 0.39,
                                         "weak_threshold": 0.5, "candidates": 5}
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
