from pathlib import Path

import config
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_a_trimmed_config_still_boots_and_a_misspelled_key_does_not():
    # `_load` hands the whole section over, so the per-key fallbacks it applied are gone
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    rest = {"llm": raw["llm"], "postgres": raw["postgres"], "mcp_integrations": {}}
    trimmed = {k: v for k, v in raw["service"].items() if k not in ("rerank", "agent", "fts")}

    assert config.AppConfig(**trimmed, **rest).rerank == config.RerankCfg()
    with pytest.raises(Exception, match="reranck"):
        config.AppConfig(**{**raw["service"], "reranck": {}}, **rest)
