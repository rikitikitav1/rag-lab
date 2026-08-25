import os

import yaml
from pydantic import BaseModel

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


class RoleCfg(BaseModel):
    model: str
    options: dict = {}


class RetrievalCfg(BaseModel):
    distance_threshold: float
    results_limit: int
    limit_vector: int
    limit_keywords: int
    rrf_k: int


class RerankCfg(BaseModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    candidates: int = 20
    top: int = 3


class AgentCfg(BaseModel):
    max_hops: int = 4
    fallback_policy: str = "corpus_first"
    gate_candidates: int = 5
    weak_threshold: float = 0.5
    gate_signal: str = "distance"
    topic_threshold: float | None = None
    weak_distance: float = 0.39


class FtsCfg(BaseModel):
    languages: dict[str, str] = {"en": "english", "ru": "russian"}
    fallback: str = "english"


class CorpusCfg(BaseModel):
    description: str = (
        "the technical knowledge corpus (interview banks, "
        "system-design-primer, redis docs)"
    )


class IngestionCfg(BaseModel):
    batch_size: int
    commit_size: int
    chunk_max_size: int


class InterviewCfg(BaseModel):
    base_url: str
    language: str
    repos: list[str]


class SourcesCfg(BaseModel):
    interview: InterviewCfg


class LlmCfg(BaseModel):
    base_url: str
    roles: dict[str, RoleCfg]
    candidates: list[str] = []

    @property
    def pull_models(self) -> list[str]:
        return list({r.model for r in self.roles.values()} | set(self.candidates))


class PostgresCfg(BaseModel):
    host: str
    port: int
    dbname: str
    user: str


class McpIntegrationsCfg(BaseModel):
    secret_env: list[str] = []

    def secret(self, name: str) -> str:
        if name not in self.secret_env:
            return ""
        return os.getenv(name, "")


class AppConfig(BaseModel):
    retrieval: RetrievalCfg
    rerank: RerankCfg
    agent: AgentCfg
    ingestion: IngestionCfg
    fts: FtsCfg
    corpus: CorpusCfg
    ignored_sources: set[str]
    repos_dir: str
    prompts_dir: str
    sources: SourcesCfg
    llm: LlmCfg
    postgres: PostgresCfg
    mcp_integrations: McpIntegrationsCfg


def _load(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    service = raw["service"]
    return AppConfig(
        retrieval=service["retrieval"],
        rerank=service.get("rerank", {}),
        agent=service.get("agent", {}),
        ingestion=service["ingestion"],
        fts=service.get("fts", {}),
        corpus=service.get("corpus", {}),
        ignored_sources=service["ignored_sources"],
        repos_dir=service["repos_dir"],
        prompts_dir=service["prompts_dir"],
        sources=service["sources"],
        llm=raw["llm"],
        postgres=raw["postgres"],
        mcp_integrations=raw.get("mcp_integrations", {}),
    )


settings = _load(CONFIG_PATH)
