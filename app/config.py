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


class QuestionsCfg(BaseModel):
    path: str


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


class AppConfig(BaseModel):
    retrieval: RetrievalCfg
    rerank: RerankCfg
    ingestion: IngestionCfg
    ignored_sources: set[str]
    repos_dir: str
    prompts_dir: str
    sources: SourcesCfg
    questions: QuestionsCfg
    llm: LlmCfg
    postgres: PostgresCfg


def _load(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    service = raw["service"]
    return AppConfig(
        retrieval=service["retrieval"],
        rerank=service.get("rerank", {}),
        ingestion=service["ingestion"],
        ignored_sources=service["ignored_sources"],
        repos_dir=service["repos_dir"],
        prompts_dir=service["prompts_dir"],
        sources=service["sources"],
        questions=service["questions"],
        llm=raw["llm"],
        postgres=raw["postgres"],
    )


settings = _load(CONFIG_PATH)
