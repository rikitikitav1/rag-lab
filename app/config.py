import os
from typing import Literal

import yaml
from pydantic import BaseModel, Field

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
    keyword_query: str = "and"
    keyword_rank: str = "ts_rank"
    keyword_norm: int = 0
    query_lang: str = "function_words"
    # a number pins the depth; "auto" asks the planner for the deepest rung of the ladder
    # that still walks the index, because where it stops walking moves with the table
    ef_search: int | Literal["auto"] = "auto"
    ef_ladder: list[int] = [100, 200, 400]
    recall_gate: float = 0.98
    max_mrr_loss: float = 0.01
    max_questions_lost: int = 0
    index_alive_recall: float = 0.9
    index_alive_questions: int = 40
    criterion_sets: list[str] = ["paraphrased_v2_ru", "paraphrased_v2"]
    veto_sets: list[str] = ["veto_v1"]


class RerankCfg(BaseModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    candidates: int = 20


class AgentCfg(BaseModel):
    max_hops: int = 4
    fallback_policy: str = "corpus_first"
    gate_candidates: int = 5
    weak_threshold: float = 0.5
    gate_signal: str = "distance"
    # one number, or one per language: the axis is a distance to this corpus, and an
    # off-domain english question sits closer to an english corpus than a russian one
    topic_threshold: float | dict[str, float] | None = None
    weak_distance: float = 0.39

    def topic_threshold_for(self, language: str | None) -> float | None:
        if not isinstance(self.topic_threshold, dict):
            return self.topic_threshold
        if not self.topic_threshold:
            return None
        # membership, not truthiness: zero is how a language switches the axis off, and
        # `or` turned that into the most permissive threshold instead of into no gate
        if language in self.topic_threshold:
            return self.topic_threshold[language]
        # a language nobody measured gets the most permissive of the measured thresholds:
        # we refuse only where refusing was shown not to cost a real question
        return max(self.topic_threshold.values())


class FtsCfg(BaseModel):
    languages: dict[str, str] = {"en": "english", "ru": "russian"}
    fallback: str = "english"


# typed like the gates that judge it: a key nobody reads and a mix nobody meant are both
# refused at start, not discovered by a cut that came out wrong
class PolicyCfg(BaseModel):
    model_config = {"extra": "forbid"}
    chunker: Literal["legacy", "rooted", "structured"]
    max_chunk_size: int = Field(gt=0)
    ceiling_on: Literal["body", "content"] = "body"
    # a block repeated verbatim across half of a source's files is dropped, unless it is
    # the only carrier of its section. Off by default: it changes the cut, so it is a
    # corpus variant of its own and never a switch under an existing one
    drop_boilerplate: bool = False

    # derived, not declared: two keys deciding one thing is how they came to disagree
    @property
    def header_prefix(self) -> bool:
        return self.chunker != "legacy"

    def model_dump(self, **kw) -> dict:
        return {**super().model_dump(**kw), "header_prefix": self.header_prefix}


class CorpusCfg(BaseModel):
    description: str = (
        "the technical knowledge corpus (interview banks, "
        "system-design-primer, redis docs)"
    )
    variant: str = "baseline"
    variants: dict[str, PolicyCfg] = {}

    def policy(self, variant: str | None = None) -> dict:
        name = variant or self.variant
        if name not in self.variants:
            raise ValueError(f"corpus variant '{name}' has no declared policy in config")
        return self.variants[name].model_dump()

    # for the paths that must not raise on an unknown cut
    def policy_or_none(self, variant: str) -> dict | None:
        declared = self.variants.get(variant)
        return declared.model_dump() if declared else None


class GateCfg(BaseModel):
    model_config = {"extra": "forbid"}
    min: float | None = None
    max: float | None = None


# named fields rather than a free dict: a typo in the config of the instrument that
# judges the corpus must fail the start, not quietly gate nothing
class MetricGatesCfg(BaseModel):
    model_config = {"extra": "forbid"}
    section_coverage: GateCfg | None = None
    prefix_dominates: GateCfg | None = None
    dup_in_file: GateCfg | None = None
    dup_in_source: GateCfg | None = None
    boilerplate: GateCfg | None = None
    tiny: GateCfg | None = None
    orphans: GateCfg | None = None
    size_cut: GateCfg | None = None
    soup: GateCfg | None = None
    code_only: GateCfg | None = None


class MetricWeightsCfg(BaseModel):
    model_config = {"extra": "forbid"}
    section_coverage: float = 0
    prefix_dominates: float = 0
    dup_in_file: float = 0
    dup_in_source: float = 0
    boilerplate: float = 0
    tiny: float = 0
    orphans: float = 0
    size_cut: float = 0
    soup: float = 0
    code_only: float = 0


class IngestQualityCfg(BaseModel):
    # thresholds live here, not in code: they are turned by hand and land in every report
    hard_gates: MetricGatesCfg = MetricGatesCfg()
    soft_gates: MetricGatesCfg = MetricGatesCfg()
    history_per_variant: int = 20
    score_formula: str = "v1"
    weights: MetricWeightsCfg = MetricWeightsCfg()


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
    context_length: int = 8192
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
    ingest_quality: IngestQualityCfg = IngestQualityCfg()
    fts: FtsCfg
    corpus: CorpusCfg
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
        ingest_quality=service.get("ingest_quality", {}),
        fts=service.get("fts", {}),
        corpus=service.get("corpus", {}),
        repos_dir=service["repos_dir"],
        prompts_dir=service["prompts_dir"],
        sources=service["sources"],
        llm=raw["llm"],
        postgres=raw["postgres"],
        mcp_integrations=raw.get("mcp_integrations", {}),
    )


settings = _load(CONFIG_PATH)
