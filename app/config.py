import os
from typing import Literal

import samplers
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
# a file that replaces `llm.roles`, as the layout of a host without a card does
CONFIG_OVERLAY = os.getenv("CONFIG_OVERLAY")


# a mistyped key fails the start, and so does a missing one: no default stands in for a measured number
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleCfg(_Strict):
    model: str
    # unnamed means the seeded ollama, which is a default only while every config model is pulled
    engine: str | None = None
    options: dict = {}

    @field_validator("options")
    @classmethod
    def _sampler_keys_only(cls, v: dict) -> dict:
        return samplers.check(v)


# named as a run's record names them, so nothing translates between the two
class KeywordCfg(_Strict):
    query: str
    rank: str
    norm: int
    query_lang: str


class RetrievalCfg(_Strict):
    distance_threshold: float
    results_limit: int
    limit_vector: int
    limit_keywords: int
    rrf_k: int
    keyword: KeywordCfg
    # "auto" asks the planner for the deepest rung still walking the index, which moves
    ef_search: int | Literal["auto"]


class SearchDepthCfg(_Strict):
    ef_ladder: list[int]
    recall_gate: float
    max_mrr_loss: float
    max_questions_lost: int


class IndexAliveCfg(_Strict):
    recall: float
    questions: int


# what a run is judged on, read by the preflight and the reports and never by a query
class VerdictCfg(_Strict):
    criterion_sets: list[str]
    veto_sets: list[str]
    search_depth: SearchDepthCfg
    index_alive: IndexAliveCfg


class RerankCfg(_Strict):
    enabled: bool
    candidates: int


class AgentGateCfg(_Strict):
    signal: str
    weak_distance: float
    # both read at `signal: cross_encoder` and `either`
    weak_threshold: float
    candidates: int


class AgentCfg(_Strict):
    max_hops: int
    fallback_policy: str
    gate: AgentGateCfg
    # an off-domain english question sits closer to an english corpus than a russian one
    topic_threshold: float | dict[str, float] | None

    def topic_threshold_for(self, language: str | None) -> float | None:
        if not isinstance(self.topic_threshold, dict):
            return self.topic_threshold
        if not self.topic_threshold:
            return None
        # membership, not truthiness: zero switches the axis off, and `or` made it permissive
        if language in self.topic_threshold:
            return self.topic_threshold[language]
        # a language nobody measured gets the most permissive of the measured thresholds
        return max(self.topic_threshold.values())


class FtsCfg(_Strict):
    languages: dict[str, str]
    fallback: str


# typed like the gates that judge it: a typo fails the start, not the cut
class PolicyCfg(_Strict):
    chunker: Literal["legacy", "rooted", "structured"]
    max_chunk_size: int = Field(gt=0)
    ceiling_on: Literal["body", "content"] = "body"
    # off by default: it changes the cut, so it is a corpus variant of its own
    drop_boilerplate: bool = False

    # derived, not declared: two keys deciding one thing is how they came to disagree
    @property
    def header_prefix(self) -> bool:
        return self.chunker != "legacy"

    def model_dump(self, **kw) -> dict:
        return {**super().model_dump(**kw), "header_prefix": self.header_prefix}


class CorpusCfg(_Strict):
    description: str
    variant: str
    variants: dict[str, PolicyCfg]

    def policy(self, variant: str | None = None) -> dict:
        name = variant or self.variant
        if name not in self.variants:
            raise ValueError(f"corpus variant '{name}' has no declared policy in config")
        return self.variants[name].model_dump()

    # for the paths that must not raise on an unknown cut
    def policy_or_none(self, variant: str) -> dict | None:
        declared = self.variants.get(variant)
        return declared.model_dump() if declared else None


class GateCfg(_Strict):
    min: float | None = None
    max: float | None = None


# named fields, not a free dict: a typo in the judge's config must fail the start
class MetricGatesCfg(_Strict):
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


class MetricWeightsCfg(_Strict):
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


class IngestQualityCfg(_Strict):
    # thresholds live here, not in code: they are turned by hand and land in every report
    hard_gates: MetricGatesCfg
    soft_gates: MetricGatesCfg
    history_per_variant: int
    score_formula: str
    weights: MetricWeightsCfg


class IngestionCfg(_Strict):
    batch_size: int
    commit_size: int


class InterviewCfg(_Strict):
    base_url: str
    language: str
    repos: list[str]


class SourcesCfg(_Strict):
    interview: InterviewCfg


class EngineCfg(_Strict):
    name: str
    kind: Literal["ollama", "vllm", "openai_compatible"]
    env_prefix: str
    placement: Literal["gpu", "cpu", "gpu+cpu", "remote"]


class LlmCfg(_Strict):
    base_url: str
    roles: dict[str, RoleCfg]
    context_length: int

    # a role on another engine is registered through `/v1/model`, not pulled through `/api/pull`
    @property
    def pull_models(self) -> list[str]:
        return list({r.model for r in self.roles.values() if r.engine is None})


class PostgresCfg(_Strict):
    host: str
    port: int
    dbname: str
    user: str


class McpIntegrationsCfg(_Strict):
    secret_env: list[str]

    def secret(self, name: str) -> str:
        if name not in self.secret_env:
            return ""
        return os.getenv(name, "")


class AppConfig(_Strict):
    retrieval: RetrievalCfg
    verdict: VerdictCfg
    rerank: RerankCfg
    agent: AgentCfg
    ingestion: IngestionCfg
    ingest_quality: IngestQualityCfg
    fts: FtsCfg
    corpus: CorpusCfg
    repos_dir: str
    prompts_dir: str
    sources: SourcesCfg
    engines: list[EngineCfg]
    llm: LlmCfg
    postgres: PostgresCfg
    mcp_integrations: McpIntegrationsCfg


# the roles a stand cannot answer without: a layer dropping one fails at load; the rest are optional
REQUIRED_ROLES = ("generation", "embedding", "judging")


# a layer replaces the whole role table: merged, a judge without `engine` would inherit `vllm`
def _roles_of(overlay: str) -> dict:
    with open(overlay) as f:
        layer = yaml.safe_load(f) or {}
    llm = layer.get("llm") if isinstance(layer, dict) else None
    if set(layer or {}) != {"llm"} or not isinstance(llm, dict) or set(llm) != {"roles"}:
        raise ValueError(f"{overlay}: a layer carries `llm.roles` and nothing else")
    missing = [role for role in REQUIRED_ROLES if role not in (llm["roles"] or {})]
    if missing:
        raise ValueError(f"{overlay}: the layer drops {missing}, and the stand cannot answer without them")
    return llm["roles"]


# a data file is named by its path, beside the config that names it
def _data(here: str, path: str):
    with open(os.path.join(here, path)) as f:
        return yaml.safe_load(f)


def _load(path: str, overlay: str | None = None) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overlay:
        raw["llm"]["roles"] = _roles_of(overlay)
    here = os.path.dirname(os.path.abspath(path))
    if isinstance(raw.get("sources"), dict):
        raw["sources"] = {name: _data(here, file) for name, file in raw["sources"].items()}
    return AppConfig(**raw)


settings = _load(CONFIG_PATH, CONFIG_OVERLAY)


# the keyword leg by the name the record uses, which is the name the config uses
KEYWORD_SWITCHES = tuple(KeywordCfg.model_fields)


def keyword_switches() -> dict:
    return settings.retrieval.keyword.model_dump()
