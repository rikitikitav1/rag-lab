import os
from typing import Literal, get_args

import samplers
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tool_names import Tool
from vocabulary import QueryLanguageRule

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
    # the prompt versions the seed activates on an empty database, by purpose
    prompts: dict[str, int] = {}

    @field_validator("options")
    @classmethod
    def _sampler_keys_only(cls, v: dict) -> dict:
        return samplers.check(v)


# a Russian question's words in English beside it, so the English text it asks about is in the keyword search's reach
class KeywordTranslationCfg(_Strict):
    enabled: bool = False
    model_dir: str = "datasets/models/opus-mt-ru-en-ctranslate2"
    # the keyword search reads the translation instead of the question: the Russian words matched Russian chunks only
    replaces: bool = False


# a question's jargon (постгрес, k8s) reworded to the names the docs use, ranked beside the question's own words
class KeywordAliasesCfg(_Strict):
    enabled: bool = False


KeywordQuery = Literal["and", "or"]
KEYWORD_QUERY_MODES = get_args(KeywordQuery)


# named as a run's record names them, so nothing translates between the two
class KeywordCfg(_Strict):
    query: KeywordQuery
    rank: str
    norm: int
    query_lang: QueryLanguageRule
    translation: KeywordTranslationCfg = KeywordTranslationCfg()
    aliases: KeywordAliasesCfg = KeywordAliasesCfg()
    # candidates hold a word in at most this share of chunks, ranked by every word; 0 is off, else at least 0.001
    max_term_share: float = Field(0.0, ge=0.0, le=1.0)

    # the table keeps no word rarer than its floor, so a share below it would read every word as rare
    @field_validator("max_term_share")
    @classmethod
    def _above_the_floor(cls, value: float) -> float:
        from corpus_keys import TERM_SHARE_FLOOR

        if 0 < value < TERM_SHARE_FLOOR:
            raise ValueError(f"max_term_share is 0 (off) or at least {TERM_SHARE_FLOOR}")
        return value


class RetrievalCfg(_Strict):
    distance_threshold: float
    results_limit: int
    limit_vector: int
    limit_keywords: int
    rrf_k: int
    keyword: KeywordCfg
    # "auto" asks the planner for the deepest rung still walking the index, which moves
    ef_search: int | Literal["auto"]
    # a filtered hnsw walk stops at ef_search and returns fewer rows; relaxed_order walks on until the filter is met
    filtered_scan: Literal["off", "relaxed_order"] = "off"
    # the same text under several files of one source returns once, at its best rank
    collapse_copies_in_source: bool = False


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

    # whether the last lookup fell back: the caller says so on the row, a silent `max` misled us once
    def topic_threshold_is_measured(self, language: str | None) -> bool:
        if not isinstance(self.topic_threshold, dict) or not self.topic_threshold:
            return True
        return language in self.topic_threshold

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
    chunker: Literal["rooted", "structured"]
    max_chunk_size: int = Field(gt=0)
    ceiling_on: Literal["body", "content"] = "body"
    # off by default: it changes the cut, so it is a corpus variant of its own
    drop_boilerplate: bool = False
    # a whole section with a body shorter than this joins its file's previous chunk, or the next; 0 is off
    merge_tiny_sections_under: int = Field(default=0, ge=0)
    # a run of at least this many dot-leader lines is a table of contents or an index and is cut out; 0 is off
    contents_runs_from: int = Field(default=0, ge=0)


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


class MeasureRulesCfg(_Strict):
    tiny_share_of_ceiling: float
    boilerplate_file_share: float
    boilerplate_min_files: int
    min_breaching_chunks: int
    # a chapter of at least this many chunks with at most that many sections inside is flat: its headings were lost
    flat_min_chunks: int = Field(ge=1)
    flat_max_sections: int = Field(ge=1)
    soup_alnum_ratio: float
    prose_word_letters: int


class IngestQualityCfg(_Strict):
    # thresholds live here, not in code: they are turned by hand and land in every report
    measure: MeasureRulesCfg
    hard_gates: MetricGatesCfg
    soft_gates: MetricGatesCfg
    history_per_variant: int
    score_formula: str
    weights: MetricWeightsCfg


class RouteCfg(_Strict):
    min_layer_chars: int
    min_raster_run: int
    suspect_min_words: int
    reread_below_layer_f1: float = Field(ge=0, le=1)
    reread_settings: str
    reread_cells_slack: float = Field(ge=0, le=1)
    splice_tables: bool
    seam_window: int = Field(ge=0)
    seam_margin: float = Field(ge=0, lt=0.5)
    epub_skip: list[str]
    mono_spread: float = Field(gt=0)
    mono_faces: list[str] = Field(min_length=1)
    headings_by_number: bool
    man_page_titles: bool
    listing_callouts: bool
    mono_by_step: bool
    code_row_rules: list[Literal["run_on", "once", "numbers"]]
    outline_levels: bool
    contents_outline: bool
    html_one_title: bool
    numbered_levels: bool
    decode_entities: bool
    drop_lone_pipes: bool
    join_layer_hyphens: bool
    restore_dashes: bool
    join_broken_words: bool
    unescape_bullets: bool
    unescape_underscores: bool
    picture_addresses: bool
    formula_text: bool
    demote_caption_headings: bool
    drop_running_headings: bool
    join_split_words: bool
    drop_inherited_members: bool
    drop_repeated_code: bool
    epub_chapters: bool


# what only the stand sets: the floors that decide a file's route, not how a source is read
STAND_ONLY = ("min_layer_chars", "min_raster_run", "suspect_min_words")
# route keys that move no piece: a signal's floor and a file skipped, so the route's fingerprint leaves them out
SHAPE_NO_PIECE = ("suspect_min_words", "epub_skip")
# the knobs a source may set over the stand's route, each a field of RouteCfg
SOURCE_KNOBS = tuple(name for name in RouteCfg.model_fields if name not in STAND_ONLY)


# a source's own questions read against it after they are accepted, clamped to it and open over the corpus
class SourceGateCfg(_Strict):
    clamped_min: float = Field(ge=0, le=1)
    open_min: float = Field(ge=0, le=1)
    min_questions: int = Field(ge=1)


class RawQualityCfg(_Strict):
    output_share_min: float
    output_share_max: float
    layer_f1_min: float
    mixed_script_max: float
    layer_band_engines: list[Tool]
    bad_share: float
    auto_accept_ok: bool
    # rounds of existing knobs an agent tries on one source; past them a person approves a new knob or refuses it
    agent_knob_rounds: int = Field(ge=0)
    source_gate: SourceGateCfg


class IntakeCfg(_Strict):
    route: RouteCfg
    quality: RawQualityCfg
    engines: dict[Tool, str]
    settings: dict[Tool, str]

    # the route may send a file to any tool, so every tool has its engine and its settings
    @field_validator("engines", "settings")
    @classmethod
    def _every_tool(cls, value):
        if missing := set(Tool) - set(value):
            raise ValueError(f"no entry for {sorted(missing)}")
        return value


class IngestionCfg(_Strict):
    batch_size: int
    commit_size: int


class EngineCfg(_Strict):
    name: str
    kind: Literal["ollama", "vllm", "openai_compatible", "converter"]
    env_prefix: str
    placement: Literal["gpu", "cpu", "gpu+cpu", "remote"]
    balance_reader: str | None = None


class TokenEstimateCfg(_Strict):
    latin_divisor: float
    cyrillic_extra: float


class LlmCfg(_Strict):
    base_url: str
    roles: dict[str, RoleCfg]
    context_length: int
    repetition_penalty: float
    token_estimate: TokenEstimateCfg
    # keyed by the ollama version the penalty was measured on
    measured_repeat_penalty: dict[str, float]

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


class StatsCfg(_Strict):
    bootstrap_n: int
    alpha: float
    seed: int


class JudgeCorrelationCfg(_Strict):
    rho_with_overlap: float
    partial_gap: float
    stratum_gap: float
    min_rows: int
    code_stratum: float


class JudgeLanguageCfg(_Strict):
    control_floor: float
    moves_allowed: float


class VetoCfg(_Strict):
    quotas: dict[str, int]
    min_heading: int


class GradeCurveCfg(_Strict):
    cuts: list[float | None]


class RetrievalCompareCfg(_Strict):
    candidates: int
    depth: int
    cutoffs: list[int]
    rrf_k: int


# how many questions a source's set is drawn to, and how they spread over its chapters
class QuestionSetCfg(_Strict):
    per_source: int
    per_chapter: int
    min_pairs: int
    languages: list[Literal["en", "ru"]] = Field(min_length=1)

    # a language named twice gives each pair two rows of one text, and every pair is dropped as a repeat
    @field_validator("languages")
    @classmethod
    def _each_once(cls, languages: list[str]) -> list[str]:
        if len(set(languages)) != len(languages):
            raise ValueError("languages: each language once")
        return languages


class EvalsCfg(_Strict):
    stats: StatsCfg
    judge_correlation: JudgeCorrelationCfg
    judge_language: JudgeLanguageCfg
    veto: VetoCfg
    grade_curve: GradeCurveCfg
    retrieval_compare: RetrievalCompareCfg
    question_set: QuestionSetCfg


# a row of the coverage map: the only values a source may name as a chunk's category, whatever the domain
class CategoryCfg(_Strict):
    name: str
    group: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    # the majors kept side by side; empty is one rolling version
    versions: list[str] = []


# `unsure` aliases stay out of the search: an ordinary word (клик, квадрант) would reword questions that mean it
class AliasCfg(_Strict):
    canonical: str
    aliases: list[str]
    seen_in: list[str] = []
    unsure: list[str] = []
    related: list[str] = []

    # yaml reads 404 as a number, and an empty `unsure:` as null
    @field_validator("aliases", "unsure", "related", mode="before")
    @classmethod
    def _words(cls, value) -> list[str]:
        return [str(v).lower() for v in value or []]


class AppConfig(_Strict):
    retrieval: RetrievalCfg
    verdict: VerdictCfg
    evals: EvalsCfg
    rerank: RerankCfg
    agent: AgentCfg
    ingestion: IngestionCfg
    ingest_quality: IngestQualityCfg
    intake: IntakeCfg
    fts: FtsCfg
    corpus: CorpusCfg
    repos_dir: str
    prompts_dir: str
    engines: list[EngineCfg]
    llm: LlmCfg
    postgres: PostgresCfg
    mcp_integrations: McpIntegrationsCfg
    categories: dict[str, CategoryCfg]
    aliases: dict[str, AliasCfg] = {}

    # one alias naming two technologies leaves the search to guess which one the question meant
    @field_validator("aliases")
    @classmethod
    def _one_owner_per_alias(cls, value: dict) -> dict:
        def words(text: str) -> str:
            return " ".join(text.lower().split())

        owners: dict[str, set] = {}
        for name, entry in value.items():
            for alias in {words(a) for a in entry.aliases} | {words(entry.canonical)}:
                owners.setdefault(alias, set()).add(name)
        shared = {alias: sorted(names) for alias, names in owners.items() if len(names) > 1}
        if shared:
            raise ValueError(f"an alias names one technology, these name several: {shared}")
        return value


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


CONFIG_DIR = "config"
ROLES_FILE = os.path.join(CONFIG_DIR, "roles.yaml")


# the sections a process owns live in their own file under `config/`; the roles are read apart
def _section_files(here: str) -> list[str]:
    folder = os.path.join(here, CONFIG_DIR)
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    return [os.path.join(folder, n) for n in names if n.endswith(".yaml") and not n.startswith("roles")]


# a section named twice refuses to load
def _sections(here: str, raw: dict) -> dict:
    for file in _section_files(here):
        with open(file) as f:
            part = yaml.safe_load(f) or {}
        twice = sorted(set(part) & set(raw))
        if twice:
            raise ValueError(f"{CONFIG_DIR}/{os.path.basename(file)}: {twice} already come from another file")
        raw.update(part)
    return raw


def yaml_files(folder: str) -> list[str]:
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    return [os.path.join(folder, n) for n in names if n.endswith(".yaml")]


def beside_config(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(CONFIG_PATH)), name)


# a worked-out source's file and a format's vocabulary decide what is indexed, so they count as config
def _data_files(here: str) -> list[str]:
    return [f for name in ("sources", "formats") for f in yaml_files(os.path.join(here, name))]


# every file the loader reads, a source's data file included: the one answer to which files make the config
def loaded_files(with_overlay: bool = True, with_data: bool = True) -> list[str]:
    here = os.path.dirname(os.path.abspath(CONFIG_PATH))
    overlay = [os.path.join(here, CONFIG_OVERLAY)] if CONFIG_OVERLAY and with_overlay else []
    base = [os.path.abspath(CONFIG_PATH), *_section_files(here), os.path.join(here, ROLES_FILE), *overlay]
    return base + _data_files(here) if with_data else base


# the prompt versions a fresh database starts on belong to the stand, not to a layout: an overlay does not move them
def declared_prompts() -> dict[str, int]:
    here = os.path.dirname(os.path.abspath(CONFIG_PATH))
    declared: dict[str, int] = {}
    for name, role in _roles_of(os.path.join(here, ROLES_FILE)).items():
        for purpose, version in ((role or {}).get("prompts") or {}).items():
            if purpose in declared:
                raise ValueError(f"{ROLES_FILE}: {purpose} is named by two roles, the second is {name}")
            declared[purpose] = version
    return declared


def _load(path: str, overlay: str | None = None) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "roles" in (raw.get("llm") or {}):
        raise ValueError(f"{path}: the roles live in {ROLES_FILE}, not here")
    here = os.path.dirname(os.path.abspath(path))
    raw = _sections(here, raw)
    # the roles are a layer of their own; an overlay replaces them whole, as `config/roles.cpu.yaml` does
    raw["llm"]["roles"] = _roles_of(os.path.join(here, overlay or ROLES_FILE))
    return AppConfig(**raw)


settings = _load(CONFIG_PATH, CONFIG_OVERLAY)


# the keyword leg by the name the record uses, which is the name the config uses
KEYWORD_SWITCHES = tuple(KeywordCfg.model_fields)


# the dictionary's digest rides with the switch, so two runs on two dictionaries do not read as one setting
def keyword_switches() -> dict:
    switches = settings.retrieval.keyword.model_dump()
    if switches["aliases"]["enabled"]:
        import query_aliases

        switches["aliases"]["digest"] = query_aliases.digest()
    return switches
