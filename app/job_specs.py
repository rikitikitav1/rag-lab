"""What each job type accepts, checked at both ends of the queue."""

import re
from enum import StrEnum
from typing import Annotated, Literal

import limits
import samplers
from corpus_keys import VARIANT_RE
from models.registry import MAX_MODEL_NAME, MODEL_NAME_RE, Pipeline, Role, refuse_unknown_registry
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from search_scope import CATEGORY_RE, MAX_SOURCES, VERSION_RE, Scope, refuse_malformed_scope
from sources.declaration import IntakeOverride
from tool_names import SETTINGS_NAME, settings_refusal
from vocabulary import GONE, MAX_HOPS, MESSAGE_FORMS, SOURCE_NAME, FallbackPolicy, GateSignal, Language, Orchestrator

# a generated set names its files and its reports, so its name is one a path can carry as it is
SET_NAME = r"^[\w.-]+$"
SetName = Annotated[str, Field(min_length=1, max_length=limits.MAX_SET_NAME, pattern=SET_NAME)]
# the only folder a graded pass reads: a path of its own would let a job open any file
FROZEN_POOL_RE = re.compile(r"(/app/)?datasets/candidates/[\w.-]+\.json")

# a retired arm dies on every question, so the queue refuses it as the REST door already did
Runnable = StrEnum("Runnable", {o.name: o.value for o in Orchestrator if o not in GONE})


class Spec(BaseModel):
    # a misspelt option used to be accepted and ignored, which is how a run silently kept a default
    model_config = ConfigDict(extra="forbid")


# a smoke asks nothing of the record; a closing run is the one whose number is quoted
class Purpose(StrEnum):
    smoke = "smoke"
    probe = "probe"
    closing = "closing"


class EvalRunFields(Spec):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    set_name: str | None = None
    question_ids: limits.QuestionIds = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    k: int | None = Field(default=None, ge=1, le=limits.MAX_K)
    max_hops: int | None = Field(default=None, ge=1, le=MAX_HOPS)
    model: str | None = Field(default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)
    fallback_policy: FallbackPolicy | None = None
    gate_signal: GateSignal | None = None
    weak_distance: float | None = Field(default=None, ge=0, le=2)
    topic_threshold: float | None = Field(default=None, ge=0, le=2)
    orchestrator: Runnable | None = None
    allow_cpu: bool = False
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)
    # llama3.1 renders tool schemas only in the last user message, and a tool answer buries them
    restate_tools: bool = False
    # the grader filters chunks before the generator sees them; off, every run is what it was
    grade_chunks: bool = False
    # answer only what a stopped run left unanswered, on the options it ran with
    resume: bool = False
    # the generator's sampler in this run alone: the judge shares a vLLM model with it and keeps its own
    generation_sampler: dict | None = None
    # off for a run read by a rule and not by a score: retrieval deltas, a string match, a canary
    judge: bool = True
    # what this run is for; the default is the cheap case, and only the closing one owes a promise
    purpose: Purpose = Purpose.smoke
    # the preregistration this run was made under, by name
    prereg: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)
    # the search's scope, as the chat doors take it: a book's questions asked of that book alone
    category: str | None = Field(default=None, pattern=CATEGORY_RE.pattern)
    sources: list[str] | None = Field(default=None, max_length=MAX_SOURCES)
    version: str | None = Field(default=None, pattern=VERSION_RE.pattern)

    @field_validator("generation_sampler")
    @classmethod
    def _sampler_keys(cls, value):
        return samplers.check(value) if value else value

    def scope(self) -> Scope:
        return Scope.of(self.category, self.sources, self.version)

    # the agent searches with its own queries and no filter; a scope it would drop is refused, as at the MCP door
    @model_validator(mode="after")
    def _a_scope_the_pipeline_can_read(self):
        scope = self.scope()
        if scope.narrowed and self.pipeline == Pipeline.agent:
            raise ValueError("a category, source or version scope is only supported with pipeline=single_shot")
        refuse_malformed_scope(scope)
        return self

    # the gate lives here and not on a route, so the REST door and the queue get it from one place
    @model_validator(mode="after")
    def _a_closing_run_names_its_promise(self):
        if self.purpose is Purpose.closing and not self.prereg:
            raise ValueError(
                "a closing run names the preregistration it was made under:"
                " pass `prereg`, or run it as `purpose: smoke`"
            )
        return self


# what the queue accepts is what a door may offer plus what the stand attaches to its own jobs
class EvalRun(EvalRunFields):
    experiment_id: int | None = None

    # the hole `__noop__` walked through: neither filter means every question there is
    @model_validator(mode="after")
    def _needs_a_target(self):
        if not self.set_name and not self.question_ids:
            raise ValueError("a run needs a target: name a set or the question ids, not neither")
        return self


class JudgeAnswers(Spec):
    run_name: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)
    log_ids: list[int] | None = None
    # a counter, not a flag: the sweep carries how many times it has swept, and it reaches three
    sweep: int | bool | None = None
    judge_width: int | None = Field(default=None, ge=1, le=limits.MAX_RUNS)
    judge_model: str | None = Field(default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)
    judge_prompts: dict | None = None
    control_axes: list | tuple | None = None
    control_sample: int | None = None
    control_seed: int | None = None
    experiment_id: int | None = None
    # the batch the chat's answers gather in: it waits so the judge wakes once, not once per question
    live: bool | None = None

    # no target judges every unjudged row there is, and only the sweep may mean that
    @model_validator(mode="after")
    def _names_what_it_judges(self):
        if not self.run_name and not self.log_ids and not self.sweep:
            raise ValueError("judging needs a target: a run, the log ids, or the sweep flag")
        return self


class JudgeGuestAxes(Spec):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    judge_width: int | None = Field(default=None, ge=1, le=limits.MAX_RUNS)
    log_ids: list[int] | None = None
    sample: int | None = Field(default=None, ge=1, le=limits.MAX_GUEST_ROWS)
    seed: int | None = None
    # `user_only` by default; the empty system beside the prompt is the old ruler, kept for a bridge
    messages: Literal[*MESSAGE_FORMS] = MESSAGE_FORMS[0]
    # the guest's own bench, as `judge_model` is the judge's: a copy scored by another model, no reseat
    guest_model: str | None = Field(default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)


class JudgeLanguage(Spec):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    rows: int = Field(default=40, ge=1, le=limits.MAX_GUEST_ROWS)
    # a declared cut is named: `rows` takes the first of the pool, which is not a group
    log_ids: list[int] | None = Field(default=None, max_length=limits.MAX_GUEST_ROWS)


class CompareRetrieval(Spec):
    experiment_id: int


class GradeCandidates(Spec):
    # the frozen pool this grades: a run that names no file would grade whatever is on disk today
    candidates: str = Field(min_length=1, max_length=200)

    @field_validator("candidates")
    @classmethod
    def _under_the_frozen_pools(cls, value: str) -> str:
        # a job reads a file the worker can reach, so the name is a pool of ours, not any path
        if not FROZEN_POOL_RE.fullmatch(value):
            raise ValueError("candidates names a frozen pool, like /app/datasets/candidates/<name>.json")
        return value

    form: Literal["per_chunk", "whole_text"] = "per_chunk"
    top: int = Field(default=5, ge=1, le=20)
    # a slice for a probe; a declared arm draws `sample` by `seed`, as the guest axes do
    limit: int | None = Field(default=None, ge=1, le=2000)
    sample: int | None = Field(default=None, ge=1, le=2000)
    seed: int = 0
    # the floor is taken twice in one residency, and the second pass must not repeat the first order
    shuffle: int | None = Field(default=None, ge=0, le=10_000)
    question_ids: limits.QuestionIds = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
    # an arm names the prompt it measures, so nothing has to be activated to be read
    prompt_version: int | None = Field(default=None, ge=1, le=1000)
    name: str | None = Field(default=None, max_length=limits.MAX_RUN_NAME)


class AnalyzeSource(Spec):
    source: str = Field(min_length=1)
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)
    mode: str | None = None


class CheckMcpHealth(Spec):
    integration_id: int


class HandCard(Spec):
    engine_id: int = Field(ge=1)
    # the API queues this (the chat, `/load`, a role seat): who asked, for the reader, not the turn
    asked_by: str | None = Field(default=None, max_length=64)
    # for ollama the model to load once the card is free; vLLM serves one model and needs no name
    model: str | None = Field(default=None, min_length=1, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)
    # the role door on an asleep vLLM: probe the woken server, then seat the role or fail with why
    seat: Literal["generation"] | None = None
    # the model the role held when the door asked; the seat is refused if another has come since
    seat_over: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _seats_a_named_model(self):
        if self.seat and not self.model:
            raise ValueError("seat names the model it seats")
        return self


class ModelByName(Spec):
    name: str = Field(min_length=1, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)
    # absent on jobs queued before engines existed, and then the one engine there is answers
    engine_id: int | None = Field(default=None, ge=1)

    # the typed door refused a three-segment name pointing anywhere else; the universal one did not
    @model_validator(mode="after")
    def _known_registry(self):
        refuse_unknown_registry(self.name)
        return self


class ParaphraseQuestions(Spec):
    limit: int | None = Field(default=None, ge=1)
    source: str | None = None
    set_name: str | None = None
    seed: str | int | None = None
    per_source: int | None = Field(default=None, ge=1)
    grow: bool | None = None
    originals: list | None = None


# a source's question pairs written from its sections; a probe caps the pairs it asks for
class GenerateQuestions(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    set_name: SetName
    max_pairs: int | None = Field(default=None, ge=1)
    # a smoke: go on section by section until this many pairs are kept
    kept_at_least: int | None = Field(default=None, ge=1)
    # the languages a pair is asked in, over the set's configured ones; the stand reads two
    languages: list[Literal["en", "ru"]] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _one_way_to_stop(self):
        if self.max_pairs and self.kept_at_least:
            raise ValueError("a run stops at max_pairs asked or at kept_at_least kept, not both")
        if self.languages and len(set(self.languages)) != len(self.languages):
            raise ValueError("languages: each language once")
        return self


# a set's candidate pairs of one source read from their sections; a probe caps the pairs it reads
class AcceptQuestions(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    set_name: SetName
    max_pairs: int | None = Field(default=None, ge=1)
    # a pair read once waits for the judge; asking the reader again is a choice, not a rerun's default
    again: bool = False
    # a pass that only reports, so two passes over one set can be compared before either settles it
    settle: bool = True
    # settled pairs too, read and settled again: a reader or a reading changed since they were refused
    every: bool = False


# a set's undecided pairs of one source judged on their evidence; a probe caps the pairs it reads
class JudgeQuestions(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    set_name: SetName
    max_pairs: int | None = Field(default=None, ge=1)
    settle: bool = True
    every: bool = False
    # a second judge over the role's own: the pairs read by another model, the role left seated
    model: str | None = Field(default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)


# a generation's report read again by today's checks; the name only, the folder is the stand's
class ReparseQuestions(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    set_name: SetName
    report: str = Field(pattern=r"^question_set_[\w.-]+\.json$", max_length=300)


# a set's rows given their anchors by today's rule, the source's sections read once
class AnchorQuestions(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    set_name: SetName


class ReanchorQuestions(Spec):
    set_name: SetName
    variant: str | None = None
    dry: bool = False


# a generated set to its file beside the sources, and back into the base on a later intake
class SaveQuestions(Spec):
    set_name: SetName


class LoadQuestions(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    set_name: SetName


class BuildVetoSet(Spec):
    seed: str | int | None = None
    set_name: str | None = None
    variants: list | None = None
    cut_from: str | None = Field(default=None, pattern=VARIANT_RE.pattern)
    quotas: dict | None = None


class IndexData(Spec):
    source: str | None = None
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


class ConvertSource(Spec):
    # a settings file under converters/<tool>/settings/, written before the run and hashed into its record
    settings: str = Field(pattern=f"^{SETTINGS_NAME.pattern}$")
    language: Language
    # paths under the gold's files; a path of its own would let a job read any file
    inputs: list[str] = Field(min_length=1, max_length=5000)
    out: str = Field(pattern=r"^[\w.-]{1,80}$")
    # read as the corpus reads a file (route, seams, reread, join), not by the one settings file alone
    intake: bool = False
    # where the inputs lie: the gold's files, or the store of sources, read and never written
    root: Literal["gold", "inbox"] = "gold"
    # an input read over a page range of its own file, first and last inclusive, as onboarding hands a piece
    pages: dict[str, tuple[int, int]] | None = None
    # a run's own piece size in place of the settings', stamped in its record
    pages_per_chunk: int | None = Field(default=None, ge=1)
    # an input's source by name, so it is read with that source's own intake knobs
    sources: dict[str, str] | None = None
    # intake knobs of the run itself, over every input's own: an arm is its settings plus these
    knobs: dict | None = None

    @field_validator("inputs")
    @classmethod
    def _inside_the_gold(cls, inputs):
        for path in inputs:
            if path.startswith("/") or ".." in path.split("/"):
                raise ValueError(f"{path}: a path relative to the gold's files, without ..")
        return inputs

    @model_validator(mode="after")
    def _ranges_name_inputs(self):
        as_corpus = (self.pages, self.root != "gold", self.pages_per_chunk, self.sources, self.knobs)
        if not self.intake and any(as_corpus):
            raise ValueError("pages, root and pages_per_chunk read a file as the corpus does, so they need intake")
        if self.knobs is not None:
            IntakeOverride(**self.knobs)
        for path, (first, last) in (self.pages or {}).items():
            if path not in self.inputs or not 1 <= first <= last:
                raise ValueError(f"pages {path}: an input of the job with 1 <= first <= last")
        return self


class OnboardSource(Spec):
    # a declared source by name; the route picks each file's engine, these name the settings each engine runs with
    source: str = Field(pattern=SOURCE_NAME)
    settings: dict[str, str] | None = None
    # every piece read by its tool now, for a measure of the tool: no kept reading, no kept piece, stamped in the record
    fresh: bool = False

    @field_validator("settings")
    @classmethod
    def _settings_files(cls, settings):
        for tool, name in (settings or {}).items():
            if refusal := settings_refusal(tool, name):
                raise ValueError(refusal)
        return settings


class ProbeIntake(Spec):
    source: str = Field(pattern=SOURCE_NAME)
    pages: tuple[int, int]
    knobs: dict
    # a source of several PDFs names the one to read
    file: str | None = None

    @field_validator("pages")
    @classmethod
    def _pages(cls, pages):
        if not 1 <= pages[0] <= pages[1]:
            raise ValueError("pages are [first, last], counted from 1")
        return pages

    @field_validator("knobs")
    @classmethod
    def _knobs(cls, knobs):
        IntakeOverride.model_validate(knobs)
        return knobs


class BuildVectorIndex(Spec):
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


# the share of chunks holding each word, which the keyword search's rare cut reads
class CountTerms(Spec):
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


class EmbedQuestions(Spec):
    pass


SPECS: dict[str, type[Spec]] = {
    "paraphrase_questions": ParaphraseQuestions,
    "generate_questions": GenerateQuestions,
    "accept_questions": AcceptQuestions,
    "judge_questions": JudgeQuestions,
    "reparse_questions": ReparseQuestions,
    "anchor_questions": AnchorQuestions,
    "reanchor_questions": ReanchorQuestions,
    "save_questions": SaveQuestions,
    "load_questions": LoadQuestions,
    "build_veto_set": BuildVetoSet,
    "index_data": IndexData,
    "convert_source": ConvertSource,
    "onboard_source": OnboardSource,
    "probe_intake": ProbeIntake,
    "build_vector_index": BuildVectorIndex,
    "count_terms": CountTerms,
    "embed_questions": EmbedQuestions,
    "eval_run": EvalRun,
    "judge_answers": JudgeAnswers,
    "judge_guest_axes": JudgeGuestAxes,
    "judge_language": JudgeLanguage,
    "compare_retrieval": CompareRetrieval,
    "grade_candidates": GradeCandidates,
    "analyze_source": AnalyzeSource,
    "check_mcp_health": CheckMcpHealth,
    "pull_llm_model": ModelByName,
    "hand_card": HandCard,
    "delete_llm_model": ModelByName,
}

# the question writer calls a cloud and holds no card, so it runs beside a converter
LANES = {"pull_llm_model": "io", "delete_llm_model": "io", "check_mcp_health": "io", "generate_questions": "io"}

# lower first; judging waits for runs, and the API's `hand_card` overtakes what waits
PRIORITY = {"hand_card": -2, "judge_answers": 10, "judge_guest_axes": 10, "judge_language": 10}

# a flow of runs must not hold the judge back forever: a job this old goes before all but a handover
STARVED_AFTER_MINUTES = 30


# the roles a type answers with; `hand_card` names its engine and model in the options instead
LOADS: dict[str, tuple[Role, ...]] = {
    "paraphrase_questions": (Role.paraphrasing,),
    "generate_questions": (Role.questioning,),
    "accept_questions": (Role.accepting,),
    "judge_questions": (Role.judging,),
    # no model: the replies are the ones the generator gave
    "reparse_questions": (),
    "anchor_questions": (),
    "reanchor_questions": (),
    "save_questions": (),
    "load_questions": (),
    "build_veto_set": (Role.paraphrasing,),
    "index_data": (Role.embedding,),
    # the converter is an engine, not a role: the handler takes the card for it
    "convert_source": (),
    # the converters are engines, not roles: the handler takes the card for each
    "onboard_source": (),
    "probe_intake": (),
    "embed_questions": (Role.embedding,),
    "build_vector_index": (),
    "count_terms": (),
    "analyze_source": (),
    "eval_run": (Role.generation, Role.embedding, Role.reranking),
    "compare_retrieval": (Role.reranking,),
    "grade_candidates": (Role.grading,),
    "judge_answers": (Role.judging,),
    "judge_guest_axes": (Role.ragas, Role.ragas_embedding),
    "judge_language": (Role.judging, Role.generation),
    "check_mcp_health": (),
    "pull_llm_model": (),
    "hand_card": (),
    "delete_llm_model": (),
}

# a type left out would read as loading nothing and never evict the judge
if set(LOADS) != set(SPECS):
    raise RuntimeError(f"roles not declared for: {sorted(set(LOADS) ^ set(SPECS))}")


# a type that takes whatever it is given; the universal door made the empty list the safe state
FREE = ()


def lane(job_type: str) -> str:
    return LANES.get(job_type, "default")


# the stand's own bookkeeping on a job: `_job_id` carries a prefix and these never did
WORKER_KEYS = ("deferred_seconds", "attempts", "reclaims", "waiting_because")

# the options by which a job names a model beside its roles' own
MODEL_OVERRIDES = {"generation": "model", "judging": "judge_model", "ragas": "guest_model"}


class Refused(ValueError):
    pass


# its own type: a handler on pydantic's base read a bug in our own model as the caller's mistake
def check(job_type: str, options: dict | None, *, from_the_worker: bool = False) -> BaseModel | None:
    given = {k: v for k, v in (options or {}).items() if not k.startswith("_")}
    theirs = sorted(set(given) & set(WORKER_KEYS))
    # a caller sending `attempts` past the cap buys a job that never retries, and says nothing
    if theirs and not from_the_worker:
        raise Refused(f"{theirs[0]}: the stand writes this on a retry, a caller does not")
    spec = SPECS.get(job_type)
    if spec is None:
        return None
    asked = {k: v for k, v in given.items() if k not in WORKER_KEYS}
    try:
        checked = spec.model_validate(asked)
    except ValidationError as bad:
        first = bad.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "options"
        raise Refused(f"{where}: {first['msg']}") from bad
    return checked
