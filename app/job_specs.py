"""What each job type accepts, checked at both ends of the queue."""

from enum import StrEnum
from typing import Literal

import limits
import samplers
from evals.guest_axes import MESSAGE_FORMS
from models.registry import MAX_MODEL_NAME, MODEL_NAME_RE, Pipeline, Role
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from use_cases import agent_policy
from use_cases.agent_policy import GONE, FallbackPolicy, GateSignal, Orchestrator
from use_cases.index import VARIANT_RE

# a retired arm dies on every question, so the queue refuses it as the REST door already did
Runnable = StrEnum("Runnable", {o.name: o.value for o in Orchestrator if o not in GONE})


class Spec(BaseModel):
    # a misspelt option used to be accepted and ignored, which is how a run silently kept a default
    model_config = ConfigDict(extra="forbid")


class EvalRunFields(Spec):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    set_name: str | None = None
    question_ids: limits.QuestionIds = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
    rerank: bool | None = None
    pipeline: Pipeline = Pipeline.single_shot
    language: Literal["ru", "en"] | None = None
    k: int | None = Field(default=None, ge=1, le=limits.MAX_K)
    max_hops: int | None = Field(default=None, ge=1, le=agent_policy.MAX_HOPS)
    model: str | None = Field(
        default=None, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern
    )
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

    @field_validator("generation_sampler")
    @classmethod
    def _sampler_keys(cls, value):
        return samplers.check(value) if value else value


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
    form: Literal["per_chunk", "whole_text"] = "per_chunk"
    top: int = Field(default=5, ge=1, le=20)
    # a slice for a probe; a declared arm draws `sample` by `seed`, as the guest axes do
    limit: int | None = Field(default=None, ge=1, le=2000)
    sample: int | None = Field(default=None, ge=1, le=2000)
    seed: int = 0
    # the floor is taken twice in one residency, and the second pass must not repeat the first order
    shuffle: int | None = Field(default=None, ge=0, le=10_000)
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
    model: str | None = Field(
        default=None, min_length=1, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern
    )
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
        from models.registry import refuse_unknown_registry

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


class BuildVetoSet(Spec):
    seed: str | int | None = None
    set_name: str | None = None
    variants: list | None = None
    cut_from: str | None = Field(default=None, pattern=VARIANT_RE.pattern)
    quotas: dict | None = None


class IndexData(Spec):
    source: str | None = None
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


class BuildVectorIndex(Spec):
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)


class EmbedQuestions(Spec):
    pass


SPECS: dict[str, type[Spec]] = {
    "paraphrase_questions": ParaphraseQuestions,
    "build_veto_set": BuildVetoSet,
    "index_data": IndexData,
    "build_vector_index": BuildVectorIndex,
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

LANES = {"pull_llm_model": "io", "delete_llm_model": "io", "check_mcp_health": "io"}

# lower first; judging waits for runs, and the API's `hand_card` overtakes what waits
PRIORITY = {"hand_card": -2, "judge_answers": 10, "judge_guest_axes": 10, "judge_language": 10}

# a flow of runs must not hold the judge back forever: a job this old goes before all but a handover
STARVED_AFTER_MINUTES = 30


# the roles a type answers with; `hand_card` names its engine and model in the options instead
LOADS: dict[str, tuple[Role, ...]] = {
    "paraphrase_questions": (Role.paraphrasing,),
    "build_veto_set": (Role.paraphrasing,),
    "index_data": (Role.embedding,),
    "embed_questions": (Role.embedding,),
    "build_vector_index": (),
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


# the stand's own bookkeeping on a job: `_job_id` carries a prefix and these two never did
WORKER_KEYS = ("deferred_seconds", "attempts")

# the options by which a job names a model beside its roles' own
MODEL_OVERRIDES = {"generation": "model", "judging": "judge_model", "ragas": "guest_model"}


class Refused(ValueError):
    pass


# its own type: a handler on pydantic's base read a bug in our own model as the caller's mistake
def check(job_type: str, options: dict | None, *, from_the_worker: bool = False) -> None:
    given = {k: v for k, v in (options or {}).items() if not k.startswith("_")}
    theirs = sorted(set(given) & set(WORKER_KEYS))
    # a caller sending `attempts` past the cap buys a job that never retries, and says nothing
    if theirs and not from_the_worker:
        raise Refused(f"{theirs[0]}: the stand writes this on a retry, a caller does not")
    spec = SPECS.get(job_type)
    if spec is None:
        return
    asked = {k: v for k, v in given.items() if k not in WORKER_KEYS}
    try:
        spec.model_validate(asked)
    except ValidationError as bad:
        first = bad.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "options"
        raise Refused(f"{where}: {first['msg']}") from bad
