"""What each job type accepts, checked at both ends of the queue.

A door cannot be the only place a precondition lives: `eval_run` has four enqueuers and
`judge_answers` five, and a typed door defends exactly one of them. These models are checked when a
job is enqueued, whichever door or script enqueues it, and again when the worker takes it, so a row
written straight into the table cannot walk past them either.

Unknown keys are allowed for now: several callers build their options dictionary dynamically, and
forbidding extras before those are read one by one would refuse work that is correct today.
"""

from enum import StrEnum
from typing import Literal

import limits
from models.registry import MAX_MODEL_NAME, MODEL_NAME_RE, Pipeline
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
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
    question_ids: list[int] | None = Field(default=None, max_length=limits.MAX_QUESTION_IDS)
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
    judge_model: str | None = None
    judge_prompts: dict | None = None
    control_axes: list | tuple | None = None
    control_sample: int | None = None
    control_seed: int | None = None
    experiment_id: int | None = None

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


class JudgeLanguage(Spec):
    run_name: str = Field(min_length=1, max_length=limits.MAX_RUN_NAME)
    rows: int = Field(default=40, ge=1, le=limits.MAX_GUEST_ROWS)
    # a declared cut is named: `rows` takes the first of the pool, which is not a group
    log_ids: list[int] | None = Field(default=None, max_length=limits.MAX_GUEST_ROWS)


class CompareRetrieval(Spec):
    experiment_id: int


class AnalyzeSource(Spec):
    source: str = Field(min_length=1)
    variant: str | None = Field(default=None, pattern=VARIANT_RE.pattern)
    mode: str | None = None


class CheckMcpHealth(Spec):
    integration_id: int


class ModelByName(Spec):
    name: str = Field(min_length=1, max_length=MAX_MODEL_NAME, pattern=MODEL_NAME_RE.pattern)

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
    "analyze_source": AnalyzeSource,
    "check_mcp_health": CheckMcpHealth,
    "pull_llm_model": ModelByName,
    "delete_llm_model": ModelByName,
}

LANES = {"pull_llm_model": "io", "delete_llm_model": "io", "check_mcp_health": "io"}

# the safe way round: an unclassified type evicts. `judge_guest_axes` left when relevancy arrived
KEEPS_THE_JUDGE = ("judge_answers", "check_mcp_health", "build_vector_index")


def disturbs_the_judge(job_type: str) -> bool:
    return job_type not in KEEPS_THE_JUDGE


# a renamed type would leave a dead entry here and quietly start evicting the judge on paper
if not set(KEEPS_THE_JUDGE) <= set(SPECS):
    raise RuntimeError(f"no such job type: {sorted(set(KEEPS_THE_JUDGE) - set(SPECS))}")


# a type that takes whatever it is given; the universal door made the empty list the safe state
FREE = ()


def lane(job_type: str) -> str:
    return LANES.get(job_type, "default")


# the stand's own bookkeeping on a job: `_job_id` carries a prefix and this one never did
WORKER_KEYS = ("deferred_seconds",)


class Refused(ValueError):
    pass


# its own type: a handler on pydantic's base read a bug in our own model as the caller's mistake
def check(job_type: str, options: dict | None) -> None:
    spec = SPECS.get(job_type)
    if spec is None:
        return
    asked = {
        k: v for k, v in (options or {}).items()
        if not k.startswith("_") and k not in WORKER_KEYS
    }
    try:
        spec.model_validate(asked)
    except ValidationError as bad:
        first = bad.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "options"
        raise Refused(f"{where}: {first['msg']}") from bad
