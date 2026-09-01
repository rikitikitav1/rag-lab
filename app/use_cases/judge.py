import json
from dataclasses import dataclass, field

import llm
import prompt_repo
from models.registry import Purpose
from pydantic import BaseModel, Field, ValidationError
from timing_wrappers import measure_elapsed


class Score(BaseModel):
    reason: str
    score: int = Field(ge=0, le=10)


def _objects(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    found = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(text[start:end])
    return found


# the last object that validates: the reply may show the format or nest another
def _verdict_of(text: str | None) -> "Score":
    found = _objects(text or "")
    if not found:
        raise ValueError(f"the judge answered with no JSON object: {(text or '')[:160]!r}")
    failure = None
    for candidate in reversed(found):
        try:
            return Score.model_validate_json(candidate)
        except ValidationError as e:
            failure = e
    raise ValueError(f"the judge answered {found[0][:160]!r}: {failure.errors()[0]}") from failure


SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["reason", "score"],
}


@dataclass
class Verdict:
    reason: str
    score: int
    elapsed: float = 0.0
    model: str = field(default_factory=lambda: llm.resolve_name("judging"))
    # two verdicts differ by model or by prompt, and without both the difference is unrecorded
    purpose: Purpose | None = None
    prompt_version: int | None = None

    def __str__(self) -> str:
        return f"score: {self.score}, reason: {self.reason}, model: {self.model}, elapsed: {self.elapsed}"


# named rather than active: swapping the active prompt changes the stand under everyone
@dataclass(frozen=True)
class Bench:
    model: str | None = None
    versions: dict[Purpose, int] | None = None

    def template(self, purpose: Purpose) -> tuple[str, int]:
        pinned = (self.versions or {}).get(purpose)
        if pinned is None:
            return prompt_repo.active(purpose)
        return prompt_repo.template_of(purpose, pinned), pinned


ACTIVE = Bench()


def faithful_verdict(question, answer, context, bench: Bench = ACTIVE) -> Verdict:
    system, version = bench.template(Purpose.judge_faithfulness)
    return judge(system, faithful_user(question, answer, context),
                 Purpose.judge_faithfulness, version, bench.model)


def relevance_verdict(question, answer, bench: Bench = ACTIVE) -> Verdict:
    system, version = bench.template(Purpose.judge_relevance)
    return judge(system, relevance_user(question, answer),
                 Purpose.judge_relevance, version, bench.model)


def completeness_verdict(question, answer, reference, bench: Bench = ACTIVE) -> Verdict:
    system, version = bench.template(Purpose.judge_completeness)
    return judge(system, completeness_user(question, answer, reference),
                 Purpose.judge_completeness, version, bench.model)


@measure_elapsed
def judge(system_prompt, user_prompt, purpose=None, prompt_version=None, model=None) -> Verdict:
    completion = llm.ask(
        system=system_prompt, user=user_prompt, role="judging",
        schema=SCORE_SCHEMA, model=model,
    )
    parsed = _verdict_of(completion.text)
    return Verdict(
        score=parsed.score,
        reason=parsed.reason,
        model=model or llm.resolve_name("judging"),
        purpose=purpose,
        prompt_version=prompt_version,
    )


def faithful_user(question, response, context) -> str:
    return "\n\n---\n\n".join(
        [
            f"QUESTION:\n{question}",
            f"CONTEXT:{context}",
            f"RESPONSE:{response}\n---",
        ]
    )


def relevance_user(question, answer) -> str:
    return "\n\n---\n\n".join(
        [
            f"QUESTION:\n{question}",
            f"ANSWER:{answer}\n---",
        ]
    )


def completeness_user(question, answer, reference) -> str:
    return "\n\n---\n\n".join(
        [
            f"QUESTION:\n{question}",
            f"REFERENCE:\n{reference}",
            f"ANSWER:\n{answer}\n---",
        ]
    )
