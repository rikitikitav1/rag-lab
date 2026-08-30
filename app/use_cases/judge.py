import json
from dataclasses import dataclass, field

import llm
import prompt_repo
from models.registry import Purpose
from timing_wrappers import measure_elapsed

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
    # which prompt scored this: two verdicts on the same answer differ by model or by
    # prompt, and without both in the row the difference is not in the record
    purpose: Purpose | None = None
    prompt_version: int | None = None

    def __str__(self) -> str:
        return f"score: {self.score}, reason: {self.reason}, model: {self.model}, elapsed: {self.elapsed}"


# which judge scores, named rather than read from whatever is active right now: an arm that
# swapped the active prompt or the `judging` role would change the stand under every other
# reader of it, a live answer being judged in the next thread included
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
    parsed = json.loads(completion.text)
    return Verdict(
        score=int(parsed["score"]),
        reason=parsed["reason"],
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
