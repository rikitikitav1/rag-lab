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

    def __str__(self) -> str:
        return f"score: {self.score}, reason: {self.reason}, model: {self.model}, elapsed: {self.elapsed}"


def faithful_verdict(question, answer, context) -> Verdict:
    system, user = faithful_prompts(question, answer, context)
    return judge(system, user)


def relevance_verdict(question, answer) -> Verdict:
    system, user = relevance_prompts(question, answer)
    return judge(system, user)


def completeness_verdict(question, answer, reference) -> Verdict:
    system, user = completeness_prompts(question, answer, reference)
    return judge(system, user)


@measure_elapsed
def judge(system_prompt, user_prompt) -> Verdict:
    completion = llm.ask(
        system=system_prompt, user=user_prompt, role="judging", schema=SCORE_SCHEMA
    )
    parsed = json.loads(completion.text)
    return Verdict(score=int(parsed["score"]), reason=parsed["reason"])


def faithful_prompts(question, response, context) -> tuple[str, str]:
    return (
        prompt_repo.active_template(Purpose.judge_faithfulness),
        "\n\n---\n\n".join(
            [
                f"QUESTION:\n{question}",
                f"CONTEXT:{context}",
                f"RESPONSE:{response}\n---",
            ]
        ),
    )


def relevance_prompts(question, answer) -> tuple[str, str]:
    return (
        prompt_repo.active_template(Purpose.judge_relevance),
        "\n\n---\n\n".join(
            [
                f"QUESTION:\n{question}",
                f"ANSWER:{answer}\n---",
            ]
        ),
    )


def completeness_prompts(question, answer, reference) -> tuple[str, str]:
    return (
        prompt_repo.active_template(Purpose.judge_completeness),
        "\n\n---\n\n".join(
            [
                f"QUESTION:\n{question}",
                f"REFERENCE:\n{reference}",
                f"ANSWER:\n{answer}\n---",
            ]
        ),
    )
