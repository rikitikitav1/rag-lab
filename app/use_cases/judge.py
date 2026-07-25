import json
from dataclasses import dataclass, field
from enum import StrEnum

import llm
import prompt_repo
from models.registry import Purpose
from timing_wrappers import measure_elapsed


class FaithfulVerdictOption(StrEnum):
    FAITHFUL = "faithful"
    PARTIALLY = "partially"
    UNFAITHFUL = "unfaithful"


class RelevantVerdictOption(StrEnum):
    RELEVANT = "relevant"
    PARTIALLY = "partially"
    IRRELEVANT = "irrelevant"


VERDICTS = {
    "faithful": ["faithful", "partially", "unfaithful"],
    "relevance": ["relevant", "partially", "irrelevant"],
}


@dataclass
class Verdict:
    reason: str
    verdict: FaithfulVerdictOption | RelevantVerdictOption
    elapsed: float = 0.0
    model: str = field(default_factory=lambda: llm.resolve_name("judging"))

    def __str__(self) -> str:
        return (
            f"verdict: {self.verdict}, reason: {self.reason}, "
            f"model: {self.model}, elapsed: {self.elapsed}"
        )


def faithful_verdict(question, answer, context) -> Verdict:
    system, user = faithful_prompts(question, answer, context)
    return judge(
        system,
        user,
        response_schema("faithful"),
        FaithfulVerdictOption,
    )


def relevance_verdict(question, answer) -> Verdict:
    system, user = relevance_prompts(question, answer)
    return judge(
        system,
        user,
        response_schema("relevance"),
        RelevantVerdictOption,
    )


@measure_elapsed
def judge(system_prompt, user_prompt, schema, verdict_class) -> Verdict:
    completion = llm.ask(
        system=system_prompt, user=user_prompt, role="judging", schema=schema
    )
    parsed = json.loads(completion.text)

    return Verdict(
        verdict=verdict_class(parsed["verdict"]),
        reason=parsed["reason"],
    )


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


def response_schema(check_type) -> dict:
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "verdict": {"type": "string", "enum": VERDICTS.get(check_type)},
        },
        "required": ["reason", "verdict"],
    }
