from models.registry import Role
from use_cases.model_acceptance import complaints

_WITH_SYSTEM_AND_TOOLS = "{{ .System }} {{ .Tools }} {{ .Prompt }}"


def test_a_generator_without_tool_calling_is_refused_before_it_costs_a_run():
    # the agent sends `tools` on every call, and ollama refuses them for a model without it
    said = complaints(
        Role.generation,
        {"capabilities": ["completion", "vision"], "template": _WITH_SYSTEM_AND_TOOLS},
    )
    assert said and "tools" in said[0]

    fine = {"capabilities": ["completion", "tools"], "template": _WITH_SYSTEM_AND_TOOLS}
    assert complaints(Role.generation, fine) == []


def test_a_thinking_model_is_refused_for_judging():
    # through the chat API a thinking model's answer never reaches `content`
    said = complaints(
        Role.judging,
        {"capabilities": ["completion", "thinking"], "template": _WITH_SYSTEM_AND_TOOLS},
    )
    assert said and "thinking" in said[0]


def test_a_template_that_renders_no_system_prompt_is_refused():
    said = complaints(
        Role.judging, {"capabilities": ["completion"], "template": "{{ .Prompt }}"}
    )
    assert said and "system prompt" in said[0]


def test_a_server_that_says_nothing_complains_about_nothing():
    # an empty answer is not evidence of a defect, and a probe must not block a role
    assert complaints(Role.generation, {}) == []
    assert complaints(Role.embedding, {"capabilities": ["completion"], "template": "x"}) == []
