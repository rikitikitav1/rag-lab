import llm
import logging_setup
from models.registry import Role

# what each role needs of a model, in the terms the server answers in
REQUIRED_CAPABILITY = {Role.generation: "tools"}
REFUSED_CAPABILITY = {Role.judging: "thinking", Role.paraphrasing: "thinking"}
NEEDS_SYSTEM = (Role.generation, Role.judging, Role.paraphrasing)

log = logging_setup.get_logger(__name__)


def complaints(role: Role, shown: dict) -> list[str]:
    capabilities = shown.get("capabilities") or []
    template = shown.get("template") or ""
    out = []

    # guarded on the server having said something: an empty answer refuses every new server
    needed = REQUIRED_CAPABILITY.get(role)
    if needed and capabilities and needed not in capabilities:
        out.append(
            f"{role.value} sends {needed} on every call and this model reports"
            f" {sorted(capabilities)}: the request is refused before inference, per row"
        )
    refused = REFUSED_CAPABILITY.get(role)
    if refused and refused in capabilities:
        out.append(
            f"{role.value} reads what the model writes, and a {refused} model spends its"
            " budget on a trace that never reaches `content` through the chat API"
        )
    if role in NEEDS_SYSTEM and template and "System" not in template:
        out.append(
            f"{role.value} is driven by a system prompt and this template renders none,"
            " so the prompt would be dropped without a word"
        )
    if role is Role.generation and template and "Tools" not in template:
        out.append(
            "generation on the agent path needs the template to render tool schemas,"
            " and this one does not"
        )
    return out


def refuse_unfit_model(role: Role, model_name: str) -> None:
    try:
        shown = llm.shown(model_name)
    except Exception as e:  # a probe must not become the reason a role cannot be assigned
        return _unknown(role, model_name, e)
    found = complaints(role, shown)
    if found:
        raise ValueError(f"{model_name} does not fit {role.value}: " + "; ".join(found))


def _unknown(role: Role, model_name: str, error: Exception) -> None:
    log.warning(
        "model.acceptance_unknown", role=role.value, model=model_name, error=str(error)
    )
