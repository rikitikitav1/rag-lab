import logging_setup
from engines import ollama
from models.registry import EngineKind, Placement, Role

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
    spec = _engine_of(model_name)
    # only a vLLM pooling server scores pairs; an engine not yet readable, as mid-bootstrap, is no no
    if role is Role.reranking and spec is not None and spec.kind is not EngineKind.vllm:
        raise ValueError(f"{model_name} cannot rerank: the role lives on a vLLM pooling server")
    if spec is not None and spec.kind is EngineKind.vllm:
        return _refuse_unfit_on_vllm(role, model_name, spec)
    # only ollama describes its models, so an engine that cannot be asked is not a failed probe
    if spec is not None and spec.kind is not EngineKind.ollama:
        log.info("model.acceptance_not_probed", role=role.value, model=model_name, engine=spec.name)
        return None
    try:
        shown = ollama.shown(model_name, spec)
    except Exception as e:  # a probe must not become the reason a role cannot be assigned
        return _unknown(role, model_name, e)
    found = complaints(role, shown)
    if found:
        raise ValueError(f"{model_name} does not fit {role.value}: " + "; ".join(found))


def _engine_of(model_name: str):
    import engines

    try:
        found = engines.find_model(model_name)
    except Exception:
        return None
    return found.engine if found else None


def _unknown(role: Role, model_name: str, error: Exception) -> None:
    log.warning(
        "model.acceptance_unknown", role=role.value, model=model_name, error=str(error)
    )


# this build of vLLM on the cpu has no kernels for these, so the server would never come up with them
CPU_UNREADABLE = ("AWQ", "GGUF")


def _refuse_unfit_on_vllm(role: Role, model_name: str, spec) -> None:
    from engines import vllm

    quant = vllm.artifact_of(model_name).get("quant")
    if spec.placement is Placement.cpu and quant in CPU_UNREADABLE:
        raise ValueError(f"{model_name} is {quant}, which {spec.name} on the cpu cannot serve")
    if role is not Role.generation:
        return
    # the agent calls tools, and a server started without a parser answers them as plain text
    probed = vllm.tool_calls_probed(spec, model_name)
    if probed is False:
        raise ValueError(
            f"{model_name} on {spec.name} does not return tool calls; start the server with"
            " --enable-auto-tool-choice and a --tool-call-parser"
        )
    if probed is None:
        log.warning("model.tool_probe_unknown", model=model_name, engine=spec.name)
