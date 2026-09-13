import logging_setup
from engines import CARD, CardState, ollama
from models.registry import EngineKind, Model, ModelRole, Placement, Role
from openai import AuthenticationError, PermissionDeniedError
from orm.sync_db import Session
from sqlalchemy import select

# what each role needs of a model, in the terms the server answers in
REQUIRED_CAPABILITY = {Role.generation: "tools"}
REFUSED_CAPABILITY = {Role.judging: "thinking", Role.paraphrasing: "thinking", Role.ragas: "thinking"}
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


# by the engine the caller holds: by name alone, one model on two engines was checked on neither
def refuse_unfit_model(role: Role, model_name: str, engine_id: int | None = None) -> None:
    import engines

    spec = engines.spec_of_id(engine_id) if engine_id is not None else _engine_of(model_name)
    # only a vLLM pooling server scores pairs; an engine not yet readable, as mid-bootstrap, is not a no
    if role is Role.reranking and spec is not None and spec.kind is not EngineKind.vllm:
        raise ValueError(f"{model_name} cannot rerank: the role lives on a vLLM pooling server")
    if spec is not None and spec.kind is EngineKind.vllm:
        return _refuse_unfit_on_vllm(role, model_name, spec)
    if spec is not None and spec.kind is EngineKind.openai_compatible:
        return _refuse_markup_the_parser_leaves(role, model_name, spec)
    # only ollama describes its models, so an engine that cannot be asked is not a failed probe
    if spec is not None and spec.kind is not EngineKind.ollama:
        log.info("model.acceptance_not_probed", role=role.value, model=model_name, engine=spec.name)
        return None
    if spec is not None:
        _refuse_a_silent(spec, engines.driver(spec.kind).state(spec))
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


class EngineDown(Exception):
    pass


# gone or busy past its timeout: a role on it would fail its first call, and a handover would too
def _refuse_a_silent(spec, state) -> None:
    import engines

    if state in engines.SILENT:
        raise EngineDown(f"{spec.name} is {state}: gone, or not answering in time; a role on it would fail")


# an asleep server is not asked, and no answer for this start is recorded: the card must go to it first
class NeedsProbe(Exception):
    pass


POOLING_ROLES = (Role.embedding, Role.reranking, Role.ragas_embedding)


def _refuse_unfit_on_vllm(role: Role, model_name: str, spec) -> None:
    from engines import vllm

    state = vllm.card_state(spec)
    _refuse_a_silent(spec, state)
    pooling = vllm.pools(spec)
    if pooling is not None and pooling != (role in POOLING_ROLES):
        runner = "a pooling" if pooling else "a generating"
        raise ValueError(f"{model_name} on {spec.name} is served by {runner} runner, not one for {role}")
    quant = vllm.artifact_of(model_name).get("quant")
    if spec.placement is Placement.cpu and quant in CPU_UNREADABLE:
        raise ValueError(f"{model_name} is {quant}, which {spec.name} on the cpu cannot serve")
    if role is not Role.generation:
        return
    # a parserless server answers tools in text; an asleep one is not asked, it errs or hangs the door
    if spec.placement in CARD and state != CardState.AWAKE:
        probed = vllm.probe_now(spec, model_name)
        if probed is None:
            raise NeedsProbe(f"{spec.name} is {state} and no probe of {model_name} is recorded")
    else:
        probed = vllm.tool_calls_probed(spec, model_name)
    if probed is False:
        raise ValueError(
            f"{model_name} on {spec.name} does not return tool calls; start the server with"
            " --enable-auto-tool-choice and a --tool-call-parser"
        )
    if probed is None:
        log.warning("model.tool_probe_unknown", model=model_name, engine=spec.name)


# `over` is the model the role held when the seat was asked for: a later choice is not undone
def seat(role: Role, engine_id: int, model_name: str, *, over: int | None) -> None:
    with Session() as session:
        model = session.scalar(
            select(Model).where(Model.engine_id == engine_id, Model.name == model_name)
        )
        if model is None:
            raise ValueError(f"{model_name} is not registered on engine {engine_id}")
        found = session.get(ModelRole, role, with_for_update=True)
        if found is not None and found.model_id == model.id:
            return
        if (found.model_id if found else None) != over:
            raise ValueError(f"{role.value} was reassigned since the seat was asked; nothing seated")
        if found is None:
            session.add(ModelRole(role=role, model_id=model.id))
        else:
            found.model_id = model.id
        session.commit()
    log.info("model.role_seated", role=role.value, model=model_name, engine_id=engine_id)


_PROBE_TOOL = {"type": "function", "function": {
    "name": "search_corpus", "description": "Search the interview corpus",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}}


# a cloud writes its thinking and its call markup into the text, and the row's parser must cut all of it
def _refuse_markup_the_parser_leaves(role: Role, model_name: str, spec) -> None:
    import engines
    import llm
    from engines import answer_parsers

    found = engines.find_model(model_name, spec.id)
    parser = found.parser if found else "none"
    try:
        # through `llm`, so the probe's tokens land in the job's count and its errors read as every call's
        reply = llm.complete_on(
            spec, model_name, [{"role": "user", "content": "Use the tool to find how Redis persistence works."}],
            {"max_tokens": 512, "temperature": 0, "tools": [_PROBE_TOOL]}, role,
        )
    # no key, no address or a refused key is a row no call can reach, not a probe that failed to land
    except engines.Unconfigured as e:
        raise ValueError(f"{model_name} on {spec.name} cannot be called: {e}") from e
    # a property of the row, like a missing key: every run on it would stop on its first call
    except llm.NoUsage as e:
        raise ValueError(f"{model_name} on {spec.name} sends no token usage: {e}") from e
    except RuntimeError as e:
        refused = e.__cause__
        if isinstance(refused, (AuthenticationError, PermissionDeniedError)):
            raise ValueError(f"{model_name} on {spec.name} cannot be called: http {refused.status_code}") from e
        return _unknown(role, model_name, e)
    except Exception as e:  # a probe must not become the reason a role cannot be assigned
        return _unknown(role, model_name, e)
    left = answer_parsers.parse(parser, reply.choices[0].message.content).leftover_markers
    if left:
        raise ValueError(
            f"{model_name} leaves {', '.join(left)} in its answers under the parser {parser};"
            " name one with PATCH /v1/model/{id} and answer_parser"
        )
