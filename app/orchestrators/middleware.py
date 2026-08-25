import json

import agent_tools
import logging_setup
import outcomes
import prompt_repo
from langchain.agents.middleware import AgentMiddleware
from models.registry import Purpose
from use_cases import agent_policy as policy
from use_cases import chat

log = logging_setup.get_logger(__name__)


# our policies as the standard harness expresses them: hooks around the agent loop instead of
# nodes we wire ourselves. The state they share lives in a plain dict handed in at build time
class Run:
    def __init__(self, remote: dict, gate: policy.Gate, external: bool, max_hops: int):
        self.remote = remote
        self.gate = gate
        self.external = external
        self.max_hops = max_hops
        self.sources: list = []
        self.dropped: list = []
        self.dropped_hits: list = []
        self.tool_errors: dict = {}
        self.fallback_reason = policy.FallbackReason.none
        self.fallback_opened = False
        self.announced = False
        self.nudges = 1
        self.no_evidence_prompted = False


class ToolboxMiddleware(AgentMiddleware):
    """corpus_first: the admitted tools stay out of the schema until the corpus fails."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def wrap_model_call(self, request, handler):
        offered = [agent_tools.CORPUS_TOOL]
        if self.run.external or self.run.fallback_opened:
            offered += list(self.run.remote)
        request.tools = [t for t in request.tools if t.name in offered]
        return handler(request)


class CoverageGateMiddleware(AgentMiddleware):
    """The gate reads what the corpus returned and decides whether to look outside."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def wrap_tool_call(self, request, handler):
        message = handler(request)
        name = getattr(request, "tool_call", {}).get("name") if request else None
        sources = list(getattr(message, "artifact", None) or [])
        if name != agent_tools.CORPUS_TOOL:
            self.run.sources.extend(sources)
            return message

        verdict = policy.verdict(sources, self.run.gate)
        if self.run.gate.off_topic and sources:
            verdict = policy.FallbackReason.off_topic
        if verdict in (policy.FallbackReason.weak, policy.FallbackReason.off_topic) and (
            self.run.gate.drop_weak_context
        ):
            log.info("mw.weak_context_dropped", dropped=len(sources))
            self.run.dropped.extend(s.source for s in sources)
            self.run.dropped_hits.extend(sources)
            message.content = chat.NO_RESULTS
            sources = []
        self.run.sources.extend(sources)

        if verdict and self.run.gate.announce and not self.run.external:
            notice = prompt_repo.active_template(Purpose.agent_fallback)
            message.content = (
                f"{message.content}\n\n"
                f"{notice.replace('{tools}', self.run.gate.tool_signatures)}"
            )
            self.run.announced = True
        if verdict and self.run.fallback_reason == policy.FallbackReason.none:
            self.run.fallback_reason = verdict
        if verdict and not self.run.external and self.run.remote:
            self.run.fallback_opened = True
            log.info("mw.external_opened", reason=str(verdict))
        return message


class NudgeMiddleware(AgentMiddleware):
    """A turn that narrates a call instead of issuing one gets exactly one nudge."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def after_model(self, state, runtime):
        message = state["messages"][-1]
        if getattr(message, "tool_calls", None) or not self.run.nudges:
            return None
        names = (*self.run.remote, agent_tools.CORPUS_TOOL)
        if not outcomes.narrated_tool_call(str(message.content), names):
            return None
        self.run.nudges -= 1
        log.info("mw.narrated_tool_call")
        return {
            "messages": [{"role": "user", "content": policy.TOOL_CALL_NUDGE}],
            "jump_to": "model",
        }


class NoEvidenceMiddleware(AgentMiddleware):
    """Nothing answered: ask once more without tools rather than return silence."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def after_model(self, state, runtime):
        message = state["messages"][-1]
        if getattr(message, "tool_calls", None) or self.run.sources:
            return None
        if self.run.no_evidence_prompted or not str(message.content or "").strip():
            return None
        self.run.no_evidence_prompted = True
        return {
            "messages": [
                {"role": "user", "content": prompt_repo.active_template(Purpose.agent_no_evidence)}
            ],
            "jump_to": "model",
        }


def build(run: Run) -> list:
    return [
        ToolboxMiddleware(run),
        CoverageGateMiddleware(run),
        NudgeMiddleware(run),
        NoEvidenceMiddleware(run),
    ]


def tools_for(run: Run, k=None, use_rerank=None) -> list:
    from langchain_core.tools import StructuredTool

    def make(name: str, tool):
        def call(**kwargs) -> tuple[str, list]:
            res = agent_tools.dispatch(
                name, json.dumps(kwargs), extra=run.remote, k=k, use_rerank=use_rerank,
                gate_top=run.gate.top,
            )
            if res.meta.get("error_kind"):
                run.tool_errors[name] = res.meta["error_kind"]
            return res.content, res.meta.get("sources", [])

        return StructuredTool.from_function(
            func=call,
            name=name,
            description=tool.description,
            args_schema=tool.parameters,
            response_format="content_and_artifact",
        )

    return [make(t.name, t) for t in agent_tools.registry()] + [
        make(name, tool) for name, tool in run.remote.items()
    ]
