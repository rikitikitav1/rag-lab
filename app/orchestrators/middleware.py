from dataclasses import dataclass, field

import agent_tools
import errors
import logging_setup
import outcomes
import prompt_repo
from langchain.agents.middleware import AgentMiddleware, hook_config
from models.registry import Purpose
from use_cases import agent_policy as policy
from use_cases import chat

log = logging_setup.get_logger(__name__)


# the same policies as hooks instead of nodes; what the graph keeps in state lives here
@dataclass
class Run:
    remote: dict
    gate: policy.Gate
    external: bool
    max_hops: int
    sources: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    dropped_hits: list = field(default_factory=list)
    tool_errors: dict = field(default_factory=dict)
    fallback_reason: str = policy.FallbackReason.none
    fallback_opened: bool = False
    announced: bool = False
    nudges: int = 1
    no_evidence_prompted: bool = False
    model_calls: int = 0
    forcing_final: bool = False
    forced_empty: bool = False


def _turn_results(messages: list) -> list:
    tail = []
    for message in reversed(messages):
        if getattr(message, "type", None) == "tool":
            tail.append(message)
            continue
        break
    return list(reversed(tail))


def _is_error(message) -> bool:
    # our dispatch prefixes its own text, ToolNode writes "Error: ..." when the call itself blew up
    return (
        getattr(message, "status", None) == "error"
        or str(message.content).startswith(errors.ERROR_PREFIX)
        or str(message.content).startswith(errors.ERROR_PREFIX.capitalize())
    )


def _corpus_results(results: list) -> list:
    corpus = [m for m in results if getattr(m, "name", None) == agent_tools.CORPUS_TOOL]
    # a failed search says nothing about coverage, so it stays out of the verdict
    return [m for m in corpus if not _is_error(m)]


def _artifacts(messages: list) -> list:
    return [s for m in messages for s in (getattr(m, "artifact", None) or [])]


# one verdict per turn over everything the corpus returned, the same rule the graph applies
class CoverageGateMiddleware(AgentMiddleware):
    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    # decided after the turn is read, or a hit on the last hop is thrown away unseen
    def _no_evidence(self) -> list:
        last_turn = self.run.model_calls >= self.run.max_hops or self.run.forcing_final
        if not last_turn or self.run.sources or self.run.no_evidence_prompted:
            return []
        self.run.no_evidence_prompted = True
        log.info("middleware.no_evidence_prompt", calls=self.run.model_calls)
        return [{"role": "user", "content": prompt_repo.active_template(Purpose.agent_no_evidence)}]

    def before_model(self, state, runtime):
        turn = _turn_results(state["messages"])
        if not turn:
            prompt = self._no_evidence()
            return {"messages": prompt} if prompt else None
        results = _corpus_results(turn)
        sources = _artifacts(results)
        verdict = policy.verdict(sources, self.run.gate) if results else None
        # the axis overrules the gate on the fact of a corpus call, not on the gate's opinion
        if results and self.run.gate.off_topic:
            verdict = policy.FallbackReason.off_topic
        # sources are collected in call order, because the two arms are compared row by row
        dropping = bool(verdict) and self.run.gate.drop_weak_context and verdict in (
            policy.FallbackReason.weak,
            policy.FallbackReason.off_topic,
        )
        for message in turn:
            if message in results and dropping:
                continue
            self.run.sources.extend(getattr(message, "artifact", None) or [])
        if not verdict:
            prompt = self._no_evidence()
            return {"messages": prompt} if prompt else None

        rewritten = []
        if dropping:
            log.info("middleware.weak_context_dropped", dropped=len(sources))
            self.run.dropped.extend(s.source for s in sources)
            self.run.dropped_hits.extend(sources)
            for message in results:
                # add_messages replaces a message when the same id comes back
                message.content = chat.NO_RESULTS
                message.artifact = []
                rewritten.append(message)

        if self.run.gate.announce and not self.run.external:
            notice = prompt_repo.active_template(Purpose.agent_fallback)
            last = results[-1]
            last.content = (
                f"{last.content}\n\n{notice.replace('{tools}', self.run.gate.tool_signatures)}"
            )
            if last not in rewritten:
                rewritten.append(last)
            self.run.announced = True
        if self.run.fallback_reason == policy.FallbackReason.none:
            self.run.fallback_reason = verdict
        if not self.run.external and self.run.remote:
            self.run.external = True
            self.run.fallback_opened = True
            log.info("middleware.external_opened", reason=str(verdict))
        rewritten += self._no_evidence()
        return {"messages": rewritten} if rewritten else None


# corpus_first withholds the admitted tools, and the hop budget lives here too
class ToolboxMiddleware(AgentMiddleware):
    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def wrap_model_call(self, request, handler):
        self.run.model_calls += 1
        offered = [tool.name for tool in agent_tools.registry()]
        if self.run.external:
            offered += list(self.run.remote)
        tools = [t for t in request.tools if getattr(t, "name", None) in offered]

        # the last answer, without tools, is the one the run is judged on
        if self.run.model_calls > self.run.max_hops or self.run.forcing_final:
            self.run.forcing_final = False
            log.info("middleware.forcing_final", calls=self.run.model_calls)
            return _without_tool_calls(handler(request.override(tools=[])))
        return handler(request.override(tools=tools))


# no tool runs after the cap: whatever the last turn says, the run ends with it
def _without_tool_calls(response):
    messages = getattr(response, "result", None) or [response]
    for message in messages:
        if getattr(message, "tool_calls", None):
            log.info("middleware.dropped_call_after_cap", calls=len(message.tool_calls))
            message.tool_calls = []
    return response


# a turn that narrates a call instead of issuing one gets exactly one nudge
class NudgeMiddleware(AgentMiddleware):
    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        message = state["messages"][-1]
        if getattr(message, "tool_calls", None) or not self.run.nudges:
            return None
        if self.run.model_calls > self.run.max_hops:
            return None
        names = (*self.run.remote, agent_tools.CORPUS_TOOL)
        if not outcomes.narrated_tool_call(str(message.content), names):
            return None
        self.run.nudges -= 1
        log.info("middleware.narrated_tool_call")
        return {
            "messages": [{"role": "user", "content": policy.TOOL_CALL_NUDGE}],
            "jump_to": "model",
        }


# a turn that came back with nothing gets the same last answer the graph asks for
class EmptyTurnMiddleware(AgentMiddleware):
    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        message = state["messages"][-1]
        if getattr(message, "tool_calls", None) or str(message.content).strip():
            return None
        if self.run.model_calls > self.run.max_hops or self.run.forced_empty:
            return None
        self.run.forced_empty = True
        self.run.forcing_final = True
        log.info("middleware.empty_turn", calls=self.run.model_calls)
        return {"jump_to": "model"}


def build(run: Run) -> list:
    # after_* hooks run in reverse list order, so Nudge sits last to see a turn before EmptyTurn
    return [
        CoverageGateMiddleware(run),
        ToolboxMiddleware(run),
        EmptyTurnMiddleware(run),
        NudgeMiddleware(run),
    ]
