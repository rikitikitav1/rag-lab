import json

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


# our policies as the standard harness expresses them: hooks around the agent loop instead of
# nodes we wire ourselves. What the loop keeps in AgentResult lives here for the same reason
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
        self.model_calls = 0


def _turn_results(messages: list) -> list:
    """Every tool message of the turn that just finished, newest turn only."""
    tail = []
    for message in reversed(messages):
        if getattr(message, "type", None) == "tool":
            tail.append(message)
            continue
        break
    return list(reversed(tail))


def _corpus_results(results: list) -> list:
    corpus = [m for m in results if getattr(m, "name", None) == agent_tools.CORPUS_TOOL]
    # a failed search says nothing about coverage, the loop leaves it out of the verdict
    return [m for m in corpus if not str(m.content).startswith(errors.ERROR_PREFIX)]


def _artifacts(messages: list) -> list:
    return [s for m in messages for s in (getattr(m, "artifact", None) or [])]


class CoverageGateMiddleware(AgentMiddleware):
    """One verdict per turn over everything the corpus returned, exactly as the loop does it."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def before_model(self, state, runtime):
        turn = _turn_results(state["messages"])
        if not turn:
            return None
        results = _corpus_results(turn)
        remote_sources = _artifacts([m for m in turn if m not in results])
        sources = _artifacts(results)
        verdict = policy.verdict(sources, self.run.gate) if results else None
        # the axis overrules the gate on the fact of a corpus call, not on the gate's opinion
        if results and self.run.gate.off_topic:
            verdict = policy.FallbackReason.off_topic
        if not verdict:
            self.run.sources.extend(sources + remote_sources)
            return None
        self.run.sources.extend(remote_sources)

        rewritten = []
        if verdict in (policy.FallbackReason.weak, policy.FallbackReason.off_topic) and (
            self.run.gate.drop_weak_context
        ):
            log.info("mw.weak_context_dropped", dropped=len(sources))
            self.run.dropped.extend(s.source for s in sources)
            self.run.dropped_hits.extend(sources)
            for message in results:
                # add_messages replaces a message when the same id comes back
                message.content = chat.NO_RESULTS
                message.artifact = []
                rewritten.append(message)
            sources = []
        self.run.sources.extend(sources)

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
            log.info("mw.external_opened", reason=str(verdict))
        return {"messages": rewritten} if rewritten else None


class ToolboxMiddleware(AgentMiddleware):
    """corpus_first withholds the admitted tools, and the hop budget lives here too."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def wrap_model_call(self, request, handler):
        self.run.model_calls += 1
        offered = [agent_tools.CORPUS_TOOL]
        if self.run.external:
            offered += list(self.run.remote)
        tools = [t for t in request.tools if getattr(t, "name", None) in offered]

        # the loop answers one last time without tools when the hops are spent
        if self.run.model_calls > self.run.max_hops:
            messages = list(request.messages)
            if not self.run.sources and not self.run.no_evidence_prompted:
                self.run.no_evidence_prompted = True
                messages.append(
                    {
                        "role": "user",
                        "content": prompt_repo.active_template(Purpose.agent_no_evidence),
                    }
                )
            log.info("mw.forcing_final", calls=self.run.model_calls)
            return _without_tool_calls(handler(request.override(tools=[], messages=messages)))
        return handler(request.override(tools=tools))


# the loop never runs a tool after the cap: whatever the last turn says, the run ends with it
def _without_tool_calls(response):
    messages = getattr(response, "result", None) or [response]
    for message in messages:
        if getattr(message, "tool_calls", None):
            log.info("mw.dropped_call_after_cap", calls=len(message.tool_calls))
            message.tool_calls = []
    return response


class NudgeMiddleware(AgentMiddleware):
    """A turn that narrates a call instead of issuing one gets exactly one nudge."""

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
        log.info("mw.narrated_tool_call")
        return {
            "messages": [{"role": "user", "content": policy.TOOL_CALL_NUDGE}],
            "jump_to": "model",
        }


def build(run: Run) -> list:
    # before_* run in list order, after_* in reverse, wrap_* nest with the first one outermost
    return [CoverageGateMiddleware(run), ToolboxMiddleware(run), NudgeMiddleware(run)]


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
