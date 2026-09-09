"""One recorded row driven through a graph again, with the model and the tools replaced by it.

What this compares is the machine, not the model: `chat` returns the turns the row recorded and
`dispatch` returns the pieces it recorded, so any difference is the graph's own doing. A threshold
would compare two samplings of a model instead, which is why the checklist refuses one here.
"""

from dataclasses import dataclass

import agent_tools
import llm
from use_cases import chat

# 1 the first pass: seven fields compared byte for byte over one recorded run
SCHEMA = 1


@dataclass
class Recorded:
    transcript: list
    contexts: list
    chunks: list
    sources: list
    spans: list


def of_row(ql) -> Recorded:
    metrics = ql.metrics or {}
    return Recorded(
        transcript=list(ql.transcript or []),
        contexts=list(ql.contexts or []),
        chunks=list(ql.chunks or []),
        sources=list(ql.sources or []),
        spans=list(metrics.get("spans") or []),
    )


# only the model's own turns, in order: the tool answers are replayed by `dispatch`, not asked for
def turns_of(transcript: list) -> list:
    turns = []
    for entry in transcript:
        if entry.get("role") != "assistant":
            continue
        calls = [
            _Call(str(i), _Function(c.get("name") or "", c.get("arguments") or "{}"))
            for i, c in enumerate(entry.get("tool_calls") or [])
        ]
        turns.append(
            llm.ChatTurn(
                text=entry.get("content") or None,
                tool_calls=calls,
                message={"role": "assistant", "content": entry.get("content") or "",
                         "tool_calls": entry.get("tool_calls") or []},
                prompt_tokens=0,
                completion_tokens=0,
            )
        )
    return turns


# the spans say which pieces and which sources each call produced, and `contexts` alone cannot
def results_of(recorded: Recorded) -> list:
    # by name, not by a cursor: the row's `sources` are deduplicated and the counts would slide
    by_name = {s.get("source"): s for s in recorded.sources if isinstance(s, dict)}
    out, piece_at = [], 0
    for span in recorded.spans:
        pieces = recorded.contexts[piece_at: piece_at + span["pieces"]]
        chunks = recorded.chunks[piece_at: piece_at + span["pieces"]]
        piece_at += span["pieces"]
        # the hop comes from the span: the row kept the first occurrence and its hop with it
        named = [
            {**by_name[n], "hop": span.get("hop")}
            for n in (span.get("sources") or []) if n in by_name
        ]
        # the gate rewrote this call, and the row kept what it rewrote: hand the graph the before
        if span.get("dropped"):
            named = span["dropped"]["sources"]
            pieces = [_ELIDED] * span["dropped"]["pieces"]
            chunks = [None] * len(pieces)
        out.append(
            agent_tools.ToolResult(
                # a call that found nothing answered NO_RESULTS, and "" would read as context
                content="\n\n".join(pieces) if pieces else chat.NO_RESULTS,
                meta={"sources": [_Source(s) for s in named],
                      "contexts": pieces, "chunks": chunks},
            )
        )
    return out


# the text of a dropped piece is not kept; the gate reads the scores, and the model never saw it
_ELIDED = "[dropped by the gate, text not recorded]"


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _Call:
    id: str
    function: _Function


# the row keeps a source as a dict; the graph stamps objects, and `stamped` reads attributes
class _Source:
    def __init__(self, row: dict):
        for key, value in (row or {}).items():
            setattr(self, key, value)
        self.source = (row or {}).get("source", "")


# `graph.invoke` swallows every exception into `failed`, so a divergence is collected, not raised
class _Script(RuntimeError):
    pass


# rebuilt by name, so the fallback notice carries the signatures it carried then
def _admitted(names: list) -> tuple[dict, list]:
    import agent_tools

    wanted = set(names)
    live = {t.name: t for t in agent_tools.remote_tools() if t.name.split("__")[0] in wanted}
    return live, sorted(wanted - {n.split("__")[0] for n in live})


# the row addresses itself: every switch the graph read was written into `metrics.config`
def gate_of(snapshot: dict, remote: dict | None = None):
    from use_cases import agent_policy as policy

    recorded = snapshot.get("gate") or {}
    topic = snapshot.get("topic") or {}
    gate = policy.Gate()
    if recorded:
        gate.signal = policy.GateSignal(recorded["signal"])
        gate.top, gate.threshold = recorded.get("top"), recorded.get("threshold")
        gate.distance_threshold = recorded.get("distance_threshold")
    gate.off_topic = bool(
        topic.get("score") is not None
        and topic.get("threshold") is not None
        and topic["score"] >= topic["threshold"]
    )
    gate.drop_weak_context = bool(snapshot.get("drop_weak_context"))
    gate.announce = bool(snapshot.get("mcp"))
    if remote:
        from use_cases.agent_policy import signatures

        gate.tool_signatures = signatures(remote.values())
    return gate


# the live run appends it to the same prompt; an empty language means nobody was told anything
def _system_for(system: str, snapshot: dict) -> str:
    from use_cases import chat as chat_uc

    told = chat_uc.language_directive(snapshot.get("language") or "")
    return f"{system}\n\n{told}" if told else system


# the recorded prompt version, not today's: a replay compares graphs, not prompt drift
def _pinned_templates(versions: dict, problems: list):
    import prompt_repo

    def template(purpose):
        # `active_versions` keys the row by the member name, and the value is dotted
        version = (versions or {}).get(purpose.name)
        if not version:
            return prompt_repo.active_template(purpose)
        try:
            return prompt_repo.template_of(purpose, version)
        except RuntimeError as e:
            # one row pointing at a deleted version must not take the other 189 with it
            problems.append(str(e))
            raise

    return template


def rerun(row) -> tuple:
    from models.registry import Purpose
    from orchestrators import graph
    from use_cases import agent
    from use_cases.agent import AgentResult

    metrics = row.metrics or {}
    snapshot = metrics.get("config") or {}
    recorded, problems = of_row(row), []
    remote, missing = _admitted(snapshot.get("mcp") or [])
    if missing:
        return None, [f"the run admitted {missing}, and no tool of that name is configured now"]
    # this drives the graph, and driving another arm's row through it compares two machines
    arm = (snapshot.get("orchestrator") or {}).get("name")
    if arm and arm not in REPLAYABLE_ARMS:
        return None, [f"the row was taken by `{arm}`, and this replays the graph"]
    # truthiness here and `run_debts.REPLAY_SHAPES` there must split the same shapes the same way
    if (
        (metrics.get("retrieval") or {}).get("dropped_sources")
        and metrics.get("fallback_reason") == "weak"
        and not any(s.get("dropped") for s in (metrics.get("spans") or []))
    ):
        return None, ["a weak verdict dropped context and the row predates recording it"]

    turns = turns_of(recorded.transcript)
    calls = list(zip(results_of(recorded), recorded.spans, strict=True))

    offered = []

    def chat(messages, tools=None, role=None, model=None):
        offered.append(sorted({s.get("function", {}).get("name") for s in (tools or [])} - {None}))
        if not turns:
            problems.append("the graph asked for a turn the row never recorded")
            raise _Script("out of turns")
        return turns.pop(0)

    def dispatch(name, arguments, **kwargs):
        if not calls:
            problems.append(f"the graph called {name} beyond the calls the row recorded")
            raise _Script("out of calls")
        result, span = calls.pop(0)
        if span.get("tool") != name:
            problems.append(f"the row recorded {span.get('tool')} where the graph called {name}")
        return result

    template = _pinned_templates(row.prompts, problems)
    result = AgentResult()
    try:
        system = template(Purpose.agent_system)
    except RuntimeError:
        return None, problems
    system = _system_for(system, snapshot)
    graph.invoke(
        row.question_text or "",
        system,
        graph.context(
            remote=remote, gate=gate_of(snapshot, remote),
            external=snapshot.get("fallback_policy") == "agent_choice",
            k=snapshot.get("k"), use_rerank=snapshot.get("rerank"),
            role="generation", model=None, max_hops=snapshot.get("max_hops"),
            variant=snapshot.get("variant"),
            chat=chat, dispatch=dispatch, template=template,
        ),
        result,
    )
    if turns:
        problems.append(f"{len(turns)} recorded turns the graph never asked for")
    if calls:
        problems.append(f"{len(calls)} recorded tool calls the graph never made")
    # rows written before the toolbox was recorded carry None and are not compared on it
    was_offered = metrics.get("tools_offered")
    if was_offered and [sorted(x) for x in was_offered] != offered[: len(was_offered)]:
        problems.append("the graph handed the model a different toolbox than the row records")
    # the same tail the live run takes, by name and not by a second copy of it
    agent.finish(result, (*remote, agent_tools.CORPUS_TOOL), snapshot.get("max_hops") or 0)
    return result, problems


# a byte comparison, not a threshold; `announced` replaced `prompts`, which compared row to row
FIELDS = ("transcript", "sources", "chunks", "contexts", "fallback_reason", "outcome",
          "announced", "announced_text", "spans")

# a row recorded before the field existed: comparing it would fail every old row and prove nothing
UNRECORDED = object()

# the arms this drives: `agent` is the retired hand-rolled loop the graph replaced turn for turn
REPLAYABLE_ARMS = ("langgraph_ported", "agent")


# the replay names its own calls, so the id is the one part of a span it cannot reproduce
def _comparable(spans, with_drop: bool = True) -> list:
    skip = {"tool_call_id"} if with_drop else {"tool_call_id", "dropped"}
    return [{k: v for k, v in s.items() if k not in skip} for s in (spans or [])]


# a row taken before the drop was recorded carries no such block, and demanding one fails every one
def _kept_the_drop(spans) -> bool:
    return any("dropped" in s for s in (spans or []))


def _replayed(row, result) -> dict:
    from use_cases.agent import transcript_of

    kept = _kept_the_drop((row.metrics or {}).get("spans"))
    return {
        "transcript": transcript_of(result.messages),
        "sources": [{"source": s.source, "hop": getattr(s, "hop", None)} for s in result.sources],
        "chunks": list(result.chunks),
        "contexts": list(result.contexts),
        "fallback_reason": str(result.fallback_reason),
        "outcome": str(result.outcome),
        # the row keeps this only as a prompt version, and copying it proved nothing
        "announced": bool(result.fallback_announced),
        "announced_text": result.announced_text or "",
        "spans": _comparable(result.spans, kept),
    }


def _recorded(row) -> dict:
    metrics = row.metrics or {}
    announced = "agent_fallback" in (row.prompts or {})
    text = metrics.get("announced_text")
    return {
        "transcript": list(row.transcript or []),
        "sources": [{"source": s.get("source"), "hop": s.get("hop")} for s in (row.sources or [])],
        "chunks": list(row.chunks or []),
        "contexts": list(row.contexts or []),
        "fallback_reason": str(metrics.get("fallback_reason")),
        "outcome": str(metrics.get("outcome")),
        "announced": announced,
        "announced_text": text if text else ("" if not announced else UNRECORDED),
        "spans": _comparable(metrics.get("spans"), _kept_the_drop(metrics.get("spans"))),
    }


def differences(row, result) -> list:
    was, now = _recorded(row), _replayed(row, result)
    return [f for f in FIELDS if was[f] is not UNRECORDED and was[f] != now[f]]


def report(run_name: str, rows) -> dict:
    import collections

    import version

    rows = list(rows)
    counted, differing, refused = 0, {}, {}
    # an artefact that does not say which branches it walked invites the reader to assume all
    branches = collections.Counter()
    for row in rows:
        result, problems = rerun(row)
        if result is None:
            refused[problems[0]] = refused.get(problems[0], 0) + 1
            continue
        counted += 1
        branches[str((row.metrics or {}).get("fallback_reason"))] += 1
        for field in differences(row, result) + problems:
            differing.setdefault(field, []).append(row.id)
    return {
        "schema": SCHEMA,
        "run_name": run_name,
        "fields": list(FIELDS),
        "rows": len(rows),
        "replayed": counted,
        "identical": counted - len({i for ids in differing.values() for i in ids}),
        "differing": {k: v[:10] for k, v in differing.items()},
        "branches_walked": dict(branches),
        "announced_on": sum(1 for r in rows if "agent_fallback" in (r.prompts or {})),
        # the notice text is comparable only where the row carries it, and that is worth counting
        "notice_compared_on": sum(
            1 for r in rows if _recorded(r)["announced_text"] is not UNRECORDED
        ),
        # the drop block is compared only where the row kept one: older rows never did
        "drop_compared_on": sum(
            1 for r in rows if _kept_the_drop((r.metrics or {}).get("spans"))
        ),
        "refused": refused,
        "code_version": version.CODE_VERSION,
    }
