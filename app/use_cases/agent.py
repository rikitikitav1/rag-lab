import time
from dataclasses import asdict, dataclass, field

import agent_tools
import config
import errors
import job_queue
import llm
import logging_setup
import outcomes
import prompt_repo
import version
from models.eval import QuestionLog
from models.registry import Pipeline, Purpose
from orchestrators import graph as orch_graph
from orchestrators import middleware as orch_middleware
from orchestrators import react as orch_react
from orm.sync_db import Session
from sqlalchemy.exc import SQLAlchemyError
from use_cases import chat
from use_cases.agent_policy import (
    FallbackPolicy,
    FallbackReason,
    Gate,
    GateSignal,
    Orchestrator,
    Topic,
    required_values,
    signatures,
)

import db

log = logging_setup.get_logger(__name__)


@dataclass
class AgentResult:
    text: str = ""
    success: bool = False
    hops: int = 0
    sources: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_prompt_tokens: int = 0
    truncated_hops: int = 0
    last_prompt_tokens: int = 0
    failed: bool = False
    elapsed: float = 0.0
    fallback_reason: str = FallbackReason.none
    fallback_announced: bool = False
    fallback_opened: bool = False
    no_evidence_prompted: bool = False
    dropped_sources: list = field(default_factory=list)
    dropped_hits: list = field(default_factory=list)
    outcome: str = outcomes.Outcome.error
    tool_errors: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)

    # one number per question hides where it went: the model, our own probes, or the tools
    def took(self, stage: str, started: float) -> None:
        spent = round((time.perf_counter() - started) * 1000)
        current = self.stages.setdefault(stage, {"ms": 0, "calls": 0})
        current["ms"] += spent
        current["calls"] += 1

    # only ollama's own api reports these, the openai-compat client has nothing to add
    def note_server_timings(self, meta: dict) -> None:
        for key, stage in (
            ("prompt_eval_duration", "prefill"),
            ("eval_duration", "decode"),
            ("total_duration", "model"),
        ):
            value = meta.get(key)
            if value:
                current = self.stages.setdefault(stage, {"ms": 0, "calls": 0})
                current["ms"] += round(value / 1_000_000)
                current["calls"] += 1

    # history only grows, so a shorter prompt than the last hop means the server trimmed it
    def note_prompt(self, tokens: int | None) -> None:
        if not tokens:
            return
        if self.last_prompt_tokens - tokens > 512:
            self.truncated_hops += 1
        self.last_prompt_tokens = tokens


# uses asyncio.run for remote MCP tools: call from sync context only, never from an event loop
def run(
    question: str,
    role: str = "generation",
    max_hops: int | None = None,
    run_name: str | None = None,
    language: str | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
    model: str | None = None,
    fallback_policy: str | None = None,
    gate_signal: str | None = None,
    weak_distance: float | None = None,
    topic_threshold: float | None = None,
    orchestrator: str | None = None,
) -> AgentResult:
    start = time.perf_counter()
    if max_hops is None:
        max_hops = config.settings.agent.max_hops
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    policy = FallbackPolicy(fallback_policy or config.settings.agent.fallback_policy)
    system = prompt_repo.active_template(Purpose.agent_system)
    if language:
        system += f"\n\n{chat._language_directive(language)}"
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    result = AgentResult(messages=messages)
    threshold = (
        topic_threshold
        if topic_threshold is not None
        else config.settings.agent.topic_threshold
    )
    # zero is how a run switches the axis off now that the config carries a default
    topic = Topic(threshold=threshold if threshold else None)
    if topic.threshold is not None:
        started = time.perf_counter()
        topic.score = _topic_score(question)
        result.took("topic", started)

    remote = {t.name: t for t in agent_tools.remote_tools()}
    configured = sorted({name.split("__")[0] for name in remote})
    off_topic = topic.score is not None and topic.score >= topic.threshold
    admission_ran = False
    if off_topic:
        log.info("agent.off_topic", score=round(topic.score, 3), threshold=topic.threshold)
        remote = {}
    # admission costs one llm call per tool: an off-topic question has no tool left to admit
    elif remote and policy != FallbackPolicy.agent_choice:
        started = time.perf_counter()
        remote = _admissible(question, remote, result, role=role, model=model)
        result.took("admission", started)
        admission_ran = True
    external = policy == FallbackPolicy.agent_choice
    gate = Gate(tool_signatures=signatures(remote.values()))
    if off_topic:
        gate.off_topic = True
        gate.drop_weak_context = True
    if policy == FallbackPolicy.corpus_first_weak:
        gate.signal = GateSignal(gate_signal or config.settings.agent.gate_signal)
        if gate.signal != GateSignal.distance:
            gate.top = config.settings.agent.gate_candidates
            gate.threshold = config.settings.agent.weak_threshold
        if gate.signal != GateSignal.cross_encoder:
            gate.distance_threshold = (
                weak_distance
                if weak_distance is not None
                else config.settings.agent.weak_distance
            )
        gate.drop_weak_context = gate.off_topic or bool(remote)
    orchestrator = Orchestrator(orchestrator or Orchestrator.langgraph_ported)
    # the loop is gone: the value is readable in old logs, runnable nowhere
    if orchestrator == Orchestrator.handrolled:
        raise ValueError(f"orchestrator '{orchestrator}' was removed, runs cannot ask for it")
    gate.announce = not external and bool(remote)
    if orchestrator in (Orchestrator.langgraph_idiomatic, Orchestrator.langgraph_middleware):
        ctx = orch_graph.context(
            remote=remote, k=k, use_rerank=use_rerank, role=role, model=model,
            max_hops=max_hops,
        )
        if orchestrator == Orchestrator.langgraph_middleware:
            mw_run = orch_middleware.Run(remote, gate, external, max_hops)
            orch_react.invoke(
                question, system, ctx, result,
                middleware=orch_middleware.build(mw_run), run=mw_run,
            )
        else:
            orch_react.invoke(question, system, ctx, result)
    else:
        orch_graph.invoke(
            question,
            system,
            orch_graph.context(
                remote=remote, gate=gate, external=external, k=k, use_rerank=use_rerank,
                role=role, model=model, max_hops=max_hops,
            ),
            result,
        )
    messages = result.messages

    result.sources = _unique_sources(result.sources)

    result.outcome = outcomes.classify(
        result.text,
        bool(result.sources),
        (*remote, agent_tools.CORPUS_TOOL),
        # a guard that fired is a failure, not a run that politely used up its hops
        exhausted=result.hops >= max_hops and not result.failed,
    )
    result.success = bool(result.sources and result.text)

    result.elapsed = round(time.perf_counter() - start, 3)
    log.info(
        "agent.done",
        hops=result.hops,
        success=result.success,
        outcome=result.outcome,
        sources=len(result.sources),
    )
    try:
        admitted = sorted({name.split("__")[0] for name in remote})
        _log_answer(
            question, result, run_name, language, k, use_rerank, model,
            admitted, policy, configured,
            # the idiomatic arm runs no gate, so recording one would describe a run that never was
            None if orchestrator == Orchestrator.langgraph_idiomatic else gate,
            max_hops, topic, admission_ran,
            {
                "name": str(orchestrator),
                "client": (
                    "ChatOllama"
                    if orchestrator
                    in (Orchestrator.langgraph_idiomatic, Orchestrator.langgraph_middleware)
                    else "openai-compat"
                ),
                **orch_graph.versions(),
            },
        )
    except SQLAlchemyError as e:
        log.error("agent_log.insert_failed", reason=str(e))

    if result.outcome in (
        outcomes.Outcome.narrated_call, outcomes.Outcome.error, outcomes.Outcome.exhausted
    ):
        result.text = chat.NO_RESULTS
    return result


def _admissible(
    question: str, tools: dict, result: AgentResult, role: str = "generation", model=None
) -> dict:
    system = prompt_repo.active_template(Purpose.agent_tool_match)
    admitted = {}
    for name, tool in tools.items():
        values = required_values(tool)
        if not values:
            admitted[name] = tool
            continue
        user = (
            f"Required values: {values}\n\nQuestion: {question}\n\n"
            "Does the question state every required value? Answer yes or no."
        )
        try:
            completion = llm.ask(system=system, user=user, role=role, model=model)
            verdict = (completion.text or "").strip().lower()
        except RuntimeError as e:
            log.error("agent.tool_match_failed", tool=name, error=str(e))
            result.tool_errors[name] = "tool_match"
            continue
        if verdict.startswith("yes"):
            admitted[name] = tool
        else:
            log.info("agent.tool_rejected", tool=name)
    return admitted





def _topic_score(question: str) -> float | None:
    try:
        return db.nearest_distance(llm.embed(question))
    except Exception as e:
        log.error("agent.topic_score_failed", error=str(e))
        return None


def _corpus_fingerprint() -> dict | None:
    try:
        return db.corpus_fingerprint()
    except Exception as e:
        log.error("agent.corpus_fingerprint_failed", error=str(e))
        return None


def _retrieval_snapshot(sources: list, dropped_hits: list, dropped: list | None = None) -> dict:
    def corpus_only(rows):
        return [s for s in rows if not s.source.startswith("mcp:")]

    kept = corpus_only(sources)
    # the gate decides on what retrieval returned, so its input survives the drop
    scored = kept + corpus_only(dropped_hits)
    distances = [s.vector_distance for s in scored if s.vector_distance is not None]
    scores = [s.rerank_score for s in scored if s.rerank_score is not None]
    return {
        "results_count": len(kept),
        "min_distance": min(distances) if distances else None,
        "top_rerank_score": max(scores) if scores else None,
        "dropped_sources": sorted(set(dropped or [])) or None,
    }


def _unique_sources(sources: list) -> list:
    seen: set[str] = set()
    out = []
    for s in sources:
        if s.source not in seen:
            seen.add(s.source)
            out.append(s)
    return out


def _context_from_messages(messages) -> str:
    return "\n\n".join(
        m["content"]
        for m in messages
        if isinstance(m, dict)
        and m.get("role") == "tool"
        # the standard tool node capitalises its errors, and the judge must not score one
        and not m["content"].lower().startswith(errors.ERROR_PREFIX)
        and not m["content"].startswith(chat.NO_RESULTS)
    )


def _log_answer(
    question_text: str,
    result: AgentResult,
    run_name: str | None,
    language: str | None = None,
    k: int | None = None,
    use_rerank: bool | None = None,
    model: str | None = None,
    mcp_names: list[str] | None = None,
    fallback_policy: str = FallbackPolicy.corpus_first,
    mcp_configured: list[str] | None = None,
    gate: Gate | None = None,
    max_hops: int | None = None,
    topic: Topic | None = None,
    admission_ran: bool = False,
    orchestrator: dict | None = None,
) -> None:
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    lang = chat._resolve_language(question_text, language)
    with Session() as session:
        question = chat._find_or_create_question(session, question_text, lang)
        log_row = QuestionLog(
            run_name=run_name,
            question_id=question.id,
            answered=result.success,
            answer=result.text,
            context=_context_from_messages(result.messages) or None,
            sources=[asdict(s) for s in result.sources],
            pipeline=Pipeline.agent.value,
            models={
                "generation": model or llm.resolve_name("generation"),
                "embedding": llm.resolve_name("embedding"),
            },
            prompts={
                "agent_system": prompt_repo.active_version(Purpose.agent_system),
                **(
                    {"agent_fallback": prompt_repo.active_version(Purpose.agent_fallback)}
                    if result.fallback_announced
                    else {}
                ),
                **(
                    {"agent_no_evidence": prompt_repo.active_version(Purpose.agent_no_evidence)}
                    if result.no_evidence_prompted
                    else {}
                ),
                **(
                    {"agent_tool_match": prompt_repo.active_version(Purpose.agent_tool_match)}
                    if admission_ran
                    else {}
                ),
            },
            metrics={
                "hops": result.hops,
                "no_evidence": not bool(result.sources),
                "context_tokens": result.max_prompt_tokens,
                "retrieval": _retrieval_snapshot(
                    result.sources, result.dropped_hits, result.dropped_sources
                ),
                "fallback_reason": str(result.fallback_reason),
                "fallback_opened": result.fallback_opened,
                "outcome": str(result.outcome),
                # without this the report reads a guard that fired as a run that spent its hops
                "failed": result.failed or None,
                "stages": result.stages or None,
                "tool_errors": result.tool_errors or None,
                "config": {
                    "rerank": use_rerank,
                    "orchestrator": orchestrator,
                    "fallback_policy": str(fallback_policy),
                    "gate": (
                        {
                            "signal": str(gate.signal),
                            "top": gate.top,
                            "threshold": gate.threshold,
                            "distance_threshold": gate.distance_threshold,
                        }
                        if gate and (gate.threshold or gate.distance_threshold)
                        else None
                    ),
                    "distance_threshold": round(
                        config.settings.retrieval.distance_threshold, 3
                    ),
                    "context_length": llm.server_context_length(
                        model or llm.resolve_name("generation")
                    ),
                    "truncated_hops": result.truncated_hops or None,
                    "k": k or config.settings.retrieval.results_limit,
                    "max_hops": max_hops or config.settings.agent.max_hops,
                    "corpus": config.settings.corpus.description,
                    "corpus_fingerprint": _corpus_fingerprint(),
                    "code_version": version.CODE_VERSION,
                    "drop_weak_context": bool(gate and gate.drop_weak_context),
                    "topic": (
                        {
                            "threshold": topic.threshold,
                            "score": round(topic.score, 3) if topic.score is not None else None,
                            "input": "question",
                        }
                        if topic and topic.threshold is not None
                        else None
                    ),
                    "mcp": mcp_names or [],
                    "mcp_configured": mcp_configured or [],
                },
            },
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            elapsed=result.elapsed,
        )
        session.add(log_row)
        session.commit()
        log_id = log_row.id

    if result.success and run_name is None:
        job_queue.enqueue("judge_answers", {"log_ids": [log_id]})
