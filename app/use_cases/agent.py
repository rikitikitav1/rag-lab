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
from models.eval import QuestionLog
from models.registry import Pipeline, Purpose
from orm.sync_db import Session
from sqlalchemy.exc import SQLAlchemyError
from use_cases import agent_graph, agent_react, chat
from use_cases.agent_policy import (
    TOOL_CALL_NUDGE,
    FallbackPolicy,
    FallbackReason,
    Gate,
    GateSignal,
    Orchestrator,
    Topic,
)
from use_cases.agent_policy import (
    required_values as _required_values,
)
from use_cases.agent_policy import (
    signatures as _signatures,
)
from use_cases.agent_policy import (
    verdict as _verdict,
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
    elapsed: float = 0.0
    fallback_reason: str = FallbackReason.none
    fallback_announced: bool = False
    fallback_opened: bool = False
    no_evidence_prompted: bool = False
    dropped_sources: list = field(default_factory=list)
    dropped_hits: list = field(default_factory=list)
    outcome: str = outcomes.Outcome.error
    tool_errors: dict = field(default_factory=dict)


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
        topic.score = _topic_score(question)

    remote = {t.name: t for t in agent_tools.remote_tools()}
    configured = sorted({name.split("__")[0] for name in remote})
    off_topic = topic.score is not None and topic.score >= topic.threshold
    admission_ran = False
    if off_topic:
        log.info("agent.off_topic", score=round(topic.score, 3), threshold=topic.threshold)
        remote = {}
    # admission costs one llm call per tool: an off-topic question has no tool left to admit
    elif remote and policy != FallbackPolicy.agent_choice:
        remote = _admissible(question, remote, result, role=role, model=model)
        admission_ran = True
    external = policy == FallbackPolicy.agent_choice
    gate = Gate(tool_signatures=_signatures(remote.values()))
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
    nudges = 1
    tools = agent_tools.schemas()
    if external:
        tools += [t.schema() for t in remote.values()]

    orchestrator = Orchestrator(orchestrator or Orchestrator.handrolled)
    graph_run = orchestrator != Orchestrator.handrolled
    if orchestrator == Orchestrator.langgraph_idiomatic:
        agent_react.invoke(
            question,
            system,
            agent_graph.context(
                remote=remote, k=k, use_rerank=use_rerank, role=role, model=model,
                max_hops=max_hops,
            ),
            result,
        )
        messages = result.messages
    elif graph_run:
        gate.announce = not external and bool(remote)
        agent_graph.invoke(
            question,
            system,
            agent_graph.context(
                remote=remote, gate=gate, external=external, k=k, use_rerank=use_rerank,
                role=role, model=model, max_hops=max_hops,
            ),
            result,
        )
        messages = result.messages
    else:
      for hop in range(1, max_hops + 1):
          result.hops = hop
          try:
              turn = llm.chat(messages, tools=tools, role=role, model=model)
          except RuntimeError as e:
              log.error("agent.hop_failed", hop=hop, error=str(e))
              break
          result.prompt_tokens += turn.prompt_tokens
          result.completion_tokens += turn.completion_tokens
          result.max_prompt_tokens = max(result.max_prompt_tokens, turn.prompt_tokens)
          if not turn.tool_calls and nudges and outcomes.narrated_tool_call(
              turn.text, (*remote, agent_tools.CORPUS_TOOL)
          ):
              nudges -= 1
              log.info("agent.narrated_tool_call", hop=hop)
              messages.append(turn.message or {"role": "assistant", "content": turn.text})
              messages.append({"role": "user", "content": TOOL_CALL_NUDGE})
              continue
          gate.announce = not external and bool(remote)
          if _apply_turn(
              turn, messages, result, k, use_rerank, remote if external else None, gate
          ):
              break
          if not external and remote and result.fallback_reason != FallbackReason.none:
              external = True
              result.fallback_opened = True
              tools = tools + [t.schema() for t in remote.values()]
              log.info("agent.external_opened", hop=hop, reason=result.fallback_reason)

    result.sources = _unique_sources(result.sources)

    # the graph runs its own final turn inside the flow, so only the loop needs this one
    if not result.success and orchestrator == Orchestrator.handrolled:
        log.info("agent.forcing_final", hops=result.hops, sources=len(result.sources))
        if not result.sources:
            messages.append({
                "role": "user",
                "content": prompt_repo.active_template(Purpose.agent_no_evidence),
            })
            result.no_evidence_prompted = True
        try:
            final = llm.chat(messages, role=role, model=model)
        except RuntimeError as e:
            log.error("agent.final_failed", error=str(e))
        else:
            result.hops += 1
            result.prompt_tokens += final.prompt_tokens
            result.completion_tokens += final.completion_tokens
            result.max_prompt_tokens = max(result.max_prompt_tokens, final.prompt_tokens)
            result.text = final.text or ""
            if final.finish_reason == "length":
                log.warning("agent.truncated", hops=result.hops)

    result.outcome = outcomes.classify(
        result.text,
        bool(result.sources),
        (*remote, agent_tools.CORPUS_TOOL),
        exhausted=result.hops >= max_hops,
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
                    if orchestrator == Orchestrator.langgraph_idiomatic
                    else "openai-compat"
                ),
                **(agent_graph.versions() if graph_run else {}),
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
        values = _required_values(tool)
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


def _apply_turn(
    turn: llm.ChatTurn,
    messages: list[dict],
    result: AgentResult,
    k: int | None = None,
    use_rerank: bool | None = None,
    remote: dict | None = None,
    gate: Gate | None = None,
) -> bool:
    if not turn.tool_calls:
        result.text = turn.text or ""
        result.success = bool(result.text)
        if turn.finish_reason == "length":
            log.warning("agent.truncated", hops=result.hops)
        return True

    messages.append(turn.message)
    gate = gate or Gate()
    calls = []

    for tc in turn.tool_calls:
        log.info("agent.tool_call", tool=tc.function.name, arguments=tc.function.arguments)
        res = agent_tools.dispatch(
            tc.function.name,
            tc.function.arguments,
            extra=remote,
            k=k,
            use_rerank=use_rerank,
            gate_top=gate.top,
        )
        if res.meta.get("error_kind"):
            result.tool_errors[tc.function.name] = res.meta["error_kind"]
        calls.append([tc, res, res.content, res.meta.get("sources", [])])

    corpus = [
        c for c in calls
        if c[0].function.name == agent_tools.CORPUS_TOOL and not c[1].meta.get("error_kind")
    ]
    # one verdict per turn: the system prompt asks for a search per subject, and two
    # searches must not produce two contradicting notices
    verdict = _verdict([s for c in corpus for s in c[3]], gate) if corpus else None
    # the axis judges the question, the gate judges the agent's query: once the axis says the
    # question is not ours, a lucky rewrite must not hand the context back
    if gate.off_topic and corpus:
        verdict = FallbackReason.off_topic
    if verdict in (FallbackReason.weak, FallbackReason.off_topic) and gate.drop_weak_context:
        # the 8b anchors on whatever sits in context, weak chunks included
        log.info("agent.weak_context_dropped", hop=result.hops)
        for call in corpus:
            result.dropped_sources.extend(s.source for s in call[3])
            result.dropped_hits.extend(call[3])
            call[2], call[3] = chat.NO_RESULTS, []
    if verdict and gate.announce:
        # a system message mid-conversation breaks the llama3.1 template
        notice = prompt_repo.active_template(Purpose.agent_fallback)
        last = corpus[-1]
        last[2] = f"{last[2]}\n\n{notice.replace('{tools}', gate.tool_signatures)}"
        result.fallback_announced = True
    if verdict and gate.off_topic:
        verdict = FallbackReason.off_topic
    if verdict and result.fallback_reason == FallbackReason.none:
        result.fallback_reason = verdict

    for tc, _res, content, sources in calls:
        result.sources.extend(sources)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    return False


# 13ms against a 6s answer, and a cached one would keep lying after a re-index
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
        and not m["content"].startswith(errors.ERROR_PREFIX)
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
                    "k": k or config.settings.retrieval.results_limit,
                    "max_hops": max_hops or config.settings.agent.max_hops,
                    "corpus": config.settings.corpus.description,
                    "corpus_fingerprint": _corpus_fingerprint(),
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
