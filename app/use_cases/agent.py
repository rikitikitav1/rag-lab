import time
from dataclasses import asdict, dataclass, field

import agent_tools
import config
import errors
import job_queue
import llm
import logging_setup
import prompt_repo
from models.eval import QuestionLog
from models.registry import Pipeline, Purpose
from orm.sync_db import Session
from sqlalchemy.exc import SQLAlchemyError
from use_cases import chat

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
) -> AgentResult:
    start = time.perf_counter()
    if max_hops is None:
        max_hops = config.settings.agent.max_hops
    if use_rerank is None:
        use_rerank = config.settings.rerank.enabled
    system = prompt_repo.active_template(Purpose.agent_system)
    if language:
        system += f"\n\n{chat._language_directive(language)}"
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    result = AgentResult(messages=messages)
    remote = {t.name: t for t in agent_tools.remote_tools()}
    tools = agent_tools.schemas() + [t.schema() for t in remote.values()]

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
        if _apply_turn(turn, messages, result, k, use_rerank, remote):
            break

    result.sources = _unique_sources(result.sources)

    if not result.success and result.sources:
        log.info("agent.forcing_final", hops=result.hops)
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

    if result.sources:
        result.success = bool(result.text)
    else:
        result.success = False
        if not result.text:
            result.text = chat.NO_RESULTS

    result.elapsed = round(time.perf_counter() - start, 3)
    log.info(
        "agent.done",
        hops=result.hops,
        success=result.success,
        sources=len(result.sources),
    )
    try:
        mcp_names = sorted({name.split("__")[0] for name in remote})
        _log_answer(question, result, run_name, language, k, use_rerank, model, mcp_names)
    except SQLAlchemyError as e:
        log.error("agent_log.insert_failed", reason=str(e))
    return result


def _apply_turn(
    turn: llm.ChatTurn,
    messages: list[dict],
    result: AgentResult,
    k: int | None = None,
    use_rerank: bool | None = None,
    remote: dict | None = None,
) -> bool:
    if not turn.tool_calls:
        result.text = turn.text or ""
        result.success = bool(result.text)
        if turn.finish_reason == "length":
            log.warning("agent.truncated", hops=result.hops)
        return True

    messages.append(turn.message)

    for tc in turn.tool_calls:
        log.info("agent.tool_call", tool=tc.function.name, arguments=tc.function.arguments)
        res = agent_tools.dispatch(
            tc.function.name,
            tc.function.arguments,
            extra=remote,
            k=k,
            use_rerank=use_rerank,
        )
        result.sources.extend(res.meta.get("sources", []))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.content})

    return False


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
        and m["content"] != chat.NO_RESULTS
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
            prompts={"agent_system": prompt_repo.active_version(Purpose.agent_system)},
            metrics={
                "hops": result.hops,
                "no_evidence": not bool(result.sources),
                "context_tokens": result.max_prompt_tokens,
                "config": {
                    "rerank": use_rerank,
                    "distance_threshold": round(
                        config.settings.retrieval.distance_threshold, 3
                    ),
                    "k": k or config.settings.retrieval.results_limit,
                    "mcp": mcp_names or [],
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
