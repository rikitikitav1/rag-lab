import time
from dataclasses import asdict, dataclass, field

import agent_tools
import config
import job_queue
import llm
import logging_setup
import prompt_repo
from models.eval import QuestionLog
from models.registry import Purpose
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
    elapsed: float = 0.0


def run(
    question: str,
    role: str = "generation",
    max_hops: int = config.settings.agent.max_hops,
    run_name: str | None = None,
    language: str | None = None,
) -> AgentResult:
    start = time.perf_counter()
    system = prompt_repo.active_template(Purpose.agent_system)
    if language:
        system += f"\n\n{chat._language_directive(language)}"
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    result = AgentResult(messages=messages)
    tools = agent_tools.schemas()

    for hop in range(1, max_hops + 1):
        result.hops = hop
        turn = llm.chat(messages, tools=tools, role=role)
        result.prompt_tokens += turn.prompt_tokens
        result.completion_tokens += turn.completion_tokens
        if _apply_turn(turn, messages, result):
            break

    if not result.success:
        log.warning("agent.hops_exhausted", hops=result.hops)
        final = llm.chat(messages, role=role)
        result.prompt_tokens += final.prompt_tokens
        result.completion_tokens += final.completion_tokens
        result.text = final.text or ""
        result.success = bool(result.text)

    result.elapsed = round(time.perf_counter() - start, 3)
    log.info(
        "agent.done",
        hops=result.hops,
        success=result.success,
        sources=len(result.sources),
    )
    try:
        _log_answer(question, result, run_name, language)
    except SQLAlchemyError as e:
        log.error("agent_log.insert_failed", reason=str(e))
    return result


def _apply_turn(turn: llm.ChatTurn, messages: list[dict], result: AgentResult) -> bool:
    if not turn.tool_calls:
        result.text = turn.text or ""
        result.success = bool(result.text)
        if turn.finish_reason == "length":
            log.warning("agent.truncated", hops=result.hops)
        return True

    messages.append(turn.message)

    for tc in turn.tool_calls:
        res = agent_tools.dispatch(tc.function.name, tc.function.arguments)
        result.sources.extend(res.meta.get("sources", []))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.content})

    return False


def _context_from_messages(messages) -> str:
    return "\n\n".join(
        m["content"]
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )


def _log_answer(
    question_text: str,
    result: AgentResult,
    run_name: str | None,
    language: str | None = None,
) -> None:
    lang = chat._resolve_language(question_text, language)
    with Session() as session:
        question = chat._find_or_create_question(session, question_text, lang)
        log_row = QuestionLog(
            run_name=run_name,
            question_id=question.id,
            answered=result.success,
            answer=result.text,
            context=_context_from_messages(result.messages),
            sources=[asdict(s) for s in result.sources],
            pipeline="agent",
            models={
                "generation": llm.resolve_name("generation"),
                "embedding": llm.resolve_name("embedding"),
            },
            prompts={"agent_system": prompt_repo.active_version(Purpose.agent_system)},
            metrics={"hops": result.hops},
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            elapsed=result.elapsed,
        )
        session.add(log_row)
        session.commit()
        log_id = log_row.id

    if result.success and run_name is None:
        job_queue.enqueue("judge_answers", {"log_ids": [log_id]})
