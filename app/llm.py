import os
from dataclasses import dataclass
from typing import Any

import config
import logging_setup
import requests
from models.registry import Model, ModelRole
from openai import OpenAI, OpenAIError
from orm.sync_db import Session
from sqlalchemy import select

LLM_BASE = config.settings.llm.base_url

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

_client = OpenAI(
    base_url=f"{LLM_BASE}/v1",
    api_key="ollama",
    timeout=LLM_TIMEOUT,
    max_retries=1,
)

log = logging_setup.get_logger(__name__)


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ChatTurn:
    text: str | None
    tool_calls: list
    message: Any
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None


def resolve_name(role: str) -> str:
    with Session() as session:
        name = session.scalar(
            select(Model.name)
            .join(ModelRole, ModelRole.model_id == Model.id)
            .where(ModelRole.role == role)
        )
    if name is None:
        raise RuntimeError(f"no model assigned to role {role}")
    return name


def ask(system, user, role="generation", schema=None, model=None) -> Completion:
    name = model or resolve_name(role)
    try:
        resp = _client.chat.completions.create(
            model=name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **_params(role, schema),
        )
    except OpenAIError as e:
        log.error("llm.chat_failed", model=name, error=str(e))
        raise RuntimeError(f"LLM chat failed ({name}): {e}") from e

    usage = resp.usage
    log.info(
        "llm.chat",
        model=name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    return Completion(
        text=resp.choices[0].message.content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


def chat(messages, tools=None, role="generation", model=None) -> ChatTurn:
    name = model or resolve_name(role)
    params = _params(role, None)
    if tools:
        params["tools"] = tools
    try:
        resp = _client.chat.completions.create(
            model=name,
            messages=messages,
            **params,
        )
    except OpenAIError as e:
        log.error("llm.chat_failed", model=name, error=str(e))
        raise RuntimeError(f"LLM chat failed ({name}): {e}") from e

    choice = resp.choices[0]
    message = choice.message
    usage = resp.usage
    log.info(
        "llm.chat_tools",
        model=name,
        tool_calls=len(message.tool_calls or []),
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    return ChatTurn(
        text=message.content,
        tool_calls=message.tool_calls or [],
        message=message,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        finish_reason=choice.finish_reason,
    )


def _params(role, schema) -> dict:
    opts = config.settings.llm.roles[role].options
    params = {k: opts[k] for k in ("temperature", "max_tokens") if k in opts}
    if schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": schema},
        }
    return params


def embed(prompt, role="embedding"):
    return request_embeddings_batch([prompt], role)[0]


# the server reports the window it actually loaded, which is not always the one in our config
def server_context_length(model: str) -> int | None:
    try:
        loaded = _get_request("/api/ps").get("models") or []
    except Exception as e:  # a probe must not break a run
        log.warning("llm.ps_failed", error=str(e))
        return None
    # the full tag, or llama3.1:8b would read the window of a loaded llama3.1:70b
    wanted = {model, f"{model}:latest"}
    for entry in loaded:
        if entry.get("name") in wanted:
            return entry.get("context_length")
    return None


def list_models():
    return [m["name"] for m in _get_request("/api/tags")["models"]]


def residency() -> list[dict]:
    try:
        loaded = _get_request("/api/ps").get("models") or []
    except Exception as e:  # a probe must not break a run
        log.warning("llm.ps_failed", error=str(e))
        return []
    return [
        {
            "model": m.get("name", "?"),
            "size_mb": round((m.get("size") or 0) / 2**20),
            "vram_mb": round((m.get("size_vram") or 0) / 2**20),
        }
        for m in loaded
        if m.get("size")
    ]


# a spill makes the run slower and never louder; nothing in vram at all means the card is gone
def warn_if_models_do_not_fit() -> list[str]:
    spilled, off_card = [], []
    for entry in residency():
        if entry["vram_mb"] >= entry["size_mb"]:
            continue
        spilled.append(entry["model"])
        if entry["vram_mb"] == 0:
            off_card.append(entry["model"])
        log.warning(
            "llm.model_spilled_to_cpu",
            **entry,
            context=config.settings.llm.context_length,
        )
    if off_card:
        log.error("llm.gpu_unavailable", models=off_card)
    return spilled


def models_off_the_card() -> list[str]:
    return [e["model"] for e in residency() if e["vram_mb"] == 0]


def request_embeddings_batch(texts, role="embedding"):
    name = resolve_name(role)
    try:
        resp = _client.embeddings.create(model=name, input=texts)
    except OpenAIError as e:
        log.error("llm.embed_failed", model=name, error=str(e))
        raise RuntimeError(f"LLM embed failed ({name}): {e}") from e
    log.info("llm.embed", model=name, count=len(texts))
    return [d.embedding for d in resp.data]


_HTTP_TIMEOUT = 60
_PULL_TIMEOUT = 3600


def pull_model(model):
    return _post_request("/api/pull", {"model": model, "stream": False}, timeout=_PULL_TIMEOUT)


def unload(role="embedding", model=None):
    # frees VRAM between eval phases: keep_alive 0 overrides the server default
    name = model or resolve_name(role)
    try:
        _post_request("/api/generate", {"model": name, "keep_alive": 0})
    except Exception as e:
        log.warning("llm.unload_failed", model=name, error=str(e))


# an empty generate loads the weights and answers nothing
def load_into_memory(role="generation", model=None) -> dict:
    name = model or resolve_name(role)
    _post_request("/api/generate", {"model": name})
    log.info("llm.loaded", model=name)
    return {"model": name, "context_length": server_context_length(name)}


def delete_model(model):
    response = requests.delete(
        f"{LLM_BASE}/api/delete", json={"model": model}, timeout=_HTTP_TIMEOUT
    )
    if response.status_code == 404:
        log.info("llm.model_already_absent", model=model)
        return None
    return _check(response, "/api/delete")


def add_tags(models) -> list:
    return [m if ":" in m else f"{m}:latest" for m in models]


def ensure_models() -> None:
    for model in set(add_tags(config.settings.llm.pull_models)) - set(
        add_tags(list_models())
    ):
        pull_model(model)


def _check(response, path) -> Any:
    if not response.ok:
        try:
            error = response.json().get("error", response.text)
        except ValueError:
            error = response.text

        raise RuntimeError(f"Ollama {response.status_code} on {path}: {error}")

    if not response.text:
        return None

    return response.json()


def _post_request(path, payload, timeout=_HTTP_TIMEOUT):
    return _check(requests.post(f"{LLM_BASE}{path}", json=payload, timeout=timeout), path)


def _get_request(path) -> dict:
    return _check(requests.get(f"{LLM_BASE}{path}", timeout=_HTTP_TIMEOUT), path)
