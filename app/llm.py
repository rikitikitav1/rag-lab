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

_client = OpenAI(base_url=f"{LLM_BASE}/v1", api_key="ollama")

log = logging_setup.get_logger(__name__)


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


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


def ask(system, user, role="generation", schema=None) -> Completion:
    name = resolve_name(role)
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


def list_models():
    return [m["name"] for m in _get_request("/api/tags")["models"]]


def request_embeddings_batch(texts, role="embedding"):
    name = resolve_name(role)
    try:
        resp = _client.embeddings.create(model=name, input=texts)
    except OpenAIError as e:
        log.error("llm.embed_failed", model=name, error=str(e))
        raise RuntimeError(f"LLM embed failed ({name}): {e}") from e
    log.info("llm.embed", model=name, count=len(texts))
    return [d.embedding for d in resp.data]


def pull_model(model):
    return _post_request("/api/pull", {"model": model, "stream": False})


def delete_model(model):
    response = requests.delete(f"{LLM_BASE}/api/delete", json={"model": model})
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


def _post_request(path, payload):
    return _check(requests.post(f"{LLM_BASE}{path}", json=payload), path)


def _get_request(path) -> dict:
    return _check(requests.get(f"{LLM_BASE}{path}"), path)
