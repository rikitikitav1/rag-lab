"""What actually leaves for the server, recorded before the engine layer and compared after.

A replay proves the graph is intact and never enters `llm.py`: it hands back recorded turns. So the
only honest acceptance for moving the client behind an engine is equality of the request itself,
address included, which is why this intercepts the transport rather than the function.
"""

import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "fixtures" / "wire_before_engines.json"


def _recorded(monkeypatch):
    import httpx
    import llm
    from openai import OpenAI

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({
            "url": str(request.url),
            "body": json.loads(request.content) if request.content else None,
        })
        path = request.url.path
        if path.endswith("/embeddings"):
            payload = {"data": [{"embedding": [0.0, 1.0], "index": 0}], "usage": {}}
        else:
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return httpx.Response(200, json=payload)

    client = OpenAI(
        base_url=f"{llm.LLM_BASE}/v1",
        api_key="ollama",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "resolve_name", lambda role: {
        "generation": "llama3.1:8b",
        "judging": "qwen2.5:7b",
        "embedding": "bge-m3",
        "paraphrasing": "llama3.1:8b",
    }[role])
    return seen


def _calls(monkeypatch) -> list:
    import llm

    seen = _recorded(monkeypatch)
    llm.ask("system prompt", "user question", role="generation")
    llm.ask("judge prompt", "judge input", role="judging",
            schema={"type": "object", "properties": {"score": {"type": "integer"}}})
    llm.chat(
        [{"role": "user", "content": "with tools"}],
        tools=[{"type": "function", "function": {"name": "search_corpus", "parameters": {}}}],
        role="generation",
    )
    llm.request_embeddings_batch(["first", "second"], role="embedding")
    return seen


def test_the_wire_matches_what_was_recorded_before_the_engine_layer(monkeypatch):
    # four roles, four shapes: a plain ask, an ask with a schema, a turn with tools, an embedding
    got = _calls(monkeypatch)
    assert len(got) == 4

    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=2, ensure_ascii=False, sort_keys=True), "utf-8")
        pytest.skip(f"recorded the baseline into {GOLDEN.name}, rerun to compare")

    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got == want, "the request changed: address, model, params or messages"


def test_the_recording_carries_the_address_and_not_only_the_body(monkeypatch):
    # the point of intercepting the transport: a patched `_complete` would prove nothing about where
    got = _calls(monkeypatch)
    assert all(one["url"].startswith("http") for one in got)
    assert got[0]["url"].endswith("/v1/chat/completions")
    assert got[-1]["url"].endswith("/v1/embeddings")
