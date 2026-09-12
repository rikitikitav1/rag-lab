# what leaves for the server, recorded before the engine layer and compared after; why: the arc log
import json
from pathlib import Path

import engines
import httpx
import llm
from engines import core
from stand_specs import OLLAMA as SPEC

GOLDEN = Path(__file__).parent / "fixtures" / "wire_before_engines.json"

NAMES = {
    "generation": "llama3.1:8b",
    "judging": "qwen2.5:7b",
    "embedding": "bge-m3",
    "paraphrasing": "llama3.1:8b",
}


def _answer(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json={"data": [{"embedding": [0.0, 1.0], "index": 0}],
                                         "usage": {}})
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })


def _recorded(monkeypatch) -> list:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({
            "url": str(request.url),
            "body": json.loads(request.content) if request.content else None,
        })
        return _answer(request)

    # only the transport is replaced: the address still comes out of `engines.base_url`
    built = core.OpenAI

    def with_fake_transport(**kwargs):
        return built(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(core, "OpenAI", with_fake_transport)
    engines.forget_clients()
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved(NAMES[role], SPEC))
    return seen


def _calls(monkeypatch) -> list:
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
    engines.forget_clients()
    return seen


def test_the_wire_matches_what_was_recorded_before_the_engine_layer(monkeypatch):
    # four roles, four shapes: a plain ask, an ask with a schema, a turn with tools, an embedding
    got = _calls(monkeypatch)
    assert len(got) == 4
    # a missing fixture is a lost guard, not a first run: re-recording it silently proves nothing
    assert GOLDEN.exists(), f"{GOLDEN.name} is gone; restore it rather than re-recording"

    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got == want, "the request changed: address, model, params or messages"


def test_the_recording_carries_the_address_and_not_only_the_body(monkeypatch):
    # the point of intercepting the transport: a patched `_complete` would prove nothing about where
    got = _calls(monkeypatch)
    assert all(one["url"].startswith("http") for one in got)
    assert got[0]["url"].endswith("/v1/chat/completions")
    assert got[-1]["url"].endswith("/v1/embeddings")


def test_the_address_is_the_one_the_engine_layer_builds(monkeypatch):
    # the fixture used to hold a url the test itself had formatted, so `base_url` never ran
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert engines.base_url(SPEC) == llm.LLM_BASE
    recorded = json.loads(GOLDEN.read_text(encoding="utf-8"))[0]["url"]
    assert recorded.startswith(f"{engines.base_url(SPEC)}/v1")


# the golden fixture records url and body only, so a changed key would have passed it unseen
def test_the_authorization_header_is_the_one_the_engine_layer_builds(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return _answer(request)

    built = core.OpenAI
    monkeypatch.setattr(core, "OpenAI", lambda **kw: built(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ))
    engines.forget_clients()
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved(NAMES[role], SPEC))
    llm.ask("s", "u", role="generation")
    engines.forget_clients()

    # the literal the wire carried before the layer, not what the code says today
    assert seen["auth"] == "Bearer ollama"
