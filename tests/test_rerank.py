import engines
import llm
import pytest
import rerank
from engines import vllm
from models.registry import EngineKind, Placement
from stand_specs import VLLM_RERANK as RERANK


def test_rerank_orders_by_score_and_takes_top(monkeypatch):
    rows = [("doc a",), ("doc b",), ("doc c",)]
    monkeypatch.setattr(llm, "score_pairs", lambda pairs: [0.1, 0.9, 0.5])
    assert rerank.rerank("q", rows, top=2) == [(("doc b",), 0.9), (("doc c",), 0.5)]


def test_rerank_empty_rows():
    assert rerank.rerank("q", [], top=3) == []


def test_the_device_is_where_the_role_s_engine_sits(monkeypatch):
    # the record keeps its word: a run says whether it reranked on the card or on the processor
    spec = [RERANK]
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved("BAAI/r", spec[0]))
    assert rerank.device() == "cuda"
    spec[0] = engines.EngineSpec(8, "vllm-cpu", EngineKind.vllm, "VLLM_CPU", Placement.cpu)
    assert rerank.device() == "cpu"


def test_the_cross_encoder_is_a_role_that_takes_the_card_and_asks_its_engine(monkeypatch):
    asked, carded = [], []
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved("BAAI/r", RERANK))
    monkeypatch.setattr(llm, "_before_call", lambda spec, name: carded.append(spec.name))
    monkeypatch.setattr(vllm, "score", lambda spec, model, pairs: asked.append(pairs) or [0.3])
    assert llm.score_pairs([("q", "d")]) == [0.3]
    assert carded == ["vllm-rerank"] and asked == [[("q", "d")]]
    # the server refuses an empty list with a 400: nothing is sent and no card changes hands
    assert llm.score_pairs([]) == [] and len(carded) == 1


def test_only_a_vllm_scores_pairs(monkeypatch):
    ollama = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(llm, "resolve", lambda role: engines.Resolved("x", ollama))
    # every row would meet the same wrong kind, so the run stops rather than failing each row
    with pytest.raises(llm.StandFault, match="only a vLLM pooling server"):
        llm.score_pairs([("q", "d")])


def test_the_score_client_pairs_in_order_and_in_bounded_requests(monkeypatch):
    monkeypatch.setenv("VLLM_RERANK_BASE_URL", "http://vllm-rerank:8000")
    sent = []

    class _Reply:
        def __init__(self, n):
            # the server answers in any order; the index puts each score back on its pair
            self.body = {"data": [{"index": i, "score": float(i)} for i in reversed(range(n))]}

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def post(url, json, **kw):
        sent.append((url, json))
        return _Reply(len(json["text_1"]))

    monkeypatch.setattr(vllm.requests, "post", post)
    monkeypatch.setattr(vllm, "SCORE_BATCH", 2)
    pairs = [("q1", "a"), ("q2", "b"), ("q3", "c")]
    assert vllm.score(RERANK, "BAAI/r", pairs) == [0.0, 1.0, 0.0]
    assert [s[1]["text_1"] for s in sent] == [["q1", "q2"], ["q3"]]
    assert sent[0][0].endswith("/score") and sent[0][1]["text_2"] == ["a", "b"]


def test_the_role_gate_keeps_the_cross_encoder_on_a_vllm(monkeypatch):
    from models.registry import Role
    from use_cases import model_acceptance as gate

    ollama = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(gate, "_engine_of", lambda name: ollama)
    with pytest.raises(ValueError, match="cannot rerank"):
        gate.refuse_unfit_model(Role.reranking, "bge-reranker")
    # the bootstrap seats the role on a row its own session has not committed yet
    monkeypatch.setattr(gate, "_engine_of", lambda name: None)
    gate.refuse_unfit_model(Role.reranking, "BAAI/r")
    monkeypatch.setattr(gate, "_engine_of", lambda name: RERANK)
    monkeypatch.setattr(vllm, "artifact_of", lambda repo: {"quant": "F16"})
    monkeypatch.setattr(vllm, "card_state", lambda spec: "asleep")
    monkeypatch.setattr(vllm, "pools", lambda spec: True)
    gate.refuse_unfit_model(Role.reranking, "BAAI/r")


def test_one_rule_says_when_an_answer_needs_the_reranker(monkeypatch):
    # four copies of the gate rule, and the new ones forgot that `either` scores too
    import config
    from use_cases import card_wait

    monkeypatch.setattr(config.settings.rerank, "enabled", False)
    monkeypatch.setattr(config.settings.agent, "fallback_policy", "corpus_first_weak")
    assert card_wait.reranker_needed(True) and not card_wait.reranker_needed(False)
    for signal, needed in (("cross_encoder", True), ("either", True), ("distance", False)):
        monkeypatch.setattr(config.settings.agent.gate, "signal", signal)
        assert card_wait.reranker_needed(agent=True) is needed, signal
        assert not card_wait.reranker_needed(), "the chat has no gate"
    # the gate only scores under the weak policy, so a request's own policy decides
    assert not card_wait.reranker_needed(agent=True, fallback_policy="corpus_first")
    assert card_wait.answering_roles(agent=True) == ("embedding", "generation")
    # a run sweeps the signal, and its own value decides over the config's
    assert card_wait.reranker_needed(agent=True, gate_signal="cross_encoder")
    monkeypatch.setattr(config.settings.agent.gate, "signal", "either")
    assert not card_wait.reranker_needed(agent=True, gate_signal="distance")
    monkeypatch.setattr(config.settings.rerank, "enabled", True)
    assert card_wait.answering_roles() == ("embedding", "generation", "reranking")


def test_every_answering_door_waits_for_the_card(monkeypatch):
    # the REST agent was the one door left without the guard
    import mcp_server
    from api.v1 import agent as agent_door
    from api.v1 import chat as chat_door
    from fastapi import HTTPException
    from use_cases import card_wait

    def busy(*roles):
        raise card_wait.CardHeld("the card is held by the judge on vllm", 5)

    monkeypatch.setattr(card_wait, "wait_for_the_card", busy)
    with pytest.raises(HTTPException) as refused:
        agent_door.ask(agent_door.AgentRequest(text="what is redis"))
    assert refused.value.status_code == 503
    with pytest.raises(HTTPException):
        chat_door.ask(chat_door.QuestionRequest(text="what is redis"))
    with pytest.raises(mcp_server.ToolError):
        mcp_server.answer_question("what is redis")

    # the stand says what went wrong; each door says it its own way
    def split(*roles):
        raise card_wait.CannotAnswer("this answer needs two engines of the card (ollama, vllm)")

    monkeypatch.setattr(card_wait, "wait_for_the_card", split)
    with pytest.raises(HTTPException) as refused:
        chat_door.ask(chat_door.QuestionRequest(text="what is redis"))
    assert refused.value.status_code == 409 and not refused.value.headers
    with pytest.raises(mcp_server.ToolError, match="two engines"):
        mcp_server.answer_question("what is redis")


def test_the_config_seats_the_cross_encoder_on_its_own_engine():
    import config

    role = config.settings.llm.roles["reranking"]
    assert (role.model, role.engine) == ("BAAI/bge-reranker-v2-m3", "vllm-rerank")
    assert role.engine in {engine.name for engine in config.settings.engines}


def test_a_vllm_under_a_profile_is_registered_from_intact_weights(monkeypatch):
    # it starts after the bootstrap, so asking it would leave the role unseated on every clean start
    import bootstrap

    def down(spec):
        raise ConnectionError("refused")

    added = []
    session = type("S", (), {"add": lambda self, m: added.append(m), "flush": lambda self: None})()
    monkeypatch.setattr(bootstrap.vllm, "served", down)
    intact = [False]
    monkeypatch.setattr(bootstrap.vllm, "weights_intact", lambda name: intact[0])
    assert bootstrap._register_what_vllm_serves(session, RERANK, "reranking", "BAAI/r") is None
    intact[0] = True
    model = bootstrap._register_what_vllm_serves(session, RERANK, "reranking", "BAAI/r")
    assert model.name == "BAAI/r" and added == [model]
