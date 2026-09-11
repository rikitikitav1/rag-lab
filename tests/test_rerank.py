import engines
import llm
import pytest
import rerank
from engines import vllm
from models.registry import EngineKind, Placement

RERANK = engines.EngineSpec(7, "vllm-rerank", EngineKind.vllm, "VLLM_RERANK", Placement.gpu)


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
    with pytest.raises(RuntimeError, match="only a vLLM pooling server"):
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
    gate.refuse_unfit_model(Role.reranking, "BAAI/r")


def test_a_door_that_reranks_waits_for_the_reranker_s_engine_too(monkeypatch):
    import mcp_server
    from api.v1 import chat as door

    assert door._reranking(True) == ("reranking",) and door._reranking(False) == ()
    monkeypatch.setattr(door.chat.config.settings.rerank, "enabled", True)
    assert door._reranking(None) == ("reranking",), "the default is the config's"
    monkeypatch.setattr(door.chat.config.settings.rerank, "enabled", False)
    assert mcp_server._reranking() == ()
    assert mcp_server._reranking(gated=True) == ("reranking",), "the agent's gate scores too"


def test_the_config_seats_the_cross_encoder_on_its_own_engine():
    import config
    import seed

    role = config.settings.llm.roles["reranking"]
    assert (role.model, role.engine) == ("BAAI/bge-reranker-v2-m3", "vllm-rerank")
    assert seed.SEEDED_VLLM_RERANK["name"] == role.engine


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
