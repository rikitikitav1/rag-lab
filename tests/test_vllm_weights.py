import json
from types import SimpleNamespace

import engines
import pytest
from engines import vllm
from models.registry import EngineKind, Placement

VLLM = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
CLOUD = engines.EngineSpec(9, "cloud", EngineKind.openai_compatible, "CLOUD", Placement.remote)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGINE_HF_CACHE", str(tmp_path))

    def lay(repo, config, blobs=(100, 250)):
        root = tmp_path / "hub" / ("models--" + repo.replace("/", "--"))
        snap = root / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(config))
        (root / "blobs").mkdir()
        for i, n in enumerate(blobs):
            (root / "blobs" / f"b{i}").write_bytes(b"x" * n)
            (snap / f"w{i}.safetensors").symlink_to(f"../../blobs/b{i}")
        # a blob of an older revision the snapshot no longer points at
        (root / "blobs" / "old").write_bytes(b"x" * 999)
        return root

    return lay


def test_the_weights_on_disk_say_their_quant_and_size(cache):
    config = {"quantization_config": {"quant_method": "awq"}, "torch_dtype": "float16"}
    cache("Qwen/Q-AWQ", config)
    # every blob of every revision was summed, an update counted twice
    assert vllm.artifact_of("Qwen/Q-AWQ") == {"quant": "AWQ", "size_bytes": 350 + len(json.dumps(config))}
    root = cache("Qwen/Q", {"torch_dtype": "bfloat16"})
    assert vllm.artifact_of("Qwen/Q")["quant"] == "BF16", "unquantised says its dtype"
    # files in subdirectories count, the directories themselves do not
    pooling = root / "snapshots" / "abc" / "1_Pooling"
    pooling.mkdir()
    (pooling / "config.json").write_text("{}")
    config_size = len(json.dumps({"torch_dtype": "bfloat16"}))
    assert vllm.artifact_of("Qwen/Q")["size_bytes"] == 350 + config_size + 2
    assert vllm.artifact_of("Qwen/absent") == {}, "no weights on disk, nothing to claim"


def test_the_hub_names_the_size_before_the_first_byte(monkeypatch):
    reply = SimpleNamespace(json=lambda: {"siblings": [{"size": 5}, {"size": 7}, {}]})
    monkeypatch.setattr(vllm.requests, "get", lambda *a, **kw: reply)
    assert vllm.repo_size("Qwen/Q") == 12

    def down(*a, **kw):
        raise OSError("TLS handshake timeout")

    monkeypatch.setattr(vllm.requests, "get", down)
    assert vllm.repo_size("Qwen/Q") is None, "an unknown size promises nothing"


def test_a_pull_goes_online_for_itself_only_and_lands_in_the_engines_cache(cache, monkeypatch):
    seen = {}

    def run(cmd, env, **kw):
        seen.update(cmd=cmd, env=env)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(vllm.subprocess, "run", run)
    monkeypatch.setattr(vllm, "_siblings", lambda repo: [{"rfilename": "model.safetensors"}])
    vllm.pull_weights("Qwen/Q")
    assert seen["env"]["HF_HUB_OFFLINE"] == "0", "the worker's offline switch would refuse the pull"
    assert seen["env"]["HF_HOME"] == vllm.weights_cache()
    assert seen["cmd"][-2] == "Qwen/Q"
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1", "the rest of the worker stays offline"

    monkeypatch.setattr(vllm.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=1, stderr="boom\nTLS timeout"))
    with pytest.raises(RuntimeError, match="TLS timeout"):
        vllm.pull_weights("Qwen/Q")


def test_deleting_weights_removes_the_repo_and_an_absent_one_is_not_an_error(cache):
    root = cache("Qwen/Q", {})
    vllm.delete_weights("Qwen/Q")
    assert not root.exists()
    vllm.delete_weights("Qwen/Q")


def test_pull_and_delete_ask_the_engine_the_model_sits_on(monkeypatch):
    from job_handlers import model_ops

    calls = []
    monkeypatch.setattr(model_ops, "_pair", lambda name, engine_id: engines.Resolved(name, VLLM))
    monkeypatch.setattr(model_ops, "_size_seen_before", lambda found: 10)
    monkeypatch.setattr(model_ops.engines, "refuse_if_tight",
                        lambda size, name, path=None: calls.append(("disk", path)))
    monkeypatch.setattr(model_ops.vllm, "pull_weights", lambda repo: calls.append(("pull", repo)))
    monkeypatch.setattr(model_ops.ollama, "pull_model", lambda *a: calls.append("ollama pull"))
    monkeypatch.setattr(model_ops, "record_what_the_server_holds", lambda found: None)
    model_ops.pull_llm_model({"name": "Qwen/Q", "engine_id": 3})
    assert calls == [("disk", vllm.weights_cache()), ("pull", "Qwen/Q")]

    with pytest.raises(engines.NotSupported):
        model_ops.refuse_remote(CLOUD.kind, "gpt")
    model_ops.refuse_remote(VLLM.kind, "Qwen/Q")


def test_weights_a_running_server_reads_are_not_deleted_under_it(monkeypatch):
    # every vLLM service reads the one host cache, and a silent server is not a free one
    import engines
    import requests
    from engines import vllm

    rerank = engines.EngineSpec(7, "vllm-rerank", EngineKind.vllm, "VLLM_RERANK", Placement.gpu)
    ollama = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
    monkeypatch.setattr(engines.lookup, "registered", lambda: [ollama, VLLM, rerank])
    serving = {"vllm": ["Qwen/Other"], "vllm-rerank": ["Qwen/Q"]}
    monkeypatch.setattr(vllm, "served", lambda spec: serving[spec.name])
    with pytest.raises(vllm.StillServed, match="served by vllm-rerank"):
        vllm.refuse_if_served("Qwen/Q")
    vllm.refuse_if_served("Qwen/Absent")

    def refused(spec):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(vllm, "served", refused)
    vllm.refuse_if_served("Qwen/Q")

    def silent(spec):
        raise requests.Timeout("10 s")

    monkeypatch.setattr(vllm, "served", silent)
    with pytest.raises(vllm.StillServed, match="did not answer"):
        vllm.refuse_if_served("Qwen/Q")


def test_a_refused_delete_keeps_the_row_that_names_the_weights(monkeypatch):
    # the row went first, the refusal after, and the weights stayed with no row
    from engines import vllm
    from job_handlers import model_ops

    steps = []
    monkeypatch.setattr(model_ops, "_engine_for_delete", lambda name, engine_id: VLLM)
    monkeypatch.setattr(model_ops, "refuse_if_the_weights_are_shared", lambda n, engine_id: None)

    def served(name):
        raise vllm.StillServed("served by vllm")

    monkeypatch.setattr(model_ops.vllm, "refuse_if_served", served)
    monkeypatch.setattr(model_ops, "Session", lambda: pytest.fail("the row was touched first"))
    with pytest.raises(vllm.StillServed):
        model_ops.delete_llm_model({"name": "Qwen/Q", "engine_id": 3})
    assert steps == []


def _real_cache(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("ENGINE_HF_CACHE", str(tmp_path))
    root = tmp_path / "hub" / "models--Qwen--Q"
    snap = root / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (root / "blobs").mkdir()
    big, small = b"w" * 5000, b'{"torch_dtype": "float16"}'
    names = {
        "model.safetensors": (hashlib.sha256(big).hexdigest(), big),
        "config.json": (hashlib.sha1(b"blob %d\0" % len(small) + small).hexdigest(), small),
    }
    for file, (blob, data) in names.items():
        (root / "blobs" / blob).write_bytes(data)
        (snap / file).symlink_to(f"../../blobs/{blob}")
    return root, names


def test_weights_are_checked_by_the_hash_the_hub_named_them_by(tmp_path, monkeypatch):
    root, names = _real_cache(tmp_path, monkeypatch)
    assert vllm.broken_weights("Qwen/Q") == [] and vllm.weights_intact("Qwen/Q")

    big = root / "blobs" / names["model.safetensors"][0]
    big.write_bytes(b"x" + big.read_bytes()[1:])
    assert vllm.broken_weights("Qwen/Q") == [big.name], "one byte off is a broken file"

    small = root / "blobs" / names["config.json"][0]
    small.write_bytes(b'{"torch_dtype": "float32"}')
    assert small.name in vllm.broken_weights("Qwen/Q"), "a small file is checked by its git sha1"
    small.unlink()
    assert "config.json" in vllm.broken_weights("Qwen/Q"), "a link whose blob is gone is missing"

    # a partial download counted as broken and was dropped, restarting gigabytes
    (root / "blobs" / "abc.incomplete").write_bytes(b"half")
    assert "abc.incomplete" not in vllm.broken_weights("Qwen/Q")
    assert not vllm.weights_intact("Qwen/Absent"), "no snapshot is not intact"
    assert vllm.weights_check("Qwen/Absent") == ["no snapshot"]


def test_a_pull_skips_intact_weights_and_drops_broken_ones_before_fetching(tmp_path, monkeypatch):
    root, names = _real_cache(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr(vllm.subprocess, "run",
                        lambda *a, **kw: ran.append(a) or SimpleNamespace(returncode=0, stderr=""))
    vllm.pull_weights("Qwen/Q")
    assert ran == [], "intact weights are not fetched again"

    big = root / "blobs" / names["model.safetensors"][0]
    big.write_bytes(b"x" * 5000)
    partial = root / "blobs" / "abc.incomplete"
    partial.write_bytes(b"half")
    hashed, at_the_pull = [], []
    real = vllm._intact
    monkeypatch.setattr(vllm, "_intact", lambda blob: hashed.append(blob.name) or real(blob))
    monkeypatch.setattr(vllm, "_siblings", lambda repo: None)
    monkeypatch.setattr(vllm.subprocess, "run", lambda *a, **kw: ran.append(a) or at_the_pull.append(
        partial.exists()) or SimpleNamespace(returncode=0, stderr=""))
    vllm.pull_weights("Qwen/Q")
    assert len(ran) == 1 and not big.exists(), "a broken blob keeps its name, so it must go first"
    assert at_the_pull == [True], "a partial download is kept for the resume"
    assert len(hashed) == len(set(hashed)), "each blob is hashed once per pull"
    assert "allow_patterns" in ran[0][0][2], "only what a server loads is pulled"


def test_a_pull_cut_short_is_not_whole_and_the_next_pull_resumes_it(tmp_path, monkeypatch):
    # a partial blob has no link yet, so the snapshot looked whole
    root, names = _real_cache(tmp_path, monkeypatch)
    partial = root / "blobs" / "abc.incomplete"
    at_the_pull = []
    monkeypatch.setattr(vllm.subprocess, "run", lambda *a, **kw: at_the_pull.append(
        partial.exists()) or SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(vllm, "_siblings", lambda repo: None)
    partial.write_bytes(b"half")
    assert vllm.weights_check("Qwen/Q") == ["abc.incomplete"]
    vllm.pull_weights("Qwen/Q")
    assert at_the_pull == [True], "resumed, not restarted"
    # a partial blob left after a finished pull is an orphan and goes
    assert not partial.exists() and vllm.weights_check("Qwen/Q") == [], "left after a pull, an orphan"

    (root / "snapshots" / "abc" / "model.safetensors").unlink()
    assert vllm.weights_check("Qwen/Q") == ["model.safetensors"], "a weights file never linked"

    snap = root / "snapshots" / "abc"
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"w")
    assert vllm.weights_check("Qwen/Q") == ["model-00002-of-00002.safetensors"], "every shard counts"

    (snap / "model.safetensors.index.json").unlink()
    (snap / "model-00001-of-00002.safetensors").unlink()
    (snap / "training_args.bin").write_bytes(b"t")
    assert vllm.weights_check("Qwen/Q") == ["model.safetensors"], "training args are not the weights"
    (snap / "adapter_model.bin").write_bytes(b"w")
    assert vllm.weights_check("Qwen/Q") == [], "weights under a name of their own still count"


def test_a_repo_without_safetensors_pulls_its_bin_and_one_with_both_pulls_one_format(monkeypatch):
    # bge-m3 ships only `pytorch_model.bin`
    monkeypatch.setattr(vllm, "_siblings", lambda repo: [{"rfilename": "pytorch_model.bin"}])
    assert "*.bin" in vllm._patterns("BAAI/bge-m3") and "*.safetensors" not in vllm._patterns("x")
    monkeypatch.setattr(vllm, "_siblings", lambda repo: [{"rfilename": "pytorch_model.bin"},
                                                         {"rfilename": "model.safetensors"}])
    assert "*.safetensors" in vllm._patterns("x") and "*.bin" not in vllm._patterns("x")
    monkeypatch.setattr(vllm, "_siblings", lambda repo: None)
    assert {"*.safetensors", "*.bin"} <= set(vllm._patterns("x")), "unlisted, neither is left behind"
