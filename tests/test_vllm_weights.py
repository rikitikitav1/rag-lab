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
        return root

    return lay


def test_the_weights_on_disk_say_their_quant_and_size(cache):
    cache("Qwen/Q-AWQ", {"quantization_config": {"quant_method": "awq"}, "torch_dtype": "float16"})
    assert vllm.artifact_of("Qwen/Q-AWQ") == {"quant": "AWQ", "size_bytes": 350}
    cache("Qwen/Q", {"torch_dtype": "bfloat16"})
    assert vllm.artifact_of("Qwen/Q")["quant"] == "BF16", "unquantised says its dtype"
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
    vllm.pull_weights("Qwen/Q")
    assert seen["env"]["HF_HUB_OFFLINE"] == "0", "the worker's offline switch would refuse the pull"
    assert seen["env"]["HF_HOME"] == vllm.weights_cache()
    assert seen["cmd"][-1] == "Qwen/Q"
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
        model_ops._refuse_remote(CLOUD, "gpt")
    model_ops._refuse_remote(VLLM, "Qwen/Q")


def test_weights_a_running_server_reads_are_not_deleted_under_it(monkeypatch):
    from job_handlers import model_ops

    monkeypatch.setattr(model_ops.vllm, "served", lambda spec: ["Qwen/Q"])
    with pytest.raises(ValueError, match="served by vllm"):
        model_ops._refuse_if_served(VLLM, "Qwen/Q")
    model_ops._refuse_if_served(VLLM, "Qwen/Other")

    def down(spec):
        raise OSError("connection refused")

    monkeypatch.setattr(model_ops.vllm, "served", down)
    model_ops._refuse_if_served(VLLM, "Qwen/Q")


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

    (root / "blobs" / "abc.incomplete").write_bytes(b"half")
    assert "abc.incomplete" in vllm.broken_weights("Qwen/Q")
    assert not vllm.weights_intact("Qwen/Absent"), "no snapshot is not intact"


def test_a_pull_skips_intact_weights_and_drops_broken_ones_before_fetching(tmp_path, monkeypatch):
    root, names = _real_cache(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr(vllm.subprocess, "run",
                        lambda *a, **kw: ran.append(a) or SimpleNamespace(returncode=0, stderr=""))
    vllm.pull_weights("Qwen/Q")
    assert ran == [], "intact weights are not fetched again"

    big = root / "blobs" / names["model.safetensors"][0]
    big.write_bytes(b"x" * 5000)
    vllm.pull_weights("Qwen/Q")
    assert len(ran) == 1 and not big.exists(), "a broken blob keeps its name, so it must go first"
