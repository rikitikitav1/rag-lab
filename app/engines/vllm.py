import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import logging_setup
import requests

from .core import CardState, EngineSpec, Unconfigured, api_key, base_url

log = logging_setup.get_logger(__name__)

HTTP_TIMEOUT = 10
PULL_TIMEOUT = 3 * 3600
# a wake copies the weights back from pinned host memory; measured under a second, the rest is room
WAKE_TIMEOUT = 120

# the smallest request that says whether the server parses tool calls, since its flags are not served
_PROBE = {
    "messages": [{"role": "user", "content": "Search the corpus for redis persistence."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search the corpus.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }],
    "tool_choice": "auto",
    "max_tokens": 64,
    "temperature": 0,
}


class WakeFailed(RuntimeError):
    pass


def _url(spec: EngineSpec, path: str) -> str:
    return f"{base_url(spec)}{path}"


def _headers(spec: EngineSpec) -> dict:
    return {"Authorization": f"Bearer {api_key(spec)}"}


def served(spec: EngineSpec) -> list[str]:
    seen = requests.get(_url(spec, "/v1/models"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
    seen.raise_for_status()
    return [m["id"] for m in seen.json()["data"]]


# a cross-encoder answers pairs; bounded requests, since a run sends every candidate of every question
SCORE_BATCH = 256
# 256 pairs score in about two seconds on the card; the wake ceiling was never this call's clock
SCORE_TIMEOUT = 60


def score(spec: EngineSpec, model: str, pairs: list[tuple[str, str]]) -> list[float]:
    scores = []
    for start in range(0, len(pairs), SCORE_BATCH):
        chunk = pairs[start : start + SCORE_BATCH]
        seen = requests.post(
            _url(spec, "/score"),
            json={"model": model, "text_1": [q for q, _ in chunk], "text_2": [d for _, d in chunk]},
            headers=_headers(spec),
            timeout=SCORE_TIMEOUT,
        )
        seen.raise_for_status()
        scores.extend(d["score"] for d in sorted(seen.json()["data"], key=lambda d: d["index"]))
    return scores


# the window is a start flag: a longer prompt is refused with a 400, never cut as ollama cuts it
def max_model_len(spec: EngineSpec, model: str) -> int | None:
    try:
        seen = requests.get(_url(spec, "/v1/models"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
        seen.raise_for_status()
        return next((m.get("max_model_len") for m in seen.json()["data"] if m["id"] == model), None)
    except Exception as e:
        log.warning("vllm.window_unknown", engine=spec.name, error=str(e))
        return None


# a refused connection is a process that holds no card; a timeout is a server that may still hold it
def card_state(spec: EngineSpec) -> CardState:
    try:
        seen = requests.get(_url(spec, "/is_sleeping"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
        # no sleep routes without VLLM_SERVER_DEV_MODE: such a server is awake for good
        if seen.status_code == 404:
            log.warning("vllm.no_sleep_routes", engine=spec.name)
            return CardState.AWAKE
        seen.raise_for_status()
        return CardState.ASLEEP if seen.json()["is_sleeping"] else CardState.AWAKE
    except requests.Timeout as e:
        # a ConnectTimeout is a ConnectionError too, and a host that did not answer may hold the card
        log.warning("vllm.card_state_unknown", engine=spec.name, error=str(e))
        return CardState.UNKNOWN
    except (requests.ConnectionError, Unconfigured):
        return CardState.DOWN
    except Exception as e:
        log.warning("vllm.card_state_unknown", engine=spec.name, error=str(e))
        return CardState.UNKNOWN


# None when the server cannot say; one without sleep routes is awake, so False
def is_sleeping(spec: EngineSpec) -> bool | None:
    return {CardState.ASLEEP: True, CardState.AWAKE: False}.get(card_state(spec))


# None when nobody answers: a service under a profile may simply not be up yet
def has_sleep_routes(spec: EngineSpec) -> bool | None:
    try:
        seen = requests.get(_url(spec, "/is_sleeping"), headers=_headers(spec), timeout=3)
    except Exception:
        return None
    return seen.status_code != 404


# level 1 keeps the weights in host memory, so the next wake reads nothing from disk
def sleep(spec: EngineSpec, level: int = 1) -> None:
    seen = requests.post(
        _url(spec, "/sleep"), params={"level": level}, headers=_headers(spec), timeout=WAKE_TIMEOUT
    )
    if not seen.ok:
        raise RuntimeError(f"vLLM {spec.name} did not go to sleep: http {seen.status_code}")


# a wake on a card another engine still holds fails with CUDA OOM and leaves the server asleep
def wake_up(spec: EngineSpec) -> None:
    seen = requests.post(_url(spec, "/wake_up"), headers=_headers(spec), timeout=WAKE_TIMEOUT)
    if not seen.ok:
        raise WakeFailed(f"vLLM {spec.name} did not wake: http {seen.status_code}")


# one vLLM process holds one model, so its start opens a residency; `created` is the reply's clock
def started_at(spec: EngineSpec) -> str | None:
    try:
        seen = requests.get(_url(spec, "/metrics"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
        for line in seen.text.splitlines():
            if line.startswith("process_start_time_seconds "):
                return datetime.fromtimestamp(float(line.split()[1]), timezone.utc).isoformat()
    except Exception:
        return None
    return None


# the runner the server started with, not the weights: gte-Qwen2 is `*ForCausalLM` served as an embedder
def pools(spec: EngineSpec) -> bool | None:
    try:
        seen = requests.get(_url(spec, "/server_info"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
        seen.raise_for_status()
        said = str(seen.json().get("vllm_config", ""))
    except Exception as e:
        log.warning("vllm.server_info_unknown", engine=spec.name, error=str(e))
        return None
    if "pooler_config=None" in said:
        return False
    return True if "pooler_config=PoolerConfig(" in said else None


# the flags cannot change while the process lives, so one probe per process start answers for all
_probed: dict[tuple[int, str, str], bool] = {}


# what a probe already said, here or in another process, without asking the server
def known_probe(spec: EngineSpec, model: str, started: str | None) -> bool | None:
    if started is None:
        return None
    key = (spec.id, model, started)
    if key not in _probed:
        from .lookup import recorded_tool_probe

        try:
            seen = recorded_tool_probe(spec.id, model, started)
        except Exception as e:
            log.warning("vllm.tool_probe_unread", engine=spec.name, error=str(e))
            return None
        if seen is None:
            return None
        _probed[key] = seen
    return _probed[key]


def tool_calls_probed(spec: EngineSpec, model: str) -> bool | None:
    started = started_at(spec)
    known = known_probe(spec, model, started)
    if started is None or known is not None:
        return known
    seen = _probe(spec, model)
    if seen is None:
        return None
    _probed[(spec.id, model, started)] = seen
    _record(spec, model, started, seen)
    return seen


# an unwritten answer is asked again by the next process, not lost as a false one
def _record(spec: EngineSpec, model: str, started: str, seen: bool) -> None:
    from .lookup import record_tool_probe

    try:
        record_tool_probe(spec.id, model, started, seen)
    except Exception as e:
        log.warning("vllm.tool_probe_unrecorded", engine=spec.name, error=str(e))


def _probe(spec: EngineSpec, model: str) -> bool | None:
    try:
        seen = requests.post(
            _url(spec, "/v1/chat/completions"),
            json={"model": model, **_PROBE},
            headers=_headers(spec),
            timeout=WAKE_TIMEOUT,
        )
    except Exception as e:
        log.warning("vllm.tool_probe_failed", engine=spec.name, error=str(e))
        return None
    # 0.29 without a parser answers in text, as read in its source; a refusal says the same thing
    if seen.status_code == 400:
        return False
    if not seen.ok:
        return None
    try:
        return bool(seen.json()["choices"][0]["message"].get("tool_calls"))
    except (ValueError, KeyError, IndexError, TypeError) as e:
        # an unexpected body says nothing about the parser, and a door must not answer 500 on it
        log.warning("vllm.tool_probe_unreadable", engine=spec.name, error=str(e))
        return None


# every vLLM mounts this host cache; a vLLM on another host is unsupported
def weights_cache() -> str:
    return os.getenv("ENGINE_HF_CACHE") or os.path.expanduser("~/.cache/huggingface")


def _repo_dir(repo: str) -> Path:
    return Path(weights_cache()) / "hub" / ("models--" + repo.replace("/", "--"))


def _snapshot(repo: str) -> Path | None:
    seen = sorted((_repo_dir(repo) / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
    return seen[-1] if seen else None


HUB = "https://huggingface.co"


def _hub_model(repo: str, **params):
    return requests.get(f"{HUB}/api/models/{repo}", params=params or None, timeout=10)


# a 404 is the hub saying no; any other failure says nothing either way
def repo_exists(repo: str) -> bool | None:
    try:
        seen = _hub_model(repo)
    except Exception:
        return None
    if seen.status_code == 200:
        return True
    return False if seen.status_code == 404 else None


def _siblings(repo: str) -> list[dict] | None:
    try:
        return _hub_model(repo, blobs="true").json()["siblings"]
    except Exception as e:
        log.warning("vllm.hub_unanswered", repo=repo, error=str(e))
        return None


# the hub knows the size before the first byte, and a first pull has no recorded size at all
def repo_size(repo: str) -> int | None:
    seen = _siblings(repo)
    if not seen:
        return None
    return sum(f.get("size") or 0 for f in seen) or None


# the hub names each blob by its own hash: sha256 for large files, the git blob sha1 for small ones
def broken_weights(repo: str) -> list[str]:
    root = _repo_dir(repo)
    bad = [b.name for b in (root / "blobs").glob("*") if b.is_file() and not _intact(b)]
    snap = _snapshot(repo)
    # a link whose blob is gone is a file the server will not find
    missing = [f.name for f in snap.iterdir() if f.is_symlink() and not f.exists()] if snap else []
    return bad + missing


def _intact(blob: Path) -> bool:
    name = blob.name
    if len(name) == 64:
        seen = hashlib.sha256()
    elif len(name) == 40:
        # the git blob id the hub names small files by, an integrity check and not a secret
        seen = hashlib.sha1(b"blob %d\0" % blob.stat().st_size, usedforsecurity=False)
    else:
        # a partial download is not broken, it is unfinished: `unfinished_weights` holds it against the row
        return True
    with open(blob, "rb") as f:
        for chunk in iter(lambda: f.read(2**24), b""):
            seen.update(chunk)
    return seen.hexdigest() == name


# a pull cut short leaves a `.incomplete` blob and no link for its file: kept for the resume, not whole
def unfinished_weights(repo: str) -> list[str]:
    snap = _snapshot(repo)
    partial = [b.name for b in (_repo_dir(repo) / "blobs").glob("*.incomplete")]
    missing = [f for f in _expected_files(snap) if not (snap / f).exists()] if snap else []
    return partial + missing


# read off the disk, not the hub: the worker runs offline, and an index names every shard
def _expected_files(snap: Path) -> list[str]:
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (snap / index).exists():
            shards = json.loads((snap / index).read_text()).get("weight_map", {}).values()
            return ["config.json", *sorted(set(shards))]
    # any name the repo gave its weights, `training_args.bin` aside: a fixed name looped a pull forever
    single = [f.name for f in snap.iterdir()
              if f.suffix in (".safetensors", ".bin") and "model" in f.name and f.exists()]
    return ["config.json", *(single or ["model.safetensors"])]


# what is wrong with the weights, hashed once; empty means intact, a missing snapshot is named
def weights_check(repo: str) -> list[str]:
    if _snapshot(repo) is None:
        return ["no snapshot"]
    return broken_weights(repo) + unfinished_weights(repo)


def weights_intact(repo: str) -> bool:
    return not weights_check(repo)


# a blob with the right name and the wrong bytes is never fetched again, so it goes before the pull
def _drop(repo: str, broken: list[str]) -> None:
    for name in broken:
        blob = _repo_dir(repo) / "blobs" / name
        if blob.is_file():
            log.warning("vllm.blob_broken", repo=repo, blob=name)
            blob.unlink()


# only what a server loads: `onnx` and `.pt` copies doubled a pull for nothing
_PULLED = ["*.json", "*.txt", "*.model", "tokenizer*", "*.tiktoken"]


# safetensors where the repo has them, else the pickled `.bin`: bge-m3 ships only `pytorch_model.bin`
def _patterns(repo: str) -> list[str]:
    files = _siblings(repo)
    if files is None:
        return [*_PULLED, "*.safetensors", "*.bin"]
    has_safetensors = any(f.get("rfilename", "").endswith(".safetensors") for f in files)
    return [*_PULLED, "*.safetensors" if has_safetensors else "*.bin"]


# the worker runs offline so the reranker asks nobody; a pull is the one call that must go out
def pull_weights(repo: str) -> None:
    # hashed once: the intact check and the drop each hashed every blob again
    broken = broken_weights(repo)
    if _snapshot(repo) is not None and not broken and not unfinished_weights(repo):
        log.info("vllm.weights_present", repo=repo)
        return
    _drop(repo, broken)
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_HOME": weights_cache()}
    code = ("import json, sys; from huggingface_hub import snapshot_download;"
            " snapshot_download(sys.argv[1], allow_patterns=json.loads(sys.argv[2]))")
    done = subprocess.run(
        [sys.executable, "-c", code, repo, json.dumps(_patterns(repo))],
        env=env, capture_output=True, text=True, timeout=PULL_TIMEOUT,
    )
    if done.returncode:
        said = (done.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"pull of {repo} failed: {said}")
    # the pull finished: a partial blob left now is of a file no pattern asks for, and would loop it
    for orphan in (_repo_dir(repo) / "blobs").glob("*.incomplete"):
        log.warning("vllm.partial_blob_orphaned", repo=repo, blob=orphan.name)
        orphan.unlink()


_DTYPES = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}


# vLLM has no `/api/show`, but the weights describe themselves in `config.json`
def artifact_of(repo: str) -> dict:
    snap = _snapshot(repo)
    if snap is None:
        return {}
    config = snap / "config.json"
    seen = json.loads(config.read_text()) if config.exists() else {}
    method = (seen.get("quantization_config") or {}).get("quant_method")
    # the snapshot the server reads, files only and `1_Pooling/` too: a directory counted its 4096 bytes
    files = [f for f in snap.rglob("*") if f.is_file()]
    return {
        "quant": method.upper() if method else _DTYPES.get(seen.get("torch_dtype")),
        "size_bytes": sum(f.stat().st_size for f in files) or None,
    }


class StillServed(ValueError):
    pass


def delete_weights(repo: str) -> None:
    path = _repo_dir(repo)
    if not path.exists():
        log.info("vllm.weights_already_absent", repo=repo)
        return
    shutil.rmtree(path)
