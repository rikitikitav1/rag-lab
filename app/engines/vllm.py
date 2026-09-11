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

from .core import EngineSpec, Unconfigured, api_key, base_url

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


def score(spec: EngineSpec, model: str, pairs: list[tuple[str, str]]) -> list[float]:
    scores = []
    for start in range(0, len(pairs), SCORE_BATCH):
        chunk = pairs[start : start + SCORE_BATCH]
        seen = requests.post(
            _url(spec, "/score"),
            json={"model": model, "text_1": [q for q, _ in chunk], "text_2": [d for _, d in chunk]},
            headers=_headers(spec),
            timeout=WAKE_TIMEOUT,
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
def card_state(spec: EngineSpec) -> str:
    try:
        seen = requests.get(_url(spec, "/is_sleeping"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
        seen.raise_for_status()
        return "asleep" if seen.json()["is_sleeping"] else "awake"
    except (requests.ConnectionError, Unconfigured):
        return "down"
    except Exception as e:
        log.warning("vllm.card_state_unknown", engine=spec.name, error=str(e))
        return "unknown"


# None when the server cannot say: the sleep routes exist only under VLLM_SERVER_DEV_MODE
def is_sleeping(spec: EngineSpec) -> bool | None:
    try:
        seen = requests.get(_url(spec, "/is_sleeping"), headers=_headers(spec), timeout=HTTP_TIMEOUT)
        seen.raise_for_status()
        return bool(seen.json()["is_sleeping"])
    except Exception as e:
        log.warning("vllm.sleep_state_unknown", engine=spec.name, error=str(e))
        return None


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


# the flags cannot change while the process lives, so one probe per process start answers for all
_probed: dict[tuple[int, str, str], bool] = {}


# what a probe already said, without asking: a stamp must not send a request into a judged pass
def known_probe(spec: EngineSpec, model: str, started: str | None) -> bool | None:
    return _probed.get((spec.id, model, started)) if started else None


def tool_calls_probed(spec: EngineSpec, model: str) -> bool | None:
    started = started_at(spec)
    if started is None:
        return None
    key = (spec.id, model, started)
    if key not in _probed:
        seen = _probe(spec, model)
        if seen is None:
            return None
        _probed[key] = seen
    return _probed[key]


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
    return bool(seen.json()["choices"][0]["message"].get("tool_calls"))


# the engines' weights live in the host cache the `vllm` service mounts, not in the `hf_cache` volume
def weights_cache() -> str:
    return os.getenv("ENGINE_HF_CACHE") or os.path.expanduser("~/.cache/huggingface")


def _repo_dir(repo: str) -> Path:
    return Path(weights_cache()) / "hub" / ("models--" + repo.replace("/", "--"))


def _snapshot(repo: str) -> Path | None:
    seen = sorted((_repo_dir(repo) / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
    return seen[-1] if seen else None


# the hub knows the size before the first byte, and a first pull has no recorded size at all
def repo_size(repo: str) -> int | None:
    try:
        seen = requests.get(
            f"https://huggingface.co/api/models/{repo}", params={"blobs": "true"}, timeout=10
        ).json()
        return sum(f.get("size") or 0 for f in seen["siblings"]) or None
    except Exception as e:
        log.warning("vllm.repo_size_unknown", repo=repo, error=str(e))
        return None


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
        seen = hashlib.sha1(b"blob %d\0" % blob.stat().st_size)
    else:
        # a leftover partial or a name we cannot check: not a blob the snapshot points at
        return not name.endswith(".incomplete")
    with open(blob, "rb") as f:
        for chunk in iter(lambda: f.read(2**24), b""):
            seen.update(chunk)
    return seen.hexdigest() == name


def weights_intact(repo: str) -> bool:
    return _snapshot(repo) is not None and not broken_weights(repo)


# a blob with the right name and the wrong bytes is never fetched again, so it goes before the pull
def _drop_broken(repo: str) -> None:
    root = _repo_dir(repo)
    for blob in (root / "blobs").glob("*"):
        if blob.is_file() and not _intact(blob):
            log.warning("vllm.blob_broken", repo=repo, blob=blob.name)
            blob.unlink()


# the worker runs offline so the reranker asks nobody; a pull is the one call that must go out
def pull_weights(repo: str) -> None:
    if weights_intact(repo):
        log.info("vllm.weights_present", repo=repo)
        return
    _drop_broken(repo)
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_HOME": weights_cache()}
    code = "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])"
    done = subprocess.run(
        [sys.executable, "-c", code, repo],
        env=env, capture_output=True, text=True, timeout=PULL_TIMEOUT,
    )
    if done.returncode:
        said = (done.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"pull of {repo} failed: {said}")


_DTYPES = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}


# vLLM has no `/api/show`, but the weights describe themselves in `config.json`
def artifact_of(repo: str) -> dict:
    snap = _snapshot(repo)
    if snap is None:
        return {}
    config = snap / "config.json"
    seen = json.loads(config.read_text()) if config.exists() else {}
    method = (seen.get("quantization_config") or {}).get("quant_method")
    blobs = [f for f in (_repo_dir(repo) / "blobs").glob("*") if f.is_file()]
    return {
        "quant": method.upper() if method else _DTYPES.get(seen.get("torch_dtype")),
        "size_bytes": sum(f.stat().st_size for f in blobs) or None,
    }


def delete_weights(repo: str) -> None:
    path = _repo_dir(repo)
    if not path.exists():
        log.info("vllm.weights_already_absent", repo=repo)
        return
    shutil.rmtree(path)
