import argparse
import json
import os
import time
from pathlib import Path

import requests
from engines import EngineSpec, balances, core
from models.registry import EngineKind, Placement

ROOT = Path(__file__).resolve().parent.parent
PARAGRAPH = (
    "Redis persists data with RDB snapshots taken at intervals and with an append-only file that logs every "
    "write; the two can be combined, and on restart the append-only file is preferred because it is more complete. "
)
# one call mostly input, one mostly output: two debits solve the two prices, the third checks them
CASES = (
    ("heavy_in", [{"role": "user", "content": PARAGRAPH * 70 + "\n\nReply with the single word OK."}], 16),
    ("heavy_out", [{"role": "user", "content": "Write a detailed essay of about 1200 words on how database indexes work."}], 1500),
    ("check", [
        {"role": "system", "content": "You answer interview questions briefly and precisely."},
        {"role": "user", "content": "What is the difference between a process and a thread?"},
    ], 1024),
)


# the containers get the pair through `.env`; the host reads the same file
READER = balances.GONKA


def _load_env(prefix: str) -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.startswith(f"{prefix}_") and name not in os.environ:
            os.environ[name] = _unquoted(value.strip())


# compose takes a quoted value without its quotes, and a key read with them is refused by the broker
def _unquoted(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


# a job of the stand on the same key would land its debit inside a measured call
def _refuse_a_busy_stand() -> None:
    stand = os.environ.get("RAG_LAB_URL", "http://localhost:8000")
    try:
        seen = requests.get(f"{stand}/v1/job", params={"status": ["running", "new"]}, timeout=10)
    except requests.ConnectionError:
        print(f"the stand at {stand} does not answer: no job of its own can spend the key", flush=True)
        return
    seen.raise_for_status()
    busy = [f"{job['id']} {job['type']}" for job in seen.json()]
    if busy:
        raise SystemExit(f"the stand has jobs running or queued ({', '.join(busy)}); measure on a quiet stand")


# the stand's own reader, so the script and the door cannot disagree about one route
def _balance(spec: EngineSpec) -> float:
    seen = balances.read(spec, READER)
    if seen["balance"] is None:
        raise SystemExit(f"{spec.name} did not say its balance: {seen['why']}")
    return round(float(seen["balance"]), 7)


# the broker may debit after the answer: wait until the balance moves and then holds
def _settled(spec: EngineSpec, before: float, wait: float = 90) -> float:
    seen, same, started = before, 0, time.monotonic()
    while time.monotonic() - started < wait:
        time.sleep(3)
        now = _balance(spec)
        same = same + 1 if now != before and now == seen else 0
        if same >= 2:
            return now
        seen = now
    # an unsettled balance priced the call on a debit still on its way
    raise SystemExit(f"{spec.name}: the balance did not hold still within {wait:.0f} s")


def measure(spec: EngineSpec, model: str) -> list[dict]:
    rows = []
    for case, messages, max_tokens in CASES:
        before = _balance(spec)
        reply = requests.post(
            f"{core.base_url(spec)}/v1/chat/completions", headers=core.bearer(spec), timeout=180,
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        )
        reply.raise_for_status()
        usage = reply.json().get("usage") or {}
        if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
            raise SystemExit(f"{spec.name} answered without usage: no price can be read off it")
        rows.append({
            "model": model, "case": case, "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
            "spent_usd": round(before - _settled(spec, before), 7),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


# dollars per million tokens, from the two lopsided calls
def solve(rows: list[dict]) -> dict:
    by = {r["case"]: r for r in rows}
    a1, b1, d1 = by["heavy_in"]["in"], by["heavy_in"]["out"], by["heavy_in"]["spent_usd"]
    a2, b2, d2 = by["heavy_out"]["in"], by["heavy_out"]["out"], by["heavy_out"]["spent_usd"]
    det = a1 * b2 - a2 * b1
    if det == 0:
        raise SystemExit("the two cases spent tokens in one proportion: the prices cannot be told apart")
    price_in, price_out = (d1 * b2 - d2 * b1) / det * 1e6, (a1 * d2 - a2 * d1) / det * 1e6
    check = by["check"]
    return {
        "model": rows[0]["model"], "in_per_million": round(price_in, 4), "out_per_million": round(price_out, 4),
        "check_spent_usd": check["spent_usd"],
        "check_predicted_usd": round((price_in * check["in"] + price_out * check["out"]) / 1e6, 7),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="What a broker debits per input and output token, by model.")
    parser.add_argument("--prefix", required=True, help="the engine's env_prefix, as GONKA")
    parser.add_argument("--model", action="append", required=True, help="a model the broker serves; repeat")
    args = parser.parse_args()
    _load_env(args.prefix)
    _refuse_a_busy_stand()
    spec = EngineSpec(0, args.prefix.lower(), EngineKind.openai_compatible, args.prefix, Placement.remote)
    # one call at a time, and no other client on the key, or the debits mix
    for model in args.model:
        print(json.dumps(solve(measure(spec, model))), flush=True)


if __name__ == "__main__":
    main()
