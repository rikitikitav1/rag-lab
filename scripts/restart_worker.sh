#!/usr/bin/env bash
# Restart the worker between jobs: hold the waiting line, wait out what runs now, restart, release the same jobs.
set -euo pipefail
cd "$(dirname "$0")/.."

API="${API:-http://localhost:8000}"
held=""

ids_with_status() {
  curl -sf "$API/v1/job?status=$1&limit=1000&sort_by=id&sort_order=asc" |
    python3 -c 'import json,sys; print(" ".join(str(j["id"]) for j in json.load(sys.stdin)))'
}

as_json_ids() {
  python3 -c 'import json,sys; print(json.dumps({"ids": [int(x) for x in sys.argv[1:]]}))' "$@"
}

release() {
  # a failure anywhere after the hold must not leave the line held
  if [ -n "$held" ]; then
    # shellcheck disable=SC2086
    curl -sf -X POST "$API/v1/job/resume" -H 'Content-Type: application/json' -d "$(as_json_ids $held)" |
      python3 -c 'import json,sys; print("resumed", len(json.load(sys.stdin)["resumed"]))'
  fi
}
trap release EXIT

waiting=$(ids_with_status new)
if [ -n "$waiting" ]; then
  # shellcheck disable=SC2086
  held=$(curl -sf -X POST "$API/v1/job/pause" -H 'Content-Type: application/json' -d "$(as_json_ids $waiting)" |
    python3 -c 'import json,sys; print(" ".join(str(i) for i in json.load(sys.stdin)["paused"]))')
fi
echo "held $(wc -w <<< "$held")"

# read after the hold, never before: a job claimed in between is waited out too
running=$(ids_with_status running)
if [ -n "$running" ]; then
  echo "waiting for $running"
  # shellcheck disable=SC2086
  docker compose exec -T worker python scripts/wait_jobs.py $running || true
fi
left=$(ids_with_status running)
if [ -n "$left" ]; then
  echo "still running after the wait: $left; not restarting" >&2
  exit 1
fi

docker compose restart worker
