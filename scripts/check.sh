#!/usr/bin/env bash
# The pre-commit gate in one call: full output goes to a log, the terminal gets one line per step and the failures.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="${CHECK_LOG:-temp_files/check.log}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
status=0

step() {
  local name=$1; shift
  if "$@" >> "$LOG" 2>&1; then
    echo "$name: ok"
  else
    echo "$name: FAILED"
    status=1
  fi
}

step ruff uv run ruff check app tests scripts
step inventory uv run python scripts/config_inventory.py --check
step pytest uv run pytest -q -n "${CHECK_WORKERS:-8}" -p no:cacheprovider "$@"
grep -E "^(FAILED|ERROR) |passed|failed" "$LOG" | tail -20
echo "log: $LOG"
exit $status
