#!/usr/bin/env bash
# brings the stand up with the card through CDI, or with `--cpu` on a host without one (docs/stand_modes.md)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--cpu" ]; then
  shift
  # the layer leaves the card's engines out but does not stop them, and a running stand kept answering from the card
  docker compose stop ollama vllm >/dev/null 2>&1 || true
  exec docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d "$@"
fi

# read whole before the grep: a grep that quits early sent docker a SIGPIPE, which pipefail read as no card
if ! info=$(docker info 2>&1); then
  echo "\`docker info\` failed, so whether Docker sees a card is unknown:" >&2
  printf '%s\n' "$info" | tail -n 3 >&2
  exit 1
fi
if ! grep -q "nvidia.com/gpu=all" <<<"$info"; then
  echo "No NVIDIA card is visible to Docker through CDI: \`docker info\` lists no nvidia.com/gpu=all," >&2
  echo "so the services that reserve it would not start (\"unresolvable CDI devices\")." >&2
  echo "Give Docker the card (README, Quickstart: nvidia-ctk cdi generate), or start without one:" >&2
  echo "  scripts/up.sh --cpu" >&2
  exit 1
fi

# config.yaml names the judge; the vllm service starts with that model, and nothing else names it
judge_on_vllm() {
  sed -nE 's/^ *judging: *\{ *model: *([^,} ]+) *,.*engine: *vllm[,} ].*/\1/p' config.yaml | head -n 1
}
if model=$(judge_on_vllm) && [ -n "$model" ]; then
  export VLLM_MODEL="$model"
fi

loaded_on_ollama() {
  docker compose exec -T ollama ollama ps 2>/dev/null | awk 'NR > 1 && $1 != "" {print $1}'
}

vllm_running() {
  docker compose ps --status running --services 2>/dev/null | grep -qx vllm
}

# a vLLM starting while ollama keeps a model on the card dies short of memory, and the stack waits behind it
plan_or_nothing() {
  # a dry run cannot play a one-shot container's exit, so after `down` it waited on `seed` for ever
  timeout "${UP_PLAN_TIMEOUT:-20}" docker compose --dry-run up -d "$@" 2>&1
}
plan=$(plan_or_nothing "$@") || plan=""
if grep -qE -- '-vllm-1 +(Recreate|Creat|Start)' <<<"$plan" || { [ -z "$plan" ] && ! vllm_running; }; then
  names=$(loaded_on_ollama) || names=""
  for name in $names; do
    echo "vLLM is about to start: unloading $name from ollama so it finds the card free" >&2
    docker compose exec -T ollama ollama stop "$name" >/dev/null
  done
  for _ in $(seq 1 30); do
    [ -z "$(loaded_on_ollama || true)" ] && break
    sleep 1
  done
  if [ -n "$(loaded_on_ollama || true)" ]; then
    echo "ollama still holds a model after 30 s: vLLM may not find the card free" >&2
  fi
fi
exec docker compose up -d "$@"
