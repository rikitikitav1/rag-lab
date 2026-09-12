#!/usr/bin/env bash
# brings the stand up with the card through CDI, or with `--cpu` on a host without one (docs/stand_modes.md)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--cpu" ]; then
  shift
  exec docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d "$@"
fi

if ! docker info 2>/dev/null | grep -q "nvidia.com/gpu=all"; then
  echo "No NVIDIA card is visible to Docker through CDI: \`docker info\` lists no nvidia.com/gpu=all," >&2
  echo "so the services that reserve it would not start (\"unresolvable CDI devices\")." >&2
  echo "Give Docker the card (README, Quickstart: nvidia-ctk cdi generate), or start without one:" >&2
  echo "  scripts/up.sh --cpu" >&2
  exit 1
fi
exec docker compose up -d "$@"
