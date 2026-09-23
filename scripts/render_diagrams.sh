#!/usr/bin/env bash
# Renders docs/diagrams/*.d2 and *.puml to SVG. CI runs this and fails if outputs drift.
set -euo pipefail
cd "$(dirname "$0")/../docs/diagrams"

for src in *.d2; do
  d2 --layout elk "$src" "${src%.d2}.svg"
done

# pinned, not `latest`: a new PlantUML redraws every SVG and the drift gate would fail on nothing
PLANTUML_IMAGE="plantuml/plantuml:1.2026.8"
if compgen -G "*.puml" > /dev/null; then
  docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" -w /data \
    "$PLANTUML_IMAGE" -tsvg ./*.puml
fi
