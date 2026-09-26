#!/usr/bin/env bash
# Renders docs/diagrams/*.puml to SVG. CI runs this and fails if outputs drift; a .drawio.svg carries its own picture.
set -euo pipefail
cd "$(dirname "$0")/../docs/diagrams"

# pinned, not `latest`: a new PlantUML redraws every SVG and the drift gate would fail on nothing
PLANTUML_IMAGE="plantuml/plantuml:1.2026.8"
if compgen -G "*.puml" > /dev/null; then
  docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" -w /data \
    "$PLANTUML_IMAGE" -tsvg ./*.puml
fi
