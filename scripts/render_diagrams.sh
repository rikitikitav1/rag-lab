#!/usr/bin/env bash
# Renders docs/diagrams/*.d2 to SVG. CI runs this and fails if outputs drift.
set -euo pipefail
cd "$(dirname "$0")/../docs/diagrams"
for src in *.d2; do
  d2 --layout elk "$src" "${src%.d2}.svg"
done
