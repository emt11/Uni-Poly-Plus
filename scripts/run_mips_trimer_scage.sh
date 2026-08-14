#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}

if [[ -z "${EXPERIMENT_CONFIG:-}" ]]; then
  echo "MTS production is disabled: EXPERIMENT_CONFIG is required while the configuration layer is retired." >&2
  exit 2
fi
if [[ ! -f "$EXPERIMENT_CONFIG" ]]; then
  echo "MTS production is disabled: configuration path does not exist: $EXPERIMENT_CONFIG" >&2
  exit 2
fi

# The resolver is deliberately fail-closed and must run before any GPU,
# worker, cache, output-directory, or training initialization.
exec "$PYTHON_BIN" scripts/resolve_mips_trimer_scage.py "$EXPERIMENT_CONFIG"
