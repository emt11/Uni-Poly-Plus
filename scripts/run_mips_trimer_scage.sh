#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}

if [[ -z "${EXPERIMENT_CONFIG:-}" ]]; then
  echo "MTS requires EXPERIMENT_CONFIG pointing to an active experiment JSON file." >&2
  exit 2
fi
if [[ ! -f "$EXPERIMENT_CONFIG" ]]; then
  echo "MTS production is disabled: configuration path does not exist: $EXPERIMENT_CONFIG" >&2
  exit 2
fi

RESOLVED_INPUT=$("$PYTHON_BIN" scripts/resolve_mips_trimer_scage.py \
  "$EXPERIMENT_CONFIG" --print-path)
TORCHRUN_BIN=${TORCHRUN_BIN:-/opt/conda/envs/MTS/bin/torchrun}
export CUDA_VISIBLE_DEVICES=1,2,3
# EXTRA_ARGS carries runtime-only controls (e.g. --resume_smoke) that never
# alter the resolved experiment identity.
# shellcheck disable=SC2086
exec "$TORCHRUN_BIN" --standalone --nproc_per_node=3 \
  scripts/pretrain.py --experiment_config "$RESOLVED_INPUT" ${EXTRA_ARGS:-}
