#!/bin/bash
set -euo pipefail

# Compatibility wrapper.  The second route is configured exclusively through
# the strict non-PBC experiment JSON and the canonical run.sh entry point.
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export BASELINE=${BASELINE:-MIPS-Trimer-SCAGE}
export EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-configs/mts/default.json}

exec bash scripts/run.sh
