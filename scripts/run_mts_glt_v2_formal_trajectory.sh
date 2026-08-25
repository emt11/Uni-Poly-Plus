#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${MTS_PYTHON:-/opt/conda/envs/MTS/bin/python}"
TORCHRUN_BIN="${MTS_TORCHRUN:-/opt/conda/envs/MTS/bin/torchrun}"
POLL_SECONDS="${MTS_GLT_V2_POLL_SECONDS:-120}"
DECISION="results/mts_glt_v2/screening_decision.json"

while [[ ! -f "$DECISION" ]]; do
    sleep "$POLL_SECONDS"
done

GO_20K="$($PYTHON_BIN -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1]))["go_20k"] else "false")' \
    "$DECISION")"
if [[ "$GO_20K" != "true" ]]; then
    echo MTS_GLT_V2_20K_STOPPED_BY_SCREENING_GATE
    exit 0
fi

SOURCE_CONFIG="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_config"])' \
    "$DECISION")"
FORMAL_RUN="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["formal_run_name"])' \
    "$DECISION")"
WEIGHT="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_weight"])' \
    "$DECISION")"
RUN_TAG="${FORMAL_RUN//\//_}"
CONFIG="configs/mts/glt_v2_${RUN_TAG}.json"

if [[ ! -f "$CONFIG" ]]; then
    "$PYTHON_BIN" scripts/make_mts_glt_v2_stage_config.py \
        --source "$SOURCE_CONFIG" \
        --output "$CONFIG" \
        --run-name "$FORMAL_RUN" \
        --stop-after-steps 20000 \
        --infonce-weight "$WEIGHT"
fi

echo "START_MTS_GLT_V2_FORMAL config=$CONFIG run=$FORMAL_RUN weight=$WEIGHT"
EXPERIMENT_CONFIG="$CONFIG" \
PYTHON_BIN="$PYTHON_BIN" \
TORCHRUN_BIN="$TORCHRUN_BIN" \
bash scripts/run_mips_trimer_scage.sh
echo MTS_GLT_V2_FORMAL_20K_COMPLETE
