#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${MTS_PYTHON:-/opt/conda/envs/MTS/bin/python}"
POLL_SECONDS="${MTS_GLT_V2_POLL_SECONDS:-120}"
DECISION="results/mts_glt_v2/screening_decision.json"

while [[ ! -f "$DECISION" ]]; do
    sleep "$POLL_SECONDS"
done

GO_20K="$($PYTHON_BIN -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1]))["go_20k"] else "false")' \
    "$DECISION")"
if [[ "$GO_20K" != "true" ]]; then
    echo MTS_GLT_V2_PROBES_STOPPED_BY_SCREENING_GATE
    exit 0
fi

FORMAL_RUN="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["formal_run_name"])' \
    "$DECISION")"
USE_COMPACT19="$($PYTHON_BIN -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1]))["compact19_enabled"] else "false")' \
    "$DECISION")"
if [[ "$USE_COMPACT19" == "true" ]]; then
    FUSED_MODE="o8_glt_atom_desc"
else
    FUSED_MODE="o8_glt_atom"
fi

SELECTION="results/mts_glt_v2/formal/trajectory_selection.json"
CANDIDATE_ARGS=()
for STEP in 5000 10000 20000; do
    STEP_TAG="$(printf '%03dk' "$((STEP / 1000))")"
    CHECKPOINT="results/mts_glt_v2/${FORMAL_RUN}/mts_glt_v2_probe_${STEP_TAG}.pth"
    while [[ ! -f "$CHECKPOINT" ]]; do
        sleep "$POLL_SECONDS"
    done
    RUN_NAME="${FORMAL_RUN}_probe_${STEP_TAG}"
    for MODE in o8_only "$FUSED_MODE"; do
        "$PYTHON_BIN" scripts/run_mts_glt_v2_finetune.py \
            --checkpoint "$CHECKPOINT" \
            --mode "$MODE" \
            --layers 6 \
            --attention-variant mips \
            --run-name "$RUN_NAME" \
            --gpu-ids 0 \
            --tasks xc ei egb \
            --folds 0
    done
    PROBE_REPORT="results/mts_glt_v2/downstream/${RUN_NAME}/probe_summary.json"
    "$PYTHON_BIN" scripts/report_mts_glt_v2_downstream.py \
        --run-name "$RUN_NAME" \
        --tasks xc ei egb \
        --folds 0 \
        --fused-mode "$FUSED_MODE" \
        --output "$PROBE_REPORT"
    CANDIDATE_ARGS+=(--candidate "$STEP" "$PROBE_REPORT" "$CHECKPOINT" "$RUN_NAME")
done

"$PYTHON_BIN" scripts/select_mts_glt_v2_trajectory.py \
    "${CANDIDATE_ARGS[@]}" \
    --output "$SELECTION"

SELECTED_CHECKPOINT="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
    "$SELECTION")"
SELECTED_RUN="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["run_name"])' \
    "$SELECTION")"

for MODE in o8_only "$FUSED_MODE"; do
    "$PYTHON_BIN" scripts/run_mts_glt_v2_finetune.py \
        --checkpoint "$SELECTED_CHECKPOINT" \
        --mode "$MODE" \
        --layers 6 \
        --attention-variant mips \
        --run-name "$SELECTED_RUN" \
        --gpu-ids 0,1,2,3
done
"$PYTHON_BIN" scripts/report_mts_glt_v2_downstream.py \
    --run-name "$SELECTED_RUN" \
    --tasks eat eea egb egc ei eps nc xc \
    --folds 0 1 2 3 4 \
    --fused-mode "$FUSED_MODE"
"$PYTHON_BIN" scripts/report_mts_glt_v2_final.py

echo MTS_GLT_V2_MATCHED_8X5_COMPLETE
