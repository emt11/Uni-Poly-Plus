#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${MTS_PYTHON:-/opt/conda/envs/MTS/bin/python}"
POLL_SECONDS="${MTS_GLT_V2_POLL_SECONDS:-120}"

W0_REPORT="results/mts_glt_v2/downstream/infonce_screen_a6_h_w0_5k/paired_summary.json"
W025_REPORT="results/mts_glt_v2/downstream/infonce_screen_a6_h_w025_5k/paired_summary.json"
W1_REPORT="results/mts_glt_v2/downstream/architecture_screen_a6_h_5k/paired_summary.json"
W0_CHECKPOINT="results/mts_glt_v2/infonce_screen/a6_h_w0_5k/mts_glt_v2_probe_005k.pth"
W025_CHECKPOINT="results/mts_glt_v2/infonce_screen/a6_h_w025_5k/mts_glt_v2_probe_005k.pth"
W1_CHECKPOINT="results/mts_glt_v2/architecture_screen/a6_h_5k/mts_glt_v2_probe_005k.pth"

for required in "$W0_REPORT" "$W025_REPORT"; do
    while [[ ! -f "$required" ]]; do
        sleep "$POLL_SECONDS"
    done
done

SELECTION="results/mts_glt_v2/infonce_screen/selection.json"
"$PYTHON_BIN" scripts/select_mts_glt_v2_infonce.py \
    --candidate 0 "$W0_REPORT" "$W0_CHECKPOINT" configs/mts/glt_v2_a6_h_infonce_w0_5k.json \
    --candidate 0.25 "$W025_REPORT" "$W025_CHECKPOINT" configs/mts/glt_v2_a6_h_infonce_w025_5k.json \
    --candidate 1 "$W1_REPORT" "$W1_CHECKPOINT" configs/mts/glt_v2_a6_h_5k.json \
    --output "$SELECTION"

SELECTED_CHECKPOINT="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
    "$SELECTION")"
SELECTED_RUN="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["run_name"])' \
    "$SELECTION")"

COMPACT_ROOT="results/mts_glt_v2/compact19_probe/selected_5k"
"$PYTHON_BIN" scripts/probe_mts_glt_v2_compact19.py \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --layers 6 \
    --attention-variant mips \
    --device cuda:0 \
    --output-root "$COMPACT_ROOT"

COMPACT_REPORT="$COMPACT_ROOT/report.json"
ADMITTED="$($PYTHON_BIN -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1]))["admitted"] else "false")' \
    "$COMPACT_REPORT")"

DESCRIPTOR_REPORT=""
if [[ "$ADMITTED" == "true" ]]; then
    "$PYTHON_BIN" scripts/run_mts_glt_v2_finetune.py \
        --checkpoint "$SELECTED_CHECKPOINT" \
        --mode o8_glt_atom_desc \
        --layers 6 \
        --attention-variant mips \
        --run-name "$SELECTED_RUN" \
        --gpu-ids 0 \
        --tasks xc ei egb \
        --folds 0
    DESCRIPTOR_REPORT="$COMPACT_ROOT/descriptor_paired_summary.json"
    "$PYTHON_BIN" scripts/report_mts_glt_v2_downstream.py \
        --run-name "$SELECTED_RUN" \
        --tasks xc ei egb \
        --folds 0 \
        --base-mode o8_glt_atom \
        --fused-mode o8_glt_atom_desc \
        --output "$DESCRIPTOR_REPORT"
fi

FINALIZE_ARGS=(
    --infonce-selection "$SELECTION"
    --compact-report "$COMPACT_REPORT"
    --output results/mts_glt_v2/screening_decision.json
)
if [[ -n "$DESCRIPTOR_REPORT" ]]; then
    FINALIZE_ARGS+=(--descriptor-report "$DESCRIPTOR_REPORT")
fi
"$PYTHON_BIN" scripts/finalize_mts_glt_v2_screening.py "${FINALIZE_ARGS[@]}"

GO_20K="$($PYTHON_BIN -c \
    'import json,sys; print("true" if json.load(open(sys.argv[1]))["go_20k"] else "false")' \
    results/mts_glt_v2/screening_decision.json)"
if [[ "$GO_20K" != "true" ]]; then
    "$PYTHON_BIN" scripts/report_mts_glt_v2_final.py
fi

echo MTS_GLT_V2_SCREENING_DECISION_COMPLETE
