#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}

FULL=pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/full_5k.pth
OFF=pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/off_5k.pth
while [[ ! -f "$FULL" || ! -f "$OFF" ]]; do
  if ! tmux list-windows -t Uni-Poly -F '#{window_name}' 2>/dev/null | grep -qx trimer-val-c-5k; then
    echo "matched 5k pretraining exited before both arm checkpoints were produced" >&2
    exit 1
  fi
  sleep 60
done

"$PYTHON_BIN" scripts/analyze_mts_trimer_matched_5k.py extract --arm full --device cuda:0
"$PYTHON_BIN" scripts/analyze_mts_trimer_matched_5k.py extract --arm off --device cuda:0
"$PYTHON_BIN" scripts/analyze_mts_trimer_matched_5k.py analyze-frozen

RESULT_BASE=results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/neural
LOG_BASE=logs/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/neural
for arm in full off; do
  checkpoint="pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/${arm}_5k.pth"
  run_name="trimer_validation_v1_matched_5k_${arm}"
  for mode in o8_only o8_glt_graph; do
    "$PYTHON_BIN" scripts/run_mts_glt_graphgate_finetune.py \
      --checkpoint "$checkpoint" \
      --geometry-mode "$arm" \
      --mode "$mode" \
      --run-name "$run_name" \
      --result-base "$RESULT_BASE" \
      --log-base "$LOG_BASE" \
      --tasks xc ei eps \
      --folds 0 1 2 \
      --gpu-ids 0,1,2,3 \
      --epochs 100 \
      --patience 10
  done
done

"$PYTHON_BIN" scripts/analyze_mts_trimer_matched_5k.py analyze-neural
"$PYTHON_BIN" scripts/analyze_mts_trimer_matched_5k.py report
