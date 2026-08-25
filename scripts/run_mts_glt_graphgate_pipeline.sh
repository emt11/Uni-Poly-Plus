#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}

QC=results/mts_glt_graphgate_v1/sidecar_qc/coverage_report.json
COUNTS=results/mts_glt_graphgate_v1/sidecar_qc/line_label_counts.json
[[ -f "$QC" && -f "$COUNTS" ]] || {
  echo "GraphGate sidecar QC and label counts must exist before training" >&2
  exit 2
}

mkdir -p configs/mts/glt_graphgate_v1/generated logs/mts_glt_graphgate_v1/pretrain

make_stage() {
  local steps=$1
  local tag=$2
  local result_root=$3
  local output_path=$4
  "$PYTHON_BIN" scripts/make_mts_glt_graphgate_stage_config.py \
    --stop-after "$steps" --experiment-id "$tag" \
    --result-root "$result_root" --output-path "$output_path" \
    --output "configs/mts/glt_graphgate_v1/generated/${tag}.json"
}

# A real three-rank two-step smoke has isolated outputs.
make_stage 2 graphgate_ddp_smoke \
  results/mts_glt_graphgate_v1/smoke/ddp \
  pretrained_models/mts_glt_graphgate_v1/smoke/ddp.pth
EXPERIMENT_CONFIG=configs/mts/glt_graphgate_v1/generated/graphgate_ddp_smoke.json \
  scripts/run_mips_trimer_scage.sh \
  2>&1 | tee logs/mts_glt_graphgate_v1/pretrain/ddp_smoke.log

# One scientific trajectory: step 0 -> 500 -> 5k.
EXPERIMENT_CONFIG=configs/mts/glt_graphgate_v1/graphgate_500.json \
  scripts/run_mips_trimer_scage.sh \
  2>&1 | tee logs/mts_glt_graphgate_v1/pretrain/step_000_to_500.log

make_stage 5000 graphgate_formal_5k \
  results/mts_glt_graphgate_v1/pretrain \
  pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth
EXPERIMENT_CONFIG=configs/mts/glt_graphgate_v1/generated/graphgate_formal_5k.json \
EXTRA_ARGS="--resume_state pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth.last.pt" \
  scripts/run_mips_trimer_scage.sh \
  2>&1 | tee logs/mts_glt_graphgate_v1/pretrain/step_500_to_5k.log

CHECKPOINT=results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_005k.pth
for mode in o8_only o8_glt_graph; do
  "$PYTHON_BIN" scripts/run_mts_glt_graphgate_finetune.py \
    --checkpoint "$CHECKPOINT" --mode "$mode" --run-name screen_5k \
    --tasks xc ei egc --folds 0 1 2
done
"$PYTHON_BIN" scripts/report_mts_glt_v2_downstream.py \
  --result-base results/mts_glt_graphgate_v1/downstream \
  --run-name screen_5k --tasks xc ei egc --folds 0 1 2 \
  --fused-mode o8_glt_graph \
  --output results/mts_glt_graphgate_v1/downstream/screen_5k/paired_summary.json
"$PYTHON_BIN" scripts/report_mts_glt_graphgate.py \
  --paired-summary results/mts_glt_graphgate_v1/downstream/screen_5k/paired_summary.json \
  --audit-root results/mts_glt_graphgate_v1/downstream/screen_5k/o8_glt_graph/fusion_audit_units \
  --stage screen_5k

if ! "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('results/mts_glt_graphgate_v1/downstream/screen_5k/paired_summary.json').read_text())
go=p['macro_delta']>0 and p['median_task_delta']>0 and p['positive_tasks']>=2
Path('results/mts_glt_graphgate_v1/screening_decision.json').write_text(
    json.dumps({'go_20k':go, **{k:p[k] for k in ('macro_delta','median_task_delta','positive_tasks')}}, indent=2, sort_keys=True)+'\n'
)
raise SystemExit(0 if go else 3)
PY
then
  echo "GraphGate 5k screen did not pass; stopping before 20k." >&2
  exit 3
fi

make_stage 20000 graphgate_formal_20k \
  results/mts_glt_graphgate_v1/pretrain \
  pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth
EXPERIMENT_CONFIG=configs/mts/glt_graphgate_v1/generated/graphgate_formal_20k.json \
EXTRA_ARGS="--resume_state pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth.last.pt" \
  scripts/run_mips_trimer_scage.sh \
  2>&1 | tee logs/mts_glt_graphgate_v1/pretrain/step_5k_to_20k.log

CHECKPOINT=results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_020k.pth
TASKS=(eat eea egb egc ei eps nc xc)
FOLDS=(0 1 2 3 4)
for mode in o8_only o8_glt_graph; do
  "$PYTHON_BIN" scripts/run_mts_glt_graphgate_finetune.py \
    --checkpoint "$CHECKPOINT" --mode "$mode" --run-name formal_20k \
    --tasks "${TASKS[@]}" --folds "${FOLDS[@]}"
done
"$PYTHON_BIN" scripts/report_mts_glt_v2_downstream.py \
  --result-base results/mts_glt_graphgate_v1/downstream \
  --run-name formal_20k --tasks "${TASKS[@]}" --folds "${FOLDS[@]}" \
  --fused-mode o8_glt_graph \
  --output results/mts_glt_graphgate_v1/downstream/formal_20k/paired_summary.json
"$PYTHON_BIN" scripts/report_mts_glt_graphgate.py \
  --paired-summary results/mts_glt_graphgate_v1/downstream/formal_20k/paired_summary.json \
  --audit-root results/mts_glt_graphgate_v1/downstream/formal_20k/o8_glt_graph/fusion_audit_units \
  --stage formal_20k
