#!/usr/bin/env bash
set -euo pipefail

# One-shot G2 downstream continuation for the matched G1/G2 formal cycle.
# It never starts a G2 writer before the G1 baseline is verified read-only
# (40 shards + 40 predictions) and the G2 checkpoint passes the strict
# G1/G2 audit.  The G2 8x5 campaign runs in its own fresh result directory;
# afterwards the G2-G1 formal comparison is built into G1_vs_G2/.
ROOT=/root/workspace/Uni-Poly-Plus-master
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}
BASE=$ROOT/results/mts_multiscale_topology/g_family_matched_v1
LOG_BASE=$ROOT/logs/mts_multiscale_topology/g_family_matched_v1
G1_CONFIG=$ROOT/configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json
G2_CONFIG=$ROOT/configs/mts/experiments/G2_t1_msta_cosine_distance_matched_formal_v1.json
G1_CKPT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth
G2_CKPT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G2/mts_g2_pretrain_20k.pth
G1_RESULT=$BASE/G1/downstream_formal_v1
G2_RESULT=$BASE/G2/downstream_formal_v1
G1_LOG=$LOG_BASE/G1/downstream_formal_v1
G2_LOG=$LOG_BASE/G2/downstream_formal_v1
REPORT=$BASE/G1_vs_G2
CHAIN_LOG=$LOG_BASE/G2_auto_downstream.log

mkdir -p "$(dirname "$CHAIN_LOG")"
exec > >(tee -a "$CHAIN_LOG") 2>&1
echo "[$(date -Is)] G2->downstream watcher started"

# Gate 0: the G1 baseline must already be the verified 40+40 arm (read-only).
if [[ ! -f "$G1_CKPT" || ! -f "$G1_CKPT.complete.json" ]]; then
  echo "[$(date -Is)] missing G1 baseline checkpoint/completion marker" >&2
  exit 30
fi
g1_shards=$(find "$G1_RESULT/shards/42" -mindepth 2 -maxdepth 2 -type f -name 'fold_*.csv' 2>/dev/null | wc -l | tr -d ' ')
g1_predictions=$(find "$G1_RESULT/predictions/42" -mindepth 2 -maxdepth 2 -type f -name 'fold_*.npz' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$g1_shards" != 40 || "$g1_predictions" != 40 ]]; then
  echo "[$(date -Is)] G1 baseline count gate failed: shards=$g1_shards predictions=$g1_predictions" >&2
  exit 31
fi
echo "[$(date -Is)] G1 baseline verified read-only: shards=$g1_shards predictions=$g1_predictions"

# Gate 1: G2 checkpoint must pass the strict G1/G2 audit.
cd "$ROOT"
"$PYTHON_BIN" scripts/audit_mts_g1_g2_formal.py \
  --phase checkpoint \
  --g1-config "$G1_CONFIG" \
  --g2-config "$G2_CONFIG" \
  --g1-checkpoint "$G1_CKPT" \
  --g2-checkpoint "$G2_CKPT" \
  --output "$REPORT/checkpoint_audit.json"
echo "[$(date -Is)] G1/G2 checkpoint audits passed: $REPORT/checkpoint_audit.json"

if [[ -e "$G2_RESULT" ]]; then
  echo "[$(date -Is)] refusing to overwrite G2 downstream result root: $G2_RESULT" >&2
  exit 32
fi
if tmux list-windows -t Uni-Poly -F '#{window_name}' | grep -Fxq mts-g2-finetune-formal-v1; then
  echo "[$(date -Is)] refusing duplicate downstream tmux window: mts-g2-finetune-formal-v1" >&2
  exit 33
fi
mkdir -p "$G2_RESULT" "$G2_LOG" "$G2_RESULT/artifacts"
command="cd '$ROOT' && set -o pipefail && env PYTHON_BIN='$PYTHON_BIN' PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MTS_PRETRAIN_GPU_IDS=1,2,3 MTS_FINETUNE_GPU_IDS=0,1,2,3 EXPERIMENT_CONFIG='$G2_CONFIG' FINETUNE_ONLY=1 PRETRAIN_ONLY=0 RESUME=0 RANDOM_SEED=42 FINETUNE_SEEDS=42 MTS_RUN_MULTI_SEED=0 TASKS='eat eea egb egc ei eps nc xc' FOLD_IDS='0 1 2 3 4' MTS_FINETUNE_SCHEDULE=lpt_v1 TRAIN_EPOCHS=100 MTS_FINETUNE_EPOCHS=100 MTS_FINETUNE_PATIENCE=10 MTS_FINETUNE_BATCH_SIZE=32 MTS_FINETUNE_EVAL_BATCH_SIZE=64 MTS_FINETUNE_AMP_DTYPE=fp32 FINETUNE_DATALOADER_WORKERS=2 DATALOADER_WORKERS=2 JOINT_CKPT='$G2_CKPT' ARTIFACT_DIR='$G2_RESULT/artifacts' RESULTS_DIR='$G2_RESULT' LOG_DIR='$G2_LOG' bash scripts/run_mips_trimer_scage.sh 2>&1 | tee -a '$G2_LOG/launcher.log'; rc=\${PIPESTATUS[0]}; printf '%s\\n' \"\$rc\" > '$G2_RESULT/launcher.exit'; echo \"[\$(date -Is)] G2 downstream launcher_exit=\$rc\"; exit \$rc"
tmux new-window -d -t Uni-Poly -n mts-g2-finetune-formal-v1 "$command"
echo "[$(date -Is)] started G2 downstream: window=Uni-Poly:mts-g2-finetune-formal-v1 result=$G2_RESULT"

while [[ ! -f "$G2_RESULT/launcher.exit" ]]; do
  if ! tmux list-windows -t Uni-Poly -F '#{window_name}' | grep -Fxq mts-g2-finetune-formal-v1; then
    echo "[$(date -Is)] G2 downstream window disappeared before launcher exit" >&2
    exit 34
  fi
  sleep 60
done
rc=$(tr -d '[:space:]' < "$G2_RESULT/launcher.exit")
if [[ "$rc" != 0 ]]; then
  tail -80 "$G2_LOG/launcher.log" >&2 || true
  echo "[$(date -Is)] G2 downstream failed with exit=$rc" >&2
  exit 35
fi
g2_shards=$(find "$G2_RESULT/shards/42" -mindepth 2 -maxdepth 2 -type f -name 'fold_*.csv' 2>/dev/null | wc -l | tr -d ' ')
g2_predictions=$(find "$G2_RESULT/predictions/42" -mindepth 2 -maxdepth 2 -type f -name 'fold_*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date -Is)] G2 downstream exited successfully: shards=$g2_shards predictions=$g2_predictions"
if [[ "$g2_shards" != 40 || "$g2_predictions" != 40 ]]; then
  echo "[$(date -Is)] G2 downstream count gate failed" >&2
  exit 36
fi

echo "[$(date -Is)] G2 count gates passed; building G2-G1 report"
env PYTHONPATH=. "$PYTHON_BIN" scripts/compare_mts_g1_g2_formal.py \
  --g1-root "$G1_RESULT" \
  --g2-root "$G2_RESULT" \
  --g1-config "$G1_CONFIG" \
  --g2-config "$G2_CONFIG" \
  --g1-checkpoint "$G1_CKPT" \
  --g2-checkpoint "$G2_CKPT" \
  --output-json "$REPORT/comparison.json" \
  --output-csv "$REPORT/comparison.csv" \
  --output-md "$REPORT/final_report.md" \
  --workers 2
echo "[$(date -Is)] formal G2-G1 comparison completed"
