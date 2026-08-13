#!/usr/bin/env bash
set -euo pipefail

# One-shot continuation for the matched formal cycle.  It never starts a
# downstream writer before both 20k checkpoints have completion markers and
# pass the read-only identity audit, and it runs the two 8x5 campaigns in the
# required G0-then-G1 order.
ROOT=/root/workspace/Uni-Poly-Plus-master
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}
# Recovery mode: G1_PID is optional.  When unset or no longer alive the
# watcher skips the wait loop and proceeds directly from the audited
# checkpoints.  When set and alive it keeps the historical wait-then-run
# behavior.
G1_PID=${G1_PID:-}
BASE=$ROOT/results/mts_multiscale_topology/g_family_matched_v1
LOG_BASE=$ROOT/logs/mts_multiscale_topology/g_family_matched_v1
G0_CONFIG=$ROOT/configs/mts/experiments/G0_t1_msta_matched_formal_v1.json
G1_CONFIG=$ROOT/configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json
G0_CKPT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G0/mts_g0_pretrain_20k.pth
G1_CKPT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth
G0_RESULT=$BASE/G0/downstream_formal_v1
G1_RESULT=$BASE/G1/downstream_formal_v1
G0_LOG=$LOG_BASE/G0/downstream_formal_v1
G1_LOG=$LOG_BASE/G1/downstream_formal_v1
CHAIN_LOG=$LOG_BASE/auto_downstream_chain.log

mkdir -p "$(dirname "$CHAIN_LOG")"
exec > >(tee -a "$CHAIN_LOG") 2>&1
echo "[$(date -Is)] G1->downstream watcher started: g1_pid=${G1_PID:-unset}"

if [[ -n "$G1_PID" ]] && kill -0 "$G1_PID" 2>/dev/null; then
  while kill -0 "$G1_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date -Is)] G1 launcher exited; checking both checkpoint completion markers"
else
  echo "[$(date -Is)] recovery mode: no active G1 launcher PID; proceeding directly from audited checkpoints"
fi

for checkpoint in "$G0_CKPT" "$G1_CKPT"; do
  if [[ ! -f "$checkpoint" || ! -f "$checkpoint.complete.json" ]]; then
    echo "[$(date -Is)] missing checkpoint/completion marker: $checkpoint" >&2
    exit 20
  fi
done

CHECKPOINT_AUDIT=$BASE/checkpoint_audit.json
cd "$ROOT"
"$PYTHON_BIN" scripts/audit_mts_g0_g1_formal.py \
  --phase checkpoint \
  --g0-config "$G0_CONFIG" \
  --g1-config "$G1_CONFIG" \
  --g0-checkpoint "$G0_CKPT" \
  --g1-checkpoint "$G1_CKPT" \
  --output "$CHECKPOINT_AUDIT"
echo "[$(date -Is)] both checkpoint audits passed: $CHECKPOINT_AUDIT"

launch_downstream() {
  local arm="$1" config="$2" checkpoint="$3" result_root="$4" log_root="$5" window="$6"
  local exit_file="$result_root/launcher.exit"
  if [[ -e "$result_root" ]]; then
    echo "[$(date -Is)] refusing to overwrite downstream result root: $result_root" >&2
    exit 21
  fi
  if tmux list-windows -t Uni-Poly -F '#{window_name}' | grep -Fxq "$window"; then
    echo "[$(date -Is)] refusing duplicate downstream tmux window: $window" >&2
    exit 22
  fi
  mkdir -p "$result_root" "$log_root" "$result_root/artifacts"
  local command
  command="cd '$ROOT' && set -o pipefail && env PYTHON_BIN='$PYTHON_BIN' PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MTS_PRETRAIN_GPU_IDS=1,2,3 MTS_FINETUNE_GPU_IDS=0,1,2,3 EXPERIMENT_CONFIG='$config' FINETUNE_ONLY=1 PRETRAIN_ONLY=0 RESUME=0 RANDOM_SEED=42 FINETUNE_SEEDS=42 MTS_RUN_MULTI_SEED=0 TASKS='eat eea egb egc ei eps nc xc' FOLD_IDS='0 1 2 3 4' MTS_FINETUNE_SCHEDULE=lpt_v1 TRAIN_EPOCHS=100 MTS_FINETUNE_EPOCHS=100 MTS_FINETUNE_PATIENCE=10 MTS_FINETUNE_BATCH_SIZE=32 MTS_FINETUNE_EVAL_BATCH_SIZE=64 MTS_FINETUNE_AMP_DTYPE=fp32 FINETUNE_DATALOADER_WORKERS=2 DATALOADER_WORKERS=2 JOINT_CKPT='$checkpoint' ARTIFACT_DIR='$result_root/artifacts' RESULTS_DIR='$result_root' LOG_DIR='$log_root' bash scripts/run_mips_trimer_scage.sh 2>&1 | tee -a '$log_root/launcher.log'; rc=\${PIPESTATUS[0]}; printf '%s\\n' \"\$rc\" > '$exit_file'; echo \"[\$(date -Is)] ${arm} downstream launcher_exit=\$rc\"; exit \$rc"
  tmux new-window -d -t Uni-Poly -n "$window" "$command"
  echo "[$(date -Is)] started ${arm} downstream: window=Uni-Poly:${window} result=$result_root"
}

wait_downstream() {
  local arm="$1" result_root="$2" log_root="$3" window="$4"
  local exit_file="$result_root/launcher.exit"
  while [[ ! -f "$exit_file" ]]; do
    if ! tmux list-windows -t Uni-Poly -F '#{window_name}' | grep -Fxq "$window"; then
      echo "[$(date -Is)] ${arm} downstream window disappeared before launcher exit" >&2
      exit 23
    fi
    sleep 60
  done
  local rc
  rc=$(tr -d '[:space:]' < "$exit_file")
  if [[ "$rc" != 0 ]]; then
    tail -80 "$log_root/launcher.log" >&2 || true
    echo "[$(date -Is)] ${arm} downstream failed with exit=$rc" >&2
    exit 24
  fi
  local shards predictions
  shards=$(find "$result_root/shards/42" -mindepth 2 -maxdepth 2 -type f -name 'fold_*.csv' 2>/dev/null | wc -l | tr -d ' ')
  predictions=$(find "$result_root/predictions/42" -mindepth 2 -maxdepth 2 -type f -name 'fold_*.npz' 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(date -Is)] ${arm} downstream exited successfully: shards=$shards predictions=$predictions"
  if [[ "$shards" != 40 || "$predictions" != 40 ]]; then
    echo "[$(date -Is)] ${arm} downstream count gate failed" >&2
    exit 25
  fi
}

launch_downstream g0 "$G0_CONFIG" "$G0_CKPT" "$G0_RESULT" "$G0_LOG" mts-g0-finetune-formal-v1
wait_downstream g0 "$G0_RESULT" "$G0_LOG" mts-g0-finetune-formal-v1

launch_downstream g1 "$G1_CONFIG" "$G1_CKPT" "$G1_RESULT" "$G1_LOG" mts-g1-finetune-formal-v1
wait_downstream g1 "$G1_RESULT" "$G1_LOG" mts-g1-finetune-formal-v1

echo "[$(date -Is)] both downstream count gates passed; building G1-G0 report"
env PYTHONPATH=. "$PYTHON_BIN" scripts/compare_mts_g0_g1_formal.py \
  --g0-root "$G0_RESULT" \
  --g1-root "$G1_RESULT" \
  --g0-config "$G0_CONFIG" \
  --g1-config "$G1_CONFIG" \
  --g0-checkpoint "$G0_CKPT" \
  --g1-checkpoint "$G1_CKPT" \
  --output-json "$BASE/comparison.json" \
  --output-csv "$BASE/comparison.csv" \
  --output-md "$BASE/final_report.md" \
  --workers 2
echo "[$(date -Is)] formal downstream comparison completed"
