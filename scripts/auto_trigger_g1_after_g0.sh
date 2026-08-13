#!/usr/bin/env bash
set -euo pipefail

# This is a one-shot orchestration watcher for the matched formal cycle.  It
# does not restart or resume G0, and it never launches G1 before the G0 final
# checkpoint has a completion marker and passes the strict identity audit.
ROOT=/root/workspace/Uni-Poly-Plus-master
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}
G0_PID=${G0_PID:?set G0_PID to the active G0 launcher PID}
CHAIN_LOG=${CHAIN_LOG:-$ROOT/logs/mts_multiscale_topology/g_family_matched_v1/auto_chain.log}
G0_CONFIG=$ROOT/configs/mts/experiments/G0_t1_msta_matched_formal_v1.json
G1_CONFIG=$ROOT/configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json
G0_CKPT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G0/mts_g0_pretrain_20k.pth
G1_CKPT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth
G0_ROOT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G0
G1_ROOT=$ROOT/pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1
G1_LOG=$ROOT/logs/mts_multiscale_topology/g_family_matched_v1/G1
G1_RESULT=$ROOT/results/mts_multiscale_topology/g_family_matched_v1/G1

mkdir -p "$(dirname "$CHAIN_LOG")"
exec > >(tee -a "$CHAIN_LOG") 2>&1
echo "[$(date -Is)] G0->G1 watcher started: g0_pid=$G0_PID"

while kill -0 "$G0_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date -Is)] G0 launcher exited; checking completion marker"

if [[ ! -f "$G0_CKPT" || ! -f "$G0_CKPT.complete.json" ]]; then
  echo "[$(date -Is)] G0 failed: checkpoint/completion marker missing; G1 not started" >&2
  exit 20
fi

G0_AUDIT=$ROOT/results/mts_multiscale_topology/g_family_matched_v1/G0/checkpoint_audit.json
cd "$ROOT"
"$PYTHON_BIN" - "$G0_CONFIG" "$G0_CKPT" "$G0_AUDIT" <<'PY'
import json
import sys
from pathlib import Path
from scripts.audit_mts_g0_g1_formal import audit_checkpoint, audit_configs

config_path, checkpoint_path, output_path = map(Path, sys.argv[1:])
report = audit_configs(
    Path("configs/mts/experiments/G0_t1_msta_matched_formal_v1.json"),
    Path("configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json"),
)
report["G0_checkpoint"] = audit_checkpoint(checkpoint_path, "g0", report["G0"])
report["trigger"] = "g1_after_g0_checkpoint_audit"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": report["status"], "audit": str(output_path)}, indent=2))
PY
echo "[$(date -Is)] G0 checkpoint audit passed"

if [[ -e "$G1_CKPT" || -e "$G1_CKPT.complete.json" || -e "$G1_CKPT.last.pt" ]]; then
  echo "[$(date -Is)] G1 target already exists; refusing automatic overwrite" >&2
  exit 21
fi
mkdir -p "$G1_ROOT" "$G1_LOG" "$G1_RESULT"
if tmux list-windows -t Uni-Poly -F '#{window_name}' | grep -Fxq mts-g1-pretrain-formal-v1; then
  echo "[$(date -Is)] G1 window already exists; refusing duplicate writer" >&2
  exit 22
fi

G1_COMMAND="cd $ROOT && set -o pipefail && env PYTHON_BIN=$PYTHON_BIN EXPERIMENT_CONFIG=configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json PRETRAIN_ONLY=1 MTS_PRETRAIN_GPU_IDS=1,2,3 MTS_FINETUNE_GPU_IDS=0,1,2,3 JOINT_CKPT=pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth ARTIFACT_DIR=pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1 LOG_DIR=logs/mts_multiscale_topology/g_family_matched_v1/G1 RESULTS_DIR=results/mts_multiscale_topology/g_family_matched_v1/G1 MTS_FINETUNE_SCHEDULE=lpt_v1 bash scripts/run_mips_trimer_scage.sh 2>&1 | tee logs/mts_multiscale_topology/g_family_matched_v1/G1/launcher.log"
tmux new-window -t Uni-Poly -n mts-g1-pretrain-formal-v1
tmux send-keys -t Uni-Poly:mts-g1-pretrain-formal-v1 "bash -lc '$G1_COMMAND'" C-m
echo "[$(date -Is)] started G1 in tmux Uni-Poly:mts-g1-pretrain-formal-v1"
