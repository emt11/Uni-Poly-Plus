#!/usr/bin/env bash
set -euo pipefail

# Background continuation monitor for the fresh-paired T-Pretrain-0 cycle.
# It never resumes or rewrites a training writer.  It waits for the existing
# T1 writer, validates its immutable output, then launches the two downstream
# arms in separate tmux windows and produces the paired comparison.  The
# handoff document is intentionally left for the final Codex review.

ROOT="/root/workspace/Uni-Poly-Plus-master"
SESSION="Uni-Poly"
BASE="${ROOT}/results/mts_multiscale_topology/t_pretrain0_v1"
MONITOR_LOG="${ROOT}/logs/mts_multiscale_topology/t_pretrain0_v1/background_monitor.log"
PAIR_ID="mts_t_pretrain0_matched_v1"
PYTHON_BIN="/opt/conda/envs/MTS/bin/python"
# Preserve the first failed downstream attempt as evidence.  A retry must use
# a fresh isolated result/log root rather than deleting or mutating that root.
DOWNSTREAM_ATTEMPT="${MTS_DOWNSTREAM_ATTEMPT:-retry1}"

T1_WINDOW="mts-tpretrain0-formal-T1"
T1_CKPT="${ROOT}/pretrained_models/mts_multiscale_topology/t_pretrain0_v1/T1_msta/mts_t_pretrain0_v1_T1_step20000.pth"
T1_COMPLETE="${T1_CKPT}.complete.json"
T1_REPORT="${BASE}/T1_msta/diagnostics/report.json"
T1_LOG="${ROOT}/logs/mts_multiscale_topology/t_pretrain0_v1/T1_msta/launcher.log"

T0_CKPT="${ROOT}/pretrained_models/mts_multiscale_topology/t_pretrain0_v1/T0_o8/mts_t_pretrain0_v1_T0_step20000.pth"
T0_CONFIG="${ROOT}/configs/mts/experiments/T0_o8_matched_t1_formal_v1.json"
T1_CONFIG="${ROOT}/configs/mts/experiments/T1_msta_formal_v1.json"

T0_DOWNSTREAM="${BASE}/T0_o8_downstream_${DOWNSTREAM_ATTEMPT}"
T1_DOWNSTREAM="${BASE}/T1_msta_downstream_${DOWNSTREAM_ATTEMPT}"
T0_DOWN_LOG="${ROOT}/logs/mts_multiscale_topology/t_pretrain0_v1/T0_o8_downstream_${DOWNSTREAM_ATTEMPT}"
T1_DOWN_LOG="${ROOT}/logs/mts_multiscale_topology/t_pretrain0_v1/T1_msta_downstream_${DOWNSTREAM_ATTEMPT}"

mkdir -p "$(dirname "$MONITOR_LOG")"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

window_running() {
  tmux list-windows -t "$SESSION" -F '#{window_name} #{pane_dead}' 2>/dev/null \
    | awk -v wanted="$1" '$1 == wanted && $2 == "0" { found=1 } END { exit !found }'
}

window_present() {
  tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null \
    | awk -v wanted="$1" '$1 == wanted { found=1 } END { exit !found }'
}

latest_progress() {
  local path="$1"
  rg '^\[pretrain\] stage=' "$path" 2>/dev/null | tail -1 || true
}

fail_blocked() {
  log "BLOCKED: $*"
  log "No downstream writer was started after this block."
  exit 20
}

wait_for_t1() {
  local ticks=0 progress
  log "Waiting for existing T1 writer: window=${T1_WINDOW}"
  while :; do
    if [[ -f "$T1_CKPT" && -f "$T1_COMPLETE" ]]; then
      log "T1 final checkpoint and complete manifest detected."
      return 0
    fi
    if ! window_running "$T1_WINDOW"; then
      if window_present "$T1_WINDOW"; then
        fail_blocked "T1 window exited before final checkpoint"
      fi
      fail_blocked "T1 window disappeared before final checkpoint"
    fi
    ticks=$((ticks + 1))
    if (( ticks % 2 == 0 )); then
      progress="$(latest_progress "$T1_LOG")"
      log "T1 still running${progress:+: $progress}"
    fi
    sleep 60
  done
}

# The short Python gate is kept outside the training code.  It checks the
# completed artifact, not a mutable process state, and emits a machine-readable
# gate result used by the continuation log.
validate_t1() {
  "$PYTHON_BIN" - "$T1_CKPT" "$T1_COMPLETE" "$T1_REPORT" "$PAIR_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
complete = Path(sys.argv[2])
report = Path(sys.argv[3])
pair_id = sys.argv[4]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

errors = []
complete_meta = json.loads(complete.read_text())
actual_sha = sha256(checkpoint)
if complete_meta.get('schema') != 'mts-pretrain-complete-v1':
    errors.append('complete schema mismatch')
if complete_meta.get('checkpoint_schema') != 'mts-model-v4':
    errors.append('checkpoint schema mismatch')
if complete_meta.get('checkpoint_sha256') != actual_sha:
    errors.append('complete checkpoint sha mismatch')
if complete_meta.get('optimizer_steps') != 20000:
    errors.append('complete optimizer step mismatch')

payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
meta = dict(payload.get('meta') or {})
if meta.get('model_identity') != 'T1':
    errors.append('model identity is not T1')
if meta.get('initialization') != 'fresh_paired':
    errors.append('initialization is not fresh_paired')
if meta.get('optimizer_steps') != 20000:
    errors.append('checkpoint optimizer step mismatch')
if meta.get('paired_init_id') != pair_id:
    errors.append('paired init id mismatch')
if meta.get('parent_checkpoint') not in (None, ''):
    errors.append('unexpected parent checkpoint')
source = dict(meta.get('source_contract') or {})
if source.get('initialization') != 'fresh_paired' or source.get('paired_init_id') != pair_id:
    errors.append('source contract initialization mismatch')

diagnostics = json.loads(report.read_text())
required = {0, 500, 2000, 5000, 10000, 20000}
observed = {int(value) for value in diagnostics.get('observed_steps', [])}
if not required.issubset(observed):
    errors.append(f'diagnostic milestones missing: {sorted(required - observed)}')
if diagnostics.get('all_rows_finite') is not True:
    errors.append('diagnostic rows are not all finite')
if diagnostics.get('probe_no_extra_backward') is not True:
    errors.append('probe extra-backward gate failed')
if diagnostics.get('probe_rng_restored') is not True:
    errors.append('probe RNG restoration gate failed')

result = {
    'schema': 'mts-t-pretrain0-background-gate-v1',
    'checkpoint': str(checkpoint),
    'checkpoint_sha256': actual_sha,
    'model_identity': meta.get('model_identity'),
    'initialization': meta.get('initialization'),
    'optimizer_steps': meta.get('optimizer_steps'),
    'paired_init_id': meta.get('paired_init_id'),
    'diagnostic_steps': sorted(observed),
    'all_rows_finite': diagnostics.get('all_rows_finite'),
    'probe_no_extra_backward': diagnostics.get('probe_no_extra_backward'),
    'probe_rng_restored': diagnostics.get('probe_rng_restored'),
    'pass': not errors,
    'errors': errors,
}
print(json.dumps(result, sort_keys=True))
if errors:
    raise SystemExit(1)
PY
}

validate_downstream() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
tasks = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
expected = {(task, fold) for task in tasks for fold in range(5)}
shards = sorted((root / 'shards' / '42').glob('*/fold_*.csv'))
predictions = sorted((root / 'predictions' / '42').glob('*/fold_*.npz'))
seen_shards = set()
seen_predictions = set()
errors = []
for path in shards:
    try:
        key = (path.parent.name, int(path.stem.removeprefix('fold_')))
    except ValueError:
        errors.append(f'invalid shard path: {path}')
        continue
    if key in seen_shards:
        errors.append(f'duplicate shard: {path}')
    seen_shards.add(key)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        errors.append(f'shard row count: {path}')
    if not np.isfinite(float(frame.iloc[0].get('avg_test_r2', np.nan))):
        errors.append(f'non-finite avg_test_r2: {path}')
for path in predictions:
    try:
        key = (path.parent.name, int(path.stem.removeprefix('fold_')))
    except ValueError:
        errors.append(f'invalid prediction path: {path}')
        continue
    if key in seen_predictions:
        errors.append(f'duplicate prediction: {path}')
    seen_predictions.add(key)
    with np.load(path, allow_pickle=False) as arrays:
        y_true = np.asarray(arrays['y_true'])
        y_pred = np.asarray(arrays['y_pred'])
        if y_true.shape != y_pred.shape or not y_true.size:
            errors.append(f'prediction shape: {path}')
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            errors.append(f'non-finite prediction: {path}')
if seen_shards != expected:
    errors.append(f'shard keys mismatch: missing={sorted(expected - seen_shards)} extra={sorted(seen_shards - expected)}')
if seen_predictions != expected:
    errors.append(f'prediction keys mismatch: missing={sorted(expected - seen_predictions)} extra={sorted(seen_predictions - expected)}')
result = {
    'root': str(root),
    'verified_shards': len(seen_shards),
    'verified_predictions': len(seen_predictions),
    'pass': not errors,
    'errors': errors,
}
print(json.dumps(result, sort_keys=True))
if errors:
    raise SystemExit(1)
PY
}

launch_downstream() {
  local arm="$1" config="$2" checkpoint="$3" result_root="$4" log_root="$5" window="$6"
  local exit_file="${result_root}/launcher.exit"
  local launcher_log="${log_root}/launcher.log"
  if [[ -e "$result_root" ]]; then
    fail_blocked "downstream result root already exists: $result_root"
  fi
  if window_present "$window"; then
    fail_blocked "downstream tmux window already exists: $window"
  fi
  mkdir -p "$log_root"
  local command
  command="cd '$ROOT' && env PYTHON_BIN='$PYTHON_BIN' PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MTS_PRETRAIN_GPU_IDS=1,2,3 MTS_FINETUNE_GPU_IDS=0,1,2,3 EXPERIMENT_CONFIG='$config' FINETUNE_ONLY=1 PRETRAIN_ONLY=0 RESUME=0 RANDOM_SEED=42 FINETUNE_SEEDS=42 MTS_RUN_MULTI_SEED=0 TASKS='eat eea egb egc ei eps nc xc' FOLD_IDS='0 1 2 3 4' MTS_FINETUNE_SCHEDULE=lpt_v1 TRAIN_EPOCHS=100 MTS_FINETUNE_EPOCHS=100 MTS_FINETUNE_PATIENCE=10 MTS_FINETUNE_BATCH_SIZE=32 MTS_FINETUNE_EVAL_BATCH_SIZE=64 MTS_FINETUNE_AMP_DTYPE=fp32 FINETUNE_DATALOADER_WORKERS=2 DATALOADER_WORKERS=2 JOINT_CKPT='$checkpoint' RESULTS_DIR='$result_root' LOG_DIR='$log_root' bash scripts/run_mips_trimer_scage.sh 2>&1 | tee -a '$launcher_log'; rc=\${PIPESTATUS[0]}; printf '%s\\n' \"\$rc\" > '$exit_file'; echo \"[tmux] ${arm} launcher_exit=\$rc\"; exit \$rc"
  tmux new-window -d -t "$SESSION" -n "$window" "$command"
  log "Launched downstream ${arm}: window=${window}, checkpoint=${checkpoint}, log=${launcher_log}"
}

wait_for_downstream() {
  local arm="$1" result_root="$2" log_root="$3" window="$4"
  local exit_file="${result_root}/launcher.exit" ticks=0 rc progress
  while [[ ! -f "$exit_file" ]]; do
    if ! window_running "$window"; then
      if window_present "$window"; then
        fail_blocked "${arm} downstream window exited before launcher.exit"
      fi
      fail_blocked "${arm} downstream window disappeared before launcher.exit"
    fi
    ticks=$((ticks + 1))
    if (( ticks % 2 == 0 )); then
      progress="$(find "$result_root/shards/42" -name 'fold_*.csv' 2>/dev/null | wc -l | tr -d ' ')"
      log "${arm} downstream running: verified-looking shard files=${progress}"
    fi
    sleep 60
  done
  rc="$(tr -d '[:space:]' < "$exit_file")"
  if [[ "$rc" != "0" ]]; then
    tail -80 "$log_root/launcher.log" >&2 || true
    fail_blocked "${arm} downstream launcher exited with status ${rc}"
  fi
  log "${arm} downstream launcher exited successfully; validating 40 shards and predictions."
  validate_downstream "$result_root" | tee -a "$MONITOR_LOG"
}

main() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    fail_blocked "tmux session ${SESSION} is missing"
  fi
  log "Background monitor started for cycle mts_t_pretrain0_matched_v1."
  wait_for_t1
  local gate
  if ! gate="$(validate_t1)"; then
    printf '%s\n' "$gate" | tee "${BASE}/T1_msta/diagnostics/monitor_gate.json" >> "$MONITOR_LOG"
    fail_blocked "T1 artifact/diagnostics gate failed"
  fi
  printf '%s\n' "$gate" | tee "${BASE}/T1_msta/diagnostics/monitor_gate.json" >> "$MONITOR_LOG"
  log "T-Diagnosis gate passed; proceeding to fresh-paired downstream reevaluation."

  launch_downstream "T0" "$T0_CONFIG" "$T0_CKPT" "$T0_DOWNSTREAM" "$T0_DOWN_LOG" "mts-tpretrain0-downstream-T0"
  wait_for_downstream "T0" "$T0_DOWNSTREAM" "$T0_DOWN_LOG" "mts-tpretrain0-downstream-T0"

  launch_downstream "T1" "$T1_CONFIG" "$T1_CKPT" "$T1_DOWNSTREAM" "$T1_DOWN_LOG" "mts-tpretrain0-downstream-T1"
  wait_for_downstream "T1" "$T1_DOWNSTREAM" "$T1_DOWN_LOG" "mts-tpretrain0-downstream-T1"

  log "Both downstream arms passed 40-shard validation; running fresh-paired comparison."
  "$PYTHON_BIN" scripts/compare_mts_t0_t1_formal.py \
    --t0-root "$T0_DOWNSTREAM" \
    --t1-root "$T1_DOWNSTREAM" \
    --t0-config "$T0_CONFIG" \
    --t1-config "$T1_CONFIG" \
    --t0-checkpoint "$T0_CKPT" \
    --t1-checkpoint "$T1_CKPT" \
    --fresh-paired \
    --paired-init-id "$PAIR_ID" \
    --output-json "${BASE}/formal_comparison.json" \
    --output-csv "${BASE}/formal_comparison.csv" \
    --execution-summary "${BASE}/execution_summary.json" \
    --tmux-session "$SESSION" \
    --t0-window "mts-tpretrain0-downstream-T0" \
    --t1-window "mts-tpretrain0-downstream-T1" \
    --t0-log-dir "$T0_DOWN_LOG" \
    --t1-log-dir "$T1_DOWN_LOG" \
    --workers 2 | tee -a "$MONITOR_LOG"
  log "Comparison complete. Monitor stopped for independent Codex review; handoff document was not edited."
}

main "$@"
