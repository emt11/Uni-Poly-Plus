#!/bin/bash
# Sequential launcher for the D2 arms of GLT-3D-GAIN-20260921-01.
#
# Failure contract: the log is appended, never truncated; a subprocess failure
# stops the loop immediately and propagates that subprocess's own exit code; the
# completion marker is written only when every arm succeeded.  The arm list, the
# runner and the per-run knobs are environment-overridable so the failure rules
# can be exercised with a stub runner, without touching real training budgets.
set -u
cd /root/workspace/Uni-Poly-Plus-master

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-scripts/finetune_glt_3d_gain_d2.py}"
ARMS="${ARMS:-f2d fbase fnorm fstable}"
STAGE="${STAGE:-smoke}"
TASK="${TASK:-xc}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-1}"
OUTPUT="${OUTPUT:-results/glt_3d_gain_20260921/d2_smoke}"
LOG="${LOG:-logs/glt_3d_gain_20260921/d2_smoke.log}"

mkdir -p "$(dirname "$LOG")"
echo "=== RUNNER START stage=$STAGE task=$TASK fold=$FOLD epochs=$EPOCHS arms='$ARMS' $(date -u +%FT%TZ) ===" >> "$LOG"
for arm in $ARMS; do
  echo "=== ARM=$arm START $(date -u +%FT%TZ) ===" >> "$LOG"
  "$PYTHON" "$RUNNER" \
    --arm "$arm" \
    --stage "$STAGE" \
    --config configs/mts/glt_pred_s3b_b_fp.json \
    --checkpoint results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt \
    --cohort-root data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1 \
    --cache-root data/processed/mips_trimer_scage_downstream \
    --dual-static-root data/processed/glt_dual_v2/downstream/dual_static_v1 \
    --split-root data/splits/mips_outer5_inner20 \
    --task "$TASK" --fold "$FOLD" --epochs "$EPOCHS" \
    --output "$OUTPUT" >> "$LOG" 2>&1
  code=$?
  echo "=== ARM=$arm EXIT=$code $(date -u +%FT%TZ) ===" >> "$LOG"
  if [ "$code" -ne 0 ]; then
    echo "=== ABORT after $arm (exit $code); no rerun within this budget $(date -u +%FT%TZ) ===" >> "$LOG"
    exit "$code"
  fi
done
echo "=== RUNNER DONE ALL_ARMS_OK $(date -u +%FT%TZ) ===" >> "$LOG"
exit 0
