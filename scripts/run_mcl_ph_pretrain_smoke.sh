#!/bin/bash
# Sequential launcher for the four MCL-PH pre-training smoke paths
# (MCL-PH-20260921-01/r1, P1).
#
# Failure contract: the log is appended, never truncated; a subprocess failure
# stops the loop immediately and propagates that subprocess's own exit code; a
# successful exit code is then checked against the arm's own products
# (scripts/verify_mcl_ph_arm.py: runtime record PASS, cleanup finished, required
# artifacts present, completed updates matching UPDATES); the completion marker
# is written only when every path passed both checks.  RUNNER, ARMS and the
# per-run knobs are environment-overridable so the failure rules can be
# exercised with a stub runner instead of consuming real budget.
set -u
cd /root/workspace/Uni-Poly-Plus-master

PYTHON="${PYTHON:-python3}"
MCL_RUNNER="${MCL_RUNNER:-scripts/pretrain_mcl_ph.py}"
REF_RUNNER="${REF_RUNNER:-scripts/pretrain_glt_dual.py}"
ARMS="${ARMS:-glt_ref cat gate xattn}"
UPDATES="${UPDATES:-2}"
OUTPUT="${OUTPUT:-results/mcl_ph_20260921/p1/pretrain}"
LOG="${LOG:-logs/mcl_ph_20260921/p1_pretrain_smoke.log}"
STATISTICS="${STATISTICS:-results/mcl_ph_20260921/p0/statistics.npz}"
COHORT="${COHORT:-data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1}"
CACHE="${CACHE:-data/processed/mips_trimer_scage}"
STATIC="${STATIC:-data/processed/glt_dual_v2/pi1m/dual_static_v1}"
TARGETS="${TARGETS:-data/processed/glt_dual_v2/pi1m/pretrain_targets_v1}"
CONFIG="${CONFIG:-configs/mts/glt_pred_s3b_b_fp.json}"
NPROC="${NPROC:-4}"
PREP_WORKERS="${PREP_WORKERS:-12}"
DOWNSTREAM_COHORT="${DOWNSTREAM_COHORT:-data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1}"

mkdir -p "$(dirname "$LOG")"
echo "=== RUNNER START updates=$UPDATES arms='$ARMS' $(date -u +%FT%TZ) ===" >> "$LOG"
for arm in $ARMS; do
  echo "=== ARM=$arm START $(date -u +%FT%TZ) ===" >> "$LOG"
  if [ "$arm" = "glt_ref" ]; then
    "$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" --standalone "$REF_RUNNER" \
      --config "$CONFIG" \
      --cohort-root "$COHORT" --cache-root "$CACHE" \
      --dual-static-root "$STATIC" --pretrain-target-root "$TARGETS" \
      --third-task fp --diagnostics --diagnostic-save-steps "$UPDATES" \
      --stop-after-step "$UPDATES" --prep-workers "$PREP_WORKERS" \
      --output "$OUTPUT/$arm" >> "$LOG" 2>&1
  else
    "$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" --standalone "$MCL_RUNNER" \
      --config "configs/mts/mcl_ph_$arm.json" \
      --cohort-root "$COHORT" --cache-root "$CACHE" \
      --dual-static-root "$STATIC" --statistics "$STATISTICS" \
      --shared-new-init "$OUTPUT/shared_new_init.pt" \
      --diagnostics --stop-after-step "$UPDATES" --prep-workers "$PREP_WORKERS" \
      --output "$OUTPUT/$arm" >> "$LOG" 2>&1
  fi
  code=$?
  echo "=== ARM=$arm EXIT=$code $(date -u +%FT%TZ) ===" >> "$LOG"
  if [ "$code" -ne 0 ]; then
    echo "=== ABORT after $arm (exit $code); no rerun within this budget $(date -u +%FT%TZ) ===" >> "$LOG"
    exit "$code"
  fi
  # An exit code of 0 is necessary but not sufficient: the arm is accepted only
  # if its own products say the same thing. A record left at TRAINING_COMPLETE
  # (training and export done, cleanup unfinished, no deploy package) is the
  # shape the r3 hang left behind and must never be promoted to success here.
  if [ "$arm" = "glt_ref" ]; then
    "$PYTHON" scripts/verify_mcl_ph_arm.py --label "$arm" --arm-dir "$OUTPUT/$arm" \
      --updates "$UPDATES" >> "$LOG" 2>&1
  else
    "$PYTHON" scripts/verify_mcl_ph_arm.py --label "$arm" --arm-dir "$OUTPUT/$arm" \
      --updates "$UPDATES" --strict-cleanup >> "$LOG" 2>&1
  fi
  vcode=$?
  echo "=== ARM=$arm VERIFY=$vcode $(date -u +%FT%TZ) ===" >> "$LOG"
  if [ "$vcode" -ne 0 ]; then
    echo "=== ABORT after $arm (verification $vcode); no rerun within this budget $(date -u +%FT%TZ) ===" >> "$LOG"
    exit 6
  fi
done
echo "=== RUNNER DONE ALL_ARMS_OK $(date -u +%FT%TZ) ===" >> "$LOG"
exit 0
