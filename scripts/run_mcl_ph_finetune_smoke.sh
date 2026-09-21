#!/bin/bash
# Sequential launcher for the five MCL-PH downstream smoke arms
# (MCL-PH-20260921-01/r1, P1): one epoch of XC/fold0 per arm.
#
# Failure contract: the log is appended, never truncated; a subprocess failure
# stops the loop immediately and propagates that subprocess's own exit code; the
# completion marker is written only when every arm succeeded.  RUNNER, ARMS and
# the per-run knobs are environment-overridable so the failure rules can be
# exercised with a stub runner instead of consuming real budget.
set -u
cd /root/workspace/Uni-Poly-Plus-master

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-scripts/finetune_mcl_ph.py}"
ARMS="${ARMS:-glt_ref o8_only m_cat m_gate m_xattn}"
TASK="${TASK:-xc}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-1}"
STEP="${STEP:-2}"
PRETRAIN="${PRETRAIN:-results/mcl_ph_20260921/p1/pretrain}"
OUTPUT="${OUTPUT:-results/mcl_ph_20260921/p1/finetune}"
LOG="${LOG:-logs/mcl_ph_20260921/p1_finetune_smoke.log}"
STATISTICS="${STATISTICS:-results/mcl_ph_20260921/p0/statistics.npz}"
CONFIG="${CONFIG:-configs/mts/mcl_ph_gate.json}"
COHORT="${COHORT:-data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1}"
CACHE="${CACHE:-data/processed/mips_trimer_scage_downstream}"
STATIC="${STATIC:-data/processed/glt_dual_v2/downstream/dual_static_v1}"
SPLIT="${SPLIT:-data/splits/mips_outer5_inner20}"

mkdir -p "$(dirname "$LOG")"
echo "=== RUNNER START stage=smoke task=$TASK fold=$FOLD epochs=$EPOCHS arms='$ARMS' $(date -u +%FT%TZ) ===" >> "$LOG"
for arm in $ARMS; do
  case "$arm" in
    glt_ref|o8_only) PACKAGE="$PRETRAIN/glt_ref/deploy_$(printf '%05d' "$STEP").pt" ;;
    m_cat)           PACKAGE="$PRETRAIN/cat/deploy_$(printf '%05d' "$STEP").pt" ;;
    m_gate)          PACKAGE="$PRETRAIN/gate/deploy_$(printf '%05d' "$STEP").pt" ;;
    m_xattn)         PACKAGE="$PRETRAIN/xattn/deploy_$(printf '%05d' "$STEP").pt" ;;
    *) echo "=== ABORT: unknown arm $arm $(date -u +%FT%TZ) ===" >> "$LOG"; exit 2 ;;
  esac
  echo "=== ARM=$arm START package=$PACKAGE $(date -u +%FT%TZ) ===" >> "$LOG"
  "$PYTHON" "$RUNNER" \
    --arm "$arm" --stage smoke \
    --config "$CONFIG" --checkpoint "$PACKAGE" --expected-pretrain-step "$STEP" \
    --cohort-root "$COHORT" --cache-root "$CACHE" --dual-static-root "$STATIC" \
    --split-root "$SPLIT" --statistics "$STATISTICS" \
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
