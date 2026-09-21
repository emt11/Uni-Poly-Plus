#!/bin/bash
cd /root/workspace/Uni-Poly-Plus-master
LOG=logs/glt_3d_gain_20260921/d2_smoke.log
: > "$LOG"
for arm in f2d fbase fnorm fstable; do
  echo "=== ARM=$arm START $(date -u +%FT%TZ) ===" >> "$LOG"
  python3 scripts/finetune_glt_3d_gain_d2.py \
    --arm "$arm" \
    --config configs/mts/glt_pred_s3b_b_fp.json \
    --checkpoint results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt \
    --cohort-root data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1 \
    --cache-root data/processed/mips_trimer_scage_downstream \
    --dual-static-root data/processed/glt_dual_v2/downstream/dual_static_v1 \
    --split-root data/splits/mips_outer5_inner20 \
    --task xc --fold 0 --epochs 1 \
    --output results/glt_3d_gain_20260921/d2_smoke >> "$LOG" 2>&1
  code=$?
  echo "=== ARM=$arm EXIT=$code $(date -u +%FT%TZ) ===" >> "$LOG"
  if [ "$code" -ne 0 ]; then
    echo "=== STOPPING after $arm (non-zero exit); no rerun within this budget ===" >> "$LOG"
    break
  fi
done
echo "=== RUNNER DONE $(date -u +%FT%TZ) ===" >> "$LOG"
