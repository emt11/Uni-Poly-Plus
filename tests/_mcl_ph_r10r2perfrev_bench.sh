#!/bin/bash
# r10R2-perf reverse-order cross-over: cached first, online second, per arm.
#
# Six scratch starts of the real trainer, one variable (--trajectory-cache).  This
# file is the previous launcher with the mode order swapped and a new scratch root,
# so the two orders together separate the cache effect from run order / warm cache.
set -u
cd /root/workspace/Uni-Poly-Plus-master

LOG_DIR=logs/mcl_ph_20260921
LOG="$LOG_DIR/p2r2perf_bench_rev.log"
ROOT=results/mcl_ph_20260921/p2r2_perf_bench_rev
CACHE=data/processed/mcl_ph_cache/p2_noisy_seed42_sigma003_step5000_v1
INIT=results/mcl_ph_20260921/p2/pretrain/shared_new_init.pt
mkdir -p "$LOG_DIR" "$ROOT"

cat > "$ROOT/README.md" <<'EOF'
PERF BENCH scratch root (r10R2-perf, reverse order).

Six 30-update starts of the real `scripts/pretrain_mcl_ph.py`, paired per arm with
the cached member first and the online member second.  Together with
`../p2r2_perf_bench/` (online first) this is the cross-over: each arm is measured
once in each order.  These are performance runs, not training runs: not a P2 arm,
not a model ranking, not part of the P2 update budget.
EOF

COMMON=(--cohort-root data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1
        --cache-root data/processed/mips_trimer_scage
        --dual-static-root data/processed/glt_dual_v2/pi1m/dual_static_v1
        --statistics results/mcl_ph_20260921/p0/statistics.npz
        --shared-new-init "$INIT"
        --diagnostics --prep-workers 12 --stop-after-step 30)

{
  echo "=== PERF BENCH REV START $(date -u +%FT%TZ) baseline_commit=$(git rev-parse HEAD)"
  echo "=== shared_new_init_sha256_before=$(sha256sum "$INIT" | cut -d' ' -f1)"
  echo "=== cache_manifest=$(python3 -c "import json;d=json.load(open('$CACHE/manifest.json'));print('complete',d['complete'],'shards',len(d['shards']),'positions',d['total_positions'])")"
} >> "$LOG"

for arm in cat gate xattn; do
  for mode in cached online; do
    out="$ROOT/${arm}_${mode}"
    extra=()
    if [ "$mode" = cached ]; then extra=(--trajectory-cache "$CACHE"); fi
    started=$(date -u +%s.%N)
    echo "=== PERF BENCH REV RUN arm=$arm mode=$mode start=$(date -u +%FT%TZ) output=$out" >> "$LOG"
    python3 -m torch.distributed.run --nproc_per_node=4 --standalone \
      scripts/pretrain_mcl_ph.py --config "configs/mts/mcl_ph_${arm}.json" \
      "${COMMON[@]}" "${extra[@]}" --output "$out" \
      > "$ROOT/${arm}_${mode}.stdout.log" 2>&1
    code=$?
    finished=$(date -u +%s.%N)
    wall=$(awk -v a="$started" -v b="$finished" 'BEGIN{printf "%.3f", b-a}')
    echo "=== PERF BENCH REV RUN arm=$arm mode=$mode EXIT=$code wall_seconds=$wall end=$(date -u +%FT%TZ)" >> "$LOG"
    sleep 15
  done
done

echo "=== shared_new_init_sha256_after=$(sha256sum "$INIT" | cut -d' ' -f1)" >> "$LOG"
echo "=== PERF BENCH REV END $(date -u +%FT%TZ)" >> "$LOG"
