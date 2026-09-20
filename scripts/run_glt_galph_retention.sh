#!/usr/bin/env bash
# PH retention launcher: runs the three groups one after another and stops at the
# first failure.
#
# The real exit code of each training process is captured on the line right after
# it runs (nothing is allowed between the command and `code=$?`), the status line
# is written with that code, and the chain stops immediately when it is non-zero.
# A run that failed can therefore never be followed by another group and can never
# produce an ALL_DONE marker.
#
# usage:
#   run_glt_galph_retention.sh --output-root DIR --status-log FILE \
#       --config CONFIG --protocol smoke|development [--tasks "xc eps eat"] \
#       [--folds "0 1"] [--groups "F_OFF F_CONST F_REAL"] [--dry-run]
#
# PYTHON overrides the interpreter (tests pass a stub); it defaults to python3.
set -u

PYTHON=${PYTHON:-python3}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CHECKPOINT=${CHECKPOINT:-results/glt_galph_ph_retention_20260920/p1/pretrain_C1_REPAIR_5K/deploy_05000.pt}
IDENTITY=${IDENTITY:-configs/mts/glt_galph_c1_repair_5k_identity.json}
SIDECAR=${SIDECAR:-results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream}
CONST_PROFILE=${CONST_PROFILE:-${SIDECAR}/p_train_mean_profile.npy}

OUTPUT_ROOT=""; STATUS_LOG=""; CONFIG=""; PROTOCOL=""; TASKS="xc"; FOLDS="0"
GROUP_LIST="F_OFF F_CONST F_REAL"; DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --status-log) STATUS_LOG=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --protocol) PROTOCOL=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --folds) FOLDS=$2; shift 2 ;;
    --groups) GROUP_LIST=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for required in OUTPUT_ROOT STATUS_LOG CONFIG PROTOCOL; do
  eval "value=\${$required}"
  if [ -z "$value" ]; then echo "missing required option: $required" >&2; exit 2; fi
done
case "$PROTOCOL" in
  smoke|development) ;;
  *) echo "--protocol must be smoke or development" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$STATUS_LOG")"
cd "$REPO_ROOT" || exit 2

status() { printf '%s %s\n' "$(date -Iseconds)" "$1" >> "$STATUS_LOG"; }

for GROUP in $GROUP_LIST; do
  OUT="${OUTPUT_ROOT}/${GROUP}"
  LOG="${OUTPUT_ROOT}/${GROUP}.log"
  status "START group=${GROUP} protocol=${PROTOCOL} config=${CONFIG}"
  ARGS=(--config "$CONFIG" --checkpoint "$CHECKPOINT" --checkpoint-identity "$IDENTITY"
        --ph-sidecar "$SIDECAR" --const-profile "$CONST_PROFILE"
        --raw-root data/raw
        --cohort-root data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1
        --cache-root data/processed/mips_trimer_scage_downstream
        --dual-static-root data/processed/glt_dual_v2/downstream/dual_static_v1
        --group "$GROUP" --output "$OUT")
  case "$PROTOCOL" in
    smoke) ARGS+=(--smoke) ;;
    development) ARGS+=(--development) ;;
  esac
  for TASK in $TASKS; do ARGS+=(--task "$TASK"); done
  for FOLD in $FOLDS; do ARGS+=(--fold "$FOLD"); done
  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY_RUN %s %s\n' "$PYTHON" "${ARGS[*]}" >> "$LOG"
    code=$?
  else
    "$PYTHON" scripts/finetune_glt_galformer_ph_retention.py "${ARGS[@]}" > "$LOG" 2>&1
    code=$?
  fi
  status "EXIT group=${GROUP} code=${code} log=${LOG}"
  if [ "$code" -ne 0 ]; then
    status "STOPPED_AFTER_FAILURE group=${GROUP} code=${code}"
    echo "group ${GROUP} failed with exit ${code}; not starting any later group" >&2
    exit "$code"
  fi
done

status "ALL_DONE groups=${GROUP_LIST} protocol=${PROTOCOL}"
exit 0
