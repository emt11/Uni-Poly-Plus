#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd)"
PRETRAIN_GPU_IDS=${MTS_PRETRAIN_GPU_IDS:-1,2,3}
FINETUNE_GPU_IDS=${MTS_FINETUNE_GPU_IDS:-0,1,2,3}

validate_gpu_ids() {
  local value=$1 expected=$2 label=$3
  if [[ ! "$value" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "$label must be a comma-separated GPU list; got '$value'" >&2
    exit 2
  fi
  local -a ids=()
  IFS=',' read -r -a ids <<< "$value"
  if (( ${#ids[@]} != expected )); then
    echo "$label requires exactly $expected GPUs; got '$value'" >&2
    exit 2
  fi
  if (( $(printf '%s\n' "${ids[@]}" | sort -u | wc -l) != expected )); then
    echo "$label contains duplicate GPU ids: '$value'" >&2
    exit 2
  fi
}

validate_gpu_ids "$PRETRAIN_GPU_IDS" 3 MTS_PRETRAIN_GPU_IDS
validate_gpu_ids "$FINETUNE_GPU_IDS" 4 MTS_FINETUNE_GPU_IDS
if [[ "$PRETRAIN_GPU_IDS" != "1,2,3" ]]; then
  echo "MTS_PRETRAIN_GPU_IDS is fixed to physical GPUs 1,2,3; got '$PRETRAIN_GPU_IDS'" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$PRETRAIN_GPU_IDS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MIPS_DEBUG_ATTENTION=${MIPS_DEBUG_ATTENTION:-0}

PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/MTS/bin/python}
CONFIG=${EXPERIMENT_CONFIG:-configs/mts/default.json}
eval "$("$PYTHON_BIN" scripts/resolve_mips_trimer_scage.py "$CONFIG" --shell)"

read -r -a MSTA_LAYER_INDICES_VALUES < <(
  "$PYTHON_BIN" - "$MSTA_LAYER_INDICES" <<'PY'
import json, sys
print(*json.loads(sys.argv[1]))
PY
)
read -r -a MSTA_LOCAL_SPD_VALUES < <(
  "$PYTHON_BIN" - "$MSTA_LOCAL_SPD" <<'PY'
import json, sys
print(*json.loads(sys.argv[1]))
PY
)
read -r -a MSTA_CONTEXT_SPD_VALUES < <(
  "$PYTHON_BIN" - "$MSTA_CONTEXT_SPD" <<'PY'
import json, sys
print(*json.loads(sys.argv[1]))
PY
)
MSTA_DROPOUT_ARGS=(--msta_share_relation_dropout)
if [[ "$MSTA_SHARE_RELATION_DROPOUT" != "true" ]]; then
  MSTA_DROPOUT_ARGS=(--no-msta_share_relation_dropout)
fi
MSTA_LOCAL_BIAS_ARGS=(--no-msta_local_output_bias)
if [[ "$MSTA_LOCAL_OUTPUT_BIAS" == "true" ]]; then
  MSTA_LOCAL_BIAS_ARGS=(--msta_local_output_bias)
fi
PRETRAIN_G_FAMILY_ARGS=()
TRAIN_G_FAMILY_ARGS=()
if [[ -n "${G_FAMILY_ARM:-}" ]]; then
  PRETRAIN_G_FAMILY_ARGS=(--g_family_arm "$G_FAMILY_ARM" --g_family_bundle_hash "$G_FAMILY_BUNDLE_HASH")
  TRAIN_G_FAMILY_ARGS=(--g_family_arm "$G_FAMILY_ARM" --g_family_bundle_hash "$G_FAMILY_BUNDLE_HASH")
  if [[ "$G_FAMILY_ARM" != "g0" ]]; then
    PRETRAIN_G_FAMILY_ARGS+=(
      --relation_geometry_sidecar "$RELATION_GEOMETRY_SIDECAR_PI1M_V2"
      --relation_geometry_artifact_hash "$RELATION_GEOMETRY_ARTIFACT_PI1M_V2"
    )
    TRAIN_G_FAMILY_ARGS+=(
      --relation_geometry_sidecar "$RELATION_GEOMETRY_SIDECAR_DOWNSTREAM_UNION"
      --relation_geometry_artifact_hash "$RELATION_GEOMETRY_ARTIFACT_DOWNSTREAM_UNION"
    )
  fi
  if [[ "$G_FAMILY_ARM" == "g3" ]]; then
    PRETRAIN_G_FAMILY_ARGS+=(
      --g3_permutation_sidecar "$G3_PERMUTATION_SIDECAR_PI1M_V2"
      --g3_permutation_artifact_hash "$G3_PERMUTATION_ARTIFACT_PI1M_V2"
    )
    TRAIN_G_FAMILY_ARGS+=(
      --g3_permutation_sidecar "$G3_PERMUTATION_SIDECAR_DOWNSTREAM_UNION"
      --g3_permutation_artifact_hash "$G3_PERMUTATION_ARTIFACT_DOWNSTREAM_UNION"
    )
  fi
fi
MTS_T1_INIT_ARGS=()
if [[ "${MTS_ALLOW_T1_FUNCTION_PRESERVING_INIT:-0}" == "1" ]]; then
  if [[ "$TOPOLOGY_ATTENTION_VARIANT" != "msta_last2" ]]; then
    echo "MTS_ALLOW_T1_FUNCTION_PRESERVING_INIT=1 requires topology_attention_variant=msta_last2" >&2
    exit 2
  fi
  MTS_T1_INIT_ARGS=(--allow_mts_t1_function_preserving_init)
fi

# Causal geometry-ablation switches (Plan mts_geometry_injection A0-A4).  The
# finetune subprocess reads these env vars; absent values default to the full
# production recipe (A3 semantics).
export MTS_ABLATION_ID="${ABLATION_ID:-}"
export MTS_USE_STAR_RBF="${USE_STAR_RBF:-true}"
export MTS_USE_MCL="${USE_MCL:-true}"
export MTS_MCL_RANDOM_MASK="${MCL_RANDOM_MASK:-false}"
export MTS_RANDOM_MASK_SIDECAR="${RANDOM_MASK_SIDECAR:-}"
if [[ "${MTS_USE_STAR_RBF,,}" == "false" && "${MTS_USE_MCL,,}" == "false" ]]; then
  MTS_TRAIN_CACHE_LAYERS="ru_base,topology,md200"
else
  MTS_TRAIN_CACHE_LAYERS="ru_base,topology,trimer,md200"
fi

PRETRAIN_DATASET=${PRETRAIN_DATASET:-PI1M_v2}
SEED=${RANDOM_SEED:-42}
NPROC=3
CACHE_WORKERS=${CACHE_WORKERS:-48}
CACHE_VALIDATE=${CACHE_VALIDATE:-full}
PRETRAIN_LOADER_WORKERS=${PRETRAIN_DATALOADER_WORKERS:-${DATALOADER_WORKERS:-6}}
FINETUNE_LOADER_WORKERS=${FINETUNE_DATALOADER_WORKERS:-${DATALOADER_WORKERS:-2}}
LOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-2}
SAMPLER_VERSION=${MIPS_SAMPLER_VERSION:-cost_v1}
BATCH_BALANCE=${MIPS_BATCH_BALANCE:-cost}
REBUILD_FEATURE_CACHE=${REBUILD_FEATURE_CACHE:-0}
PRETRAIN_CACHE_ONLY=${PRETRAIN_CACHE_ONLY:-0}
MTS_EXPLICIT_VALIDATE_ONLY=${MTS_EXPLICIT_VALIDATE_ONLY:-0}
PRETRAIN_ONLY=${PRETRAIN_ONLY:-0}
PRETRAIN_BENCHMARK_ONLY=${PRETRAIN_BENCHMARK_ONLY:-0}
PRETRAIN_BENCHMARK_BATCHES=${PRETRAIN_BENCHMARK_BATCHES:-100}
PRETRAIN_BENCHMARK_STAGE=${PRETRAIN_BENCHMARK_STAGE:-joint}
PRETRAIN_PROFILE=${PRETRAIN_PROFILE:-canonical_ru_angle20_v1}
PRETRAIN_BENCHMARK_JSON=${PRETRAIN_BENCHMARK_JSON:-}
PRETRAIN_SMOKE_STEPS=${PRETRAIN_SMOKE_STEPS:-0}
PRETRAIN_CHECKPOINT_INTERVAL_STEPS=${PRETRAIN_CHECKPOINT_INTERVAL_STEPS:-0}
MTS_INITIALIZATION_STATE=${MTS_INITIALIZATION_STATE:-}
MTS_PAIRED_INIT_ID=${MTS_PAIRED_INIT_ID:-}
SHARED_STEP0_ID=${SHARED_STEP0_ID:-}
if [[ -n "${G_FAMILY_ARM:-}" ]]; then
  # G-family training has its own full-UniEncoder shared step-0 identity.  A
  # caller may still provide an explicit path, but an unset path must not
  # silently fall back to the historical T-Pretrain-0 pair.
  SHARED_STEP0_ID=${SHARED_STEP0_ID:-mts_g_family_step0_v2_seed42}
  MTS_PAIRED_INIT_ID=${MTS_PAIRED_INIT_ID:-$SHARED_STEP0_ID}
  MTS_INITIALIZATION_STATE=${MTS_INITIALIZATION_STATE:-pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/${G_FAMILY_ARM}_step0.pth}
fi
MTS_DIAGNOSTICS_DIR=${MTS_DIAGNOSTICS_DIR:-}
MTS_DIAGNOSTIC_STEPS=${MTS_DIAGNOSTIC_STEPS:-0,500,2000,5000,10000,20000}
# The accepted three-rank production profile keeps global batch 1008 with
# 336 samples/rank, one optimizer accumulation step, six loader workers/rank,
# and prefetch factor two.  Explicit environment overrides remain available.
PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-336}
PRETRAIN_ACCUMULATION=${PRETRAIN_ACCUMULATION:-1}
if [[ -n "$PRETRAIN_BENCHMARK_JSON" && "$PRETRAIN_BENCHMARK_ONLY" != 1 ]]; then
  read -r PRETRAIN_BATCH_SIZE PRETRAIN_ACCUMULATION PRETRAIN_LOADER_WORKERS LOADER_PREFETCH_FACTOR < <(
    "$PYTHON_BIN" - "$PRETRAIN_BENCHMARK_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
selected = payload.get("selected") or {}
if not selected:
    raise SystemExit("benchmark promotion gate did not select a production profile")
print(
    int(selected["batch_size_per_rank"]),
    int(selected["gradient_accumulation_steps"]),
    int(selected.get("loader_workers", 0)),
    int(selected.get("loader_prefetch_factor", 2)),
)
PY
  )
fi
if [[ $((PRETRAIN_BATCH_SIZE * NPROC * PRETRAIN_ACCUMULATION)) -ne 1008 ]]; then
  echo "MTS Joint Pretraining requires global batch 1008: batch=${PRETRAIN_BATCH_SIZE}, ranks=${NPROC}, accumulation=${PRETRAIN_ACCUMULATION}. Use (42,8), (84,4), (168,2), or (336,1)." >&2
  exit 2
fi
FINETUNE_ONLY=${FINETUNE_ONLY:-${STAGE3_ONLY:-0}}
RESUME=${RESUME:-0}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}
FINETUNE_EPOCHS=${MTS_FINETUNE_EPOCHS:-$TRAIN_EPOCHS}
FINETUNE_PATIENCE=${MTS_FINETUNE_PATIENCE:-10}
FINETUNE_BATCH_SIZE=${MTS_FINETUNE_BATCH_SIZE:-32}
FINETUNE_EVAL_BATCH_SIZE=${MTS_FINETUNE_EVAL_BATCH_SIZE:-64}
FINETUNE_AMP_DTYPE=${MTS_FINETUNE_AMP_DTYPE:-fp32}
FINETUNE_SEEDS=${FINETUNE_SEEDS:-42}
MTS_RUN_MULTI_SEED=${MTS_RUN_MULTI_SEED:-0}
G_FAMILY_FORMAL_PRETRAIN=0
if [[ ( "${G_FAMILY_ARM:-}" == "g0" || "${G_FAMILY_ARM:-}" == "g1" ) \
      && "$PRETRAIN_ONLY" == 1 \
      && "${PRETRAINING_OBJECTIVE:-}" == "masked_atom_only" \
      && ( "${ANGLE_LOSS_WEIGHT:-1}" == "0" || "${ANGLE_LOSS_WEIGHT:-1}" == "0.0" ) \
      && "${SHARED_STEP0_ID:-}" == "mts_g_family_step0_v2_seed42" ]]; then
  G_FAMILY_FORMAL_PRETRAIN=1
fi
if [[ "$RESOLVED_CONFIG_SCHEMA" == "mts-experiment-v3" \
      && "$FINETUNE_ONLY" != 1 \
      && "$PRETRAIN_CACHE_ONLY" != 1 \
      && "$PRETRAIN_BENCHMARK_ONLY" != 1 \
      && "$PRETRAIN_SMOKE_STEPS" -le 0 \
      && "${VALIDATE_ONLY:-0}" != 1 \
      && "$G_FAMILY_FORMAL_PRETRAIN" != 1 ]]; then
  echo "mts-experiment-v3 is fine-tune-only except for explicit benchmark/smoke validation; set FINETUNE_ONLY=1." >&2
  exit 2
fi
case "$MODALITIES" in
  '["graph"]') MTS_MODALITY_ARGS=(graph); MTS_FP_MODE=disabled ;;
  '["graph","smiles"]') MTS_MODALITY_ARGS=(graph smiles); MTS_FP_MODE=disabled ;;
  '["graph","fp"]') MTS_MODALITY_ARGS=(graph fp); MTS_FP_MODE=attachment_count ;;
  '["graph","smiles","fp"]') MTS_MODALITY_ARGS=(graph smiles fp); MTS_FP_MODE=attachment_count ;;
  *) echo "Unsupported MTS modalities payload: $MODALITIES" >&2; exit 2 ;;
esac
MTS_CONTROLLED_ARGS=()
if [[ -n "$CONTROLLED_MODALITY" && "$CONTROLLED_MODALITY" != "null" ]]; then
  MTS_CONTROLLED_ARGS=(--controlled_modality "$CONTROLLED_MODALITY")
fi
MTS_FINETUNE_MODE_ARGS=(--finetune_mode "$FINETUNE_MODE")
if [[ "$FINETUNE_MODE" == "multitask_pcgrad" ]]; then
  MTS_FINETUNE_MODE_ARGS+=(--cross_task_aux_weight 1.0)
fi
if [[ -n "${STAGE2_DATASET:-}" || -n "${STAGE2_STEPS:-}" || -n "${MTS_PRETRAIN_PHASE:-}" ]]; then
  echo "MTS topology/geometry split is retired; use the single MTS Joint Pretraining stage." >&2
  exit 2
fi
TASKS=${TASKS:-"eat eea egb egc ei eps nc xc"}
FOLD_IDS=${FOLD_IDS:-"0 1 2 3 4"}
read -r -a TASK_LIST <<< "$TASKS"
read -r -a FOLD_LIST <<< "$FOLD_IDS"
MTS_FINETUNE_SCHEDULE=${MTS_FINETUNE_SCHEDULE:-}
# lpt_v1 reorders only the dispatch sequence of the 40 independent (task, fold)
# finetune units, longest-predicted-task first per the historical
# total_fold_wall_seconds.  It changes no seed, identity hash, command-line
# argument, resume decision, or the four dynamic GPU slots; the schedule name
# is never written into a fold's training_config_hash.
LPT_V1_TASK_ORDER=(egc egb eat xc ei eps nc eea)
LPT_V1_FOLD_ORDER=(0 1 2 3 4)

if [[ "$PRETRAIN_DATASET" != "PI1M_v2" ]]; then
  echo "MTS pretraining accepts only PRETRAIN_DATASET=PI1M_v2." >&2
  exit 2
fi
JOINT_STEPS=20000
MTS_ANGLE_OBJECTIVE=${MTS_ANGLE_OBJECTIVE:-categorical}
MTS_ANGLE_WEIGHT=${ANGLE_LOSS_WEIGHT:-${MTS_ANGLE_WEIGHT:-0.25}}
if [[ "$PRETRAIN_PROFILE" == "canonical_ru_angle20_v1" || "$PRETRAIN_PROFILE" == *"canonical_ru_angle20_v1.json" ]]; then
  MTS_ANGLE_OBJECTIVE=categorical
  if [[ -z "${G_FAMILY_ARM:-}" ]]; then
    MTS_ANGLE_WEIGHT=0.25
  fi
fi
if [[ -n "${G_FAMILY_ARM:-}" ]]; then
  if [[ "${PRETRAINING_OBJECTIVE:-}" != "masked_atom_only" || ( "${ANGLE_LOSS_WEIGHT:-1}" != "0" && "${ANGLE_LOSS_WEIGHT:-1}" != "0.0" ) ]]; then
    echo "G-family requires pretraining_objective=masked_atom_only and angle_loss_weight=0." >&2
    exit 2
  fi
  MTS_ANGLE_WEIGHT=0
fi
FINETUNE_PROFILE=${FINETUNE_PROFILE:-legacy_mts_huber_v1}
if [[ "$FINETUNE_PROFILE" != "legacy_mts_huber_v1" ]]; then
  echo "MTS only supports finetune_profile=legacy_mts_huber_v1; F/Phase-A-B-C logic is retired." >&2
  exit 2
fi
if [[ "$FINETUNE_AMP_DTYPE" != "fp32" && "$FINETUNE_AMP_DTYPE" != "bf16" ]]; then
  echo "MTS_FINETUNE_AMP_DTYPE must be fp32 or bf16." >&2
  exit 2
fi
if (( FINETUNE_EVAL_BATCH_SIZE < FINETUNE_BATCH_SIZE )); then
  echo "MTS_FINETUNE_EVAL_BATCH_SIZE must be >= MTS_FINETUNE_BATCH_SIZE." >&2
  exit 2
fi
if [[ "$MTS_ANGLE_OBJECTIVE" != "categorical" && "$MTS_ANGLE_OBJECTIVE" != "cosine" ]]; then
  echo "MTS_ANGLE_OBJECTIVE must be categorical or cosine." >&2
  exit 2
fi
if [[ -n "${JOINT_MAX_OPT_STEPS:-}" && "${JOINT_MAX_OPT_STEPS}" != "20000" ]]; then
  echo "MTS Joint Pretraining is fixed to 20000 optimizer steps." >&2
  exit 2
fi
if [[ "$PRETRAIN_BENCHMARK_STAGE" != "joint" ]]; then
  echo "PRETRAIN_BENCHMARK_STAGE must be joint." >&2
  exit 2
fi
TIER=1m

hash_training_spec() {
  "$PYTHON_BIN" - "$@" <<'PY'
import hashlib
import json
import sys
payload = json.loads(sys.argv[1])
print(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
}

JOINT_TRAINING_HASH=$(hash_training_spec "$(cat <<JSON
{"config_hash":"$CONFIG_HASH","feature_hash":"$FEATURE_CONFIG_HASH","graph_hash":"$GRAPH_MODEL_CONFIG_HASH","geometry_hash":"$GEOMETRY_MODEL_CONFIG_HASH","stage":"mts_joint_pretraining","dataset":"PI1M_v2","steps":20000,"epoch_cap":30,"batch_size":$PRETRAIN_BATCH_SIZE,"accumulation":$PRETRAIN_ACCUMULATION,"lr":0.0002,"weight_decay":0.0,"optimizer_impl":"adam","adam_betas":[0.9,0.98],"eps":1e-8,"warmup_steps":2000,"scheduler":"polynomial","scheduler_power":1,"end_lr":1e-9,"amp":"bf16","mask_ratio":0.30,"angle_objective":"$MTS_ANGLE_OBJECTIVE","angle_weight":$MTS_ANGLE_WEIGHT,"angle_bins":20,"angle_gamma":2.0,"max_grad_norm":-1.0,"checkpoint_interval_steps":$PRETRAIN_CHECKPOINT_INTERVAL_STEPS,"seed":$SEED,"loader_workers":$PRETRAIN_LOADER_WORKERS,"loader_prefetch_factor":$LOADER_PREFETCH_FACTOR,"sampler_version":"$SAMPLER_VERSION","batch_balance":"$BATCH_BALANCE"}
JSON
)")
FINETUNE_PROFILE_HASH=$(hash_training_spec '{"profile":"legacy_mts_huber_v1","graph_wrapper_trainable_from_epoch":0,"loss":"huber","huber_beta":0.5,"epochs":100,"patience":10,"batch_size":32,"graph_wrapper_lr":1e-5,"smiles_lora_lr":5e-6,"adapter_lr":1e-4,"head_lr":1e-4,"weight_decay":0.02,"warmup_epochs":5,"scheduler":"cosine","swa_start_epoch":-1,"gradient_clip":1.0}')
FINETUNE_CONFIG_HASH=$(hash_training_spec "$(cat <<JSON
{"config_hash":"$CONFIG_HASH","feature_hash":"$FEATURE_CONFIG_HASH","graph_hash":"$GRAPH_MODEL_CONFIG_HASH","geometry_hash":"$GEOMETRY_MODEL_CONFIG_HASH","source_geometry_hash":"$SOURCE_GEOMETRY_MODEL_CONFIG_HASH","geometry_mode":"$GRAPH_GEOMETRY_MODE","modalities":$MODALITIES,"fusion_mode":"$FUSION_MODE","finetune_mode":"$FINETUNE_MODE","evaluation_protocol":"$EVALUATION_PROTOCOL","modality_control":"$MODALITY_CONTROL","controlled_modality":"$CONTROLLED_MODALITY","stage":"mts_property_finetune_legacy_v1","dataset":"downstream_union","epochs":100,"batch_size":32,"eval_batch_size":$FINETUNE_EVAL_BATCH_SIZE,"amp_dtype":"$FINETUNE_AMP_DTYPE","patience":10,"graph_wrapper_lr":0.00001,"smiles_lora_lr":0.000005,"adapter_lr":0.0001,"head_lr":0.0001,"weight_decay":0.02,"warmup_epochs":5,"scheduler":"cosine","finetune_profile":"$FINETUNE_PROFILE","finetune_profile_hash":"$FINETUNE_PROFILE_HASH","target_transform":"recommended","loss":"huber","huber_beta":0.5,"gradient_clip":1.0,"head_dropout":0.25,"swa_start_epoch":-1}
JSON
)")

stage3_training_hash() {
  local fine_seed=$1
  hash_training_spec "$(cat <<JSON
{"finetune_config_hash":"$FINETUNE_CONFIG_HASH","seed":$fine_seed,"loader_workers":$FINETUNE_LOADER_WORKERS}
JSON
)"
}

# Validate the resolved model plus every JSON-derived training identity.  This
# deliberately runs after hash construction so malformed shell/JSON quoting is
# caught before a cache or training process is started.
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
  [[ "$JOINT_TRAINING_HASH" =~ ^[0-9a-f]{64}$ ]]
  [[ "$FINETUNE_PROFILE_HASH" =~ ^[0-9a-f]{64}$ ]]
  [[ "$FINETUNE_CONFIG_HASH" =~ ^[0-9a-f]{64}$ ]]
  [[ "$(stage3_training_hash 42)" =~ ^[0-9a-f]{64}$ ]]
  "$PYTHON_BIN" scripts/pretrain.py --help >/dev/null
  "$PYTHON_BIN" scripts/train.py --help >/dev/null
  echo "MIPS-Trimer-SCAGE (MTS) configuration and CLI/hash validation passed."
  exit 0
fi

ARTIFACT_DIR=${ARTIFACT_DIR:-pretrained_models/mts}
LOG_DIR=${LOG_DIR:-logs/mts}
if [[ -z "${RESULTS_DIR:-}" ]]; then
  if [[ "$RESOLVED_CONFIG_SCHEMA" == "mts-experiment-v3" ]]; then
    RESULTS_DIR="results/mts_sota_v3/$EXPERIMENT_ID"
  else
    RESULTS_DIR="results/mts_finetune_v2"
  fi
fi
mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$RESULTS_DIR"
if [[ "$RESOLVED_CONFIG_SCHEMA" == "mts-experiment-v3" || -n "${CONFIG_SOURCE_SCHEMA:-}" ]]; then
  mkdir -p "$RESULTS_DIR/configs"
  cp -f "$CONFIG" "$RESULTS_DIR/configs/resolved_input.json"
fi
# The production default is the full dual-identity canonical checkpoint
# (Plan contract-finalization §3).  A caller may still override JOINT_CKPT
# explicitly, e.g. to load the historical 50b85b artifact for read-only reuse;
# the loader itself rejects any checkpoint without source/target contract.
JOINT_CKPT=${JOINT_CKPT:-"$ARTIFACT_DIR/mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"}
if [[ "$PRETRAIN_ONLY" == 1 && "$PRETRAIN_BENCHMARK_ONLY" != 1 ]]; then
  if [[ "$RESUME" == 1 ]]; then
    [[ -f "${JOINT_CKPT}.last.pt" ]] || { echo "resume requested but ${JOINT_CKPT}.last.pt is missing" >&2; exit 2; }
    [[ ! -f "$JOINT_CKPT" && ! -f "${JOINT_CKPT}.complete.json" ]] || { echo "resume refuses an already completed checkpoint: $JOINT_CKPT" >&2; exit 2; }
  else
    [[ ! -e "$JOINT_CKPT" && ! -e "${JOINT_CKPT}.last.pt" && ! -e "${JOINT_CKPT}.complete.json" ]] || {
      echo "fresh pretraining refuses to overwrite existing output: $JOINT_CKPT" >&2
      exit 2
    }
  fi
fi
JOINT_RESUME_ARGS=()
JOINT_TEE_ARGS=()
if [[ "$RESUME" == 1 && -f "${JOINT_CKPT}.last.pt" ]]; then
  JOINT_RESUME_ARGS=(--resume_state "${JOINT_CKPT}.last.pt")
  JOINT_TEE_ARGS=(-a)
fi

COMMON=(
  --seed "$SEED"
  --config_schema "$RESOLVED_CONFIG_SCHEMA"
  --config_source_schema "${CONFIG_SOURCE_SCHEMA:-}"
  --experiment_id "$EXPERIMENT_ID"
  --feature_config_hash "$FEATURE_CONFIG_HASH"
  --o8_feature_config_hash "$FEATURE_CONFIG_HASH"
  --model_config_hash "$GRAPH_MODEL_CONFIG_HASH"
  --graph_model_config_hash "$GRAPH_MODEL_CONFIG_HASH"
  --geometry_model_config_hash "$GEOMETRY_MODEL_CONFIG_HASH"
  --source_geometry_model_config_hash "$SOURCE_GEOMETRY_MODEL_CONFIG_HASH"
  --alignment_model_config_hash none
    --training_config_hash "$CONFIG_HASH"
  --graph_encoder_type mips_trimer_scage
  --topology_representation "$TOPOLOGY_REPRESENTATION"
  --graph_input star_linking
  --modalities "${MTS_MODALITY_ARGS[@]}"
  --fp_mode "$MTS_FP_MODE"
  --fusion_type "$FUSION_MODE"
  --mips_fusion_mode none
  --projection_mode plain
  --modality_control "$MODALITY_CONTROL"
  "${MTS_CONTROLLED_ARGS[@]}"
  --smiles_modality_dropout 0.10
  --fp_modality_dropout 0.15
  --graph_modality_dropout 0.0
  --mts_num_layers 6
  --mts_hidden_dim 512
  --mts_num_heads 8
  --mips_core paper_corrected
  --mips_variant O8
  --mips_max_hops 2
  --mips_atom_feature_mode mips137
  --mips_attention_scale head_dim
  --mips_norm_mode post
  --mips_activation relu
  --mips_spd_bias_mode per_head
  --mips_path_bias_mode per_head_single_path_node
  --mips_descriptor_components md200
  --mips_descriptor_protocol source_star_sub
  --mips_descriptor_fusion_mode graph_md_residual
  --mips_descriptor_disturbance 0
  --mips_mask_policy canonical_exact
  --mips_use_descriptors
  --topology_attention_variant "$TOPOLOGY_ATTENTION_VARIANT"
  --msta_layer_indices "${MSTA_LAYER_INDICES_VALUES[@]}"
  --msta_local_spd "${MSTA_LOCAL_SPD_VALUES[@]}"
  --msta_context_spd "${MSTA_CONTEXT_SPD_VALUES[@]}"
  "${MSTA_DROPOUT_ARGS[@]}"
  "${MSTA_LOCAL_BIAS_ARGS[@]}"
  --msta_local_output_init "$MSTA_LOCAL_OUTPUT_INIT"
  --spatial_mode trimer_scage
  --graph_geometry_mode "$GRAPH_GEOMETRY_MODE"
  --mcl_distance_percentiles 0.20 0.50
  --trimer_num_candidates 4
  --trimer_max_heavy_atoms 384
)
CACHE=(
  --feature_cache_workers "$CACHE_WORKERS"
  --feature_cache_chunksize 2
  --feature_cache_partial_every 5000
  --feature_cache_item_timeout 240
  --cache_validate "$CACHE_VALIDATE"
)
REBUILD=()
if [[ "$REBUILD_FEATURE_CACHE" == 1 ]]; then
  REBUILD=(--rebuild_feature_cache)
fi
BENCHMARK_ARGS=()
if [[ "$PRETRAIN_BENCHMARK_ONLY" == 1 ]]; then
  BENCHMARK_ARGS=(--benchmark_only --benchmark_batches "$PRETRAIN_BENCHMARK_BATCHES")
fi
SMOKE_ARGS=()
if [[ "$PRETRAIN_SMOKE_STEPS" -gt 0 ]]; then
  SMOKE_ARGS=(--resume_smoke --max_optimizer_steps "$PRETRAIN_SMOKE_STEPS")
fi
INITIALIZATION_ARGS=()
if [[ -n "$MTS_INITIALIZATION_STATE" ]]; then
  INITIALIZATION_ARGS=(
    --initialization_state "$MTS_INITIALIZATION_STATE"
    --paired_init_id "${MTS_PAIRED_INIT_ID:-mts_t_pretrain0_matched_v1}"
  )
fi
DIAGNOSTIC_ARGS=()
if [[ -n "$MTS_DIAGNOSTICS_DIR" ]]; then
  DIAGNOSTIC_ARGS=(
    --diagnostics_dir "$MTS_DIAGNOSTICS_DIR"
    --diagnostic_steps "$MTS_DIAGNOSTIC_STEPS"
  )
fi
PRETRAIN_IDENTITY_ARGS=(
  --pretraining_objective "$PRETRAINING_OBJECTIVE"
  --angle_loss_weight "$ANGLE_LOSS_WEIGHT"
)
if [[ -n "${SHARED_STEP0_ID:-}" ]]; then
  PRETRAIN_IDENTITY_ARGS+=(--shared_step0_id "$SHARED_STEP0_ID")
fi
TRAIN_IDENTITY_ARGS=()
if [[ -n "${SHARED_STEP0_ID:-}" ]]; then
  TRAIN_IDENTITY_ARGS+=(--shared_step0_id "$SHARED_STEP0_ID")
fi

run_cache() {
  local dataset=$1
  local layers=$2
  shift 2
  # A finalized immutable layer needs no Dataset/cache-only pass.  The old
  # dispatcher reopened and iterated all PI1M_v2 records twice before every
  # benchmark or training launch, even though the bundle gate below had
  # already verified the same .done/.frozen markers.  Only fall back to the
  # builder when a layer is not frozen or an explicit rebuild was requested.
  if [[ "$REBUILD_FEATURE_CACHE" != 1 \
        && "$TOPOLOGY_REPRESENTATION" == "canonical_lifted" ]]; then
    if "$PYTHON_BIN" - "$layers" <<'PY'
import sys
from pathlib import Path
from scripts.audit_mips_trimer_cache import _specs

layer = sys.argv[1]
spec = _specs(Path.cwd()).get(layer)
if spec is None:
    raise SystemExit(1)
root = Path(spec["root"])
raise SystemExit(0 if (root / ".done").is_file() and (root / ".frozen").is_file() else 1)
PY
    then
      echo "[cache] $dataset/$layers is frozen; skip cache-only rebuild scan."
      return 0
    fi
  fi
  if ps -eo args= | grep -F "scripts/pretrain.py" | grep -F -- "--cache_layers trimer" | grep -v grep >/dev/null; then
    echo "A Trimer cache writer is already active; refusing to start a second writer." >&2
    exit 3
  fi
  env CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/pretrain.py \
    --dataset_name "$dataset" --pretrain_stage mts_joint_pretraining \
    "${COMMON[@]}" "${CACHE[@]}" "${PRETRAIN_G_FAMILY_ARGS[@]}" \
    "${PRETRAIN_IDENTITY_ARGS[@]}" "$@" \
    --cache_layers "$layers" --cache_only
}

if [[ "$FINETUNE_ONLY" != 1 ]]; then
  if [[ "$PRETRAIN_ONLY" != 1 && "$PRETRAIN_BENCHMARK_ONLY" != 1 ]]; then
    if [[ "$TOPOLOGY_REPRESENTATION" == "explicit_k_ru" \
          && "$MTS_EXPLICIT_VALIDATE_ONLY" == 1 ]]; then
      # Read-only joint validation: dependency closure opens RU, explicit
      # Topology and the frozen canonical Trimer, with no missing writer jobs.
      "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from src.dataset.dataset import UniDataset
from src.dataset.mips_trimer_contract import TOPOLOGY_EXPLICIT

d = UniDataset.__new__(UniDataset)
d.root = str(Path.cwd() / "data")
d.cache_layers = ("ru_base", "topology", "trimer")
d.topology_representation = TOPOLOGY_EXPLICIT
specs = d._lmdb_cache_specs({})
for layer in ("ru_base", "topology", "trimer"):
    root = Path(specs[layer]["root"])
    if not (root / ".done").is_file():
        raise SystemExit(f"read-only explicit validation requires completed {layer}: {root}")
if not (Path(specs["trimer"]["root"]) / ".frozen").is_file():
    raise SystemExit("read-only explicit validation requires frozen canonical Trimer")
PY
      run_cache "$PRETRAIN_DATASET" "trimer"
    else
      run_cache "$PRETRAIN_DATASET" "topology" "${REBUILD[@]}"
    fi
    # Explicit k-RU changes only Topology.  Its finite Trimer is the existing
    # frozen canonical-identity layer, so scanning PI1M_v2 a second time via a
    # no-op Trimer cache pass is both unnecessary and misleading.
    if [[ "$TOPOLOGY_REPRESENTATION" == "canonical_lifted" ]]; then
      run_cache "$PRETRAIN_DATASET" "trimer"
    fi
    if [[ "$TOPOLOGY_REPRESENTATION" == "canonical_lifted" ]]; then
      env CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/prepare_mts_angle_cache.py
      if [[ "$MTS_ANGLE_OBJECTIVE" == "cosine" ]]; then
        env CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/build_mts_angle_v2.py
      fi
    fi
  fi
fi

# Freeze-time cache preparation is deliberately completed before any GPU
# pretraining.  The feature-source cohort is the downstream union only; it
# is never used as a Stage 1/1.5 sampling cohort.
if [[ "$FINETUNE_ONLY" != 1 && "$PRETRAIN_ONLY" != 1 && "$PRETRAIN_BENCHMARK_ONLY" != 1 || "$PRETRAIN_CACHE_ONLY" == 1 ]]; then
  env CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/train.py \
    "${COMMON[@]}" "${CACHE[@]}" "${TRAIN_G_FAMILY_ARGS[@]}" \
    --feature_source_dataset smi_all \
    --cache_layers "$MTS_TRAIN_CACHE_LAYERS" \
    --tasks "${TASK_LIST[@]}" --fold_ids "${FOLD_LIST[@]}" --cache_only
fi
if [[ "$PRETRAIN_CACHE_ONLY" == 1 ]]; then
  exit 0
fi

if [[ "$FINETUNE_ONLY" != 1 ]]; then
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from scripts.audit_mips_trimer_cache import _specs
from src.dataset.mips_cache_validation import verify_frozen_cache_bundle
specs = _specs(Path.cwd())
try:
    verify_frozen_cache_bundle(
        specs,
        store_path=Path(specs["topology"]["root"]).parents[1]
        / "validation" / "store.json",
        required_layers=specs.keys(),
    )
except Exception as exc:
    raise SystemExit(
        "cache bundle verification failed; run finalize_mips_trimer_cache.py: "
        + str(exc)
    )
PY
fi

# A corrected explicit run is meaningful only after its independent exact-
# union Topology artifact has passed joint validation against the frozen
# canonical Trimer.  Cache-only materialisation is exempt because it creates
# that artifact; benchmark/smoke must never train from a merely partial LMDB.
if [[ "$TOPOLOGY_REPRESENTATION" == "explicit_k_ru" \
      && "$PRETRAIN_CACHE_ONLY" != 1 ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from src.dataset.dataset import UniDataset
from src.dataset.mips_trimer_contract import TOPOLOGY_EXPLICIT

d = UniDataset.__new__(UniDataset)
d.root = str(Path.cwd() / "data")
d.cache_layers = ("ru_base", "topology", "trimer")
d.topology_representation = TOPOLOGY_EXPLICIT
root = Path(d._lmdb_cache_specs({})["topology"]["root"])
acceptance = Path("results/mts_explicit_k_ru/final_acceptance.json")
if not (root / ".done").is_file() or not (root / ".frozen").is_file():
    raise SystemExit("explicit_k_ru benchmark/smoke requires a frozen exact-union Topology cache")
if not acceptance.is_file() or not bool(json.loads(acceptance.read_text()).get("passed")):
    raise SystemExit("explicit_k_ru benchmark/smoke requires passed final_acceptance.json")
PY
fi

LAUNCH=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone
  --nproc_per_node "$NPROC"
)
if [[ "$FINETUNE_ONLY" != 1 ]]; then
  "${LAUNCH[@]}" scripts/pretrain.py \
    --dataset_name "$PRETRAIN_DATASET" \
    --feature_source_dataset "$PRETRAIN_DATASET" \
    --pretrain_stage mts_joint_pretraining \
    "${COMMON[@]}" "${CACHE[@]}" "${PRETRAIN_G_FAMILY_ARGS[@]}" \
    "${PRETRAIN_IDENTITY_ARGS[@]}" \
    --cache_layers topology,trimer \
    --pretrain_profile "$PRETRAIN_PROFILE" \
    --training_config_hash "$JOINT_TRAINING_HASH" \
    --epochs 30 --max_optimizer_steps "$JOINT_STEPS" \
    --batch_size "$PRETRAIN_BATCH_SIZE" --gradient_accumulation_steps "$PRETRAIN_ACCUMULATION" \
    --loader_workers "$PRETRAIN_LOADER_WORKERS" --loader_prefetch_factor "$LOADER_PREFETCH_FACTOR" --batch_balance "$BATCH_BALANCE" --amp_dtype bf16 \
    --lr 2e-4 --warmup_steps 2000 --mips_scheduler polynomial --scheduler_power 1 --end_lr 1e-9 \
    --graph_mask_ratio 0.30 --angle_objective "$MTS_ANGLE_OBJECTIVE" \
    --graph_angle_weight "$MTS_ANGLE_WEIGHT" --mips_spd_weight 0 --mips_path_bond_weight 0 \
    --no-dynamic_pretrain_loss --max_grad_norm -1 \
    --checkpoint_interval_steps "$PRETRAIN_CHECKPOINT_INTERVAL_STEPS" \
    "${INITIALIZATION_ARGS[@]}" \
    "${DIAGNOSTIC_ARGS[@]}" \
    "${SMOKE_ARGS[@]}" \
    "${BENCHMARK_ARGS[@]}" \
    "${JOINT_RESUME_ARGS[@]}" \
    --save_path "$JOINT_CKPT" \
    2>&1 | tee "${JOINT_TEE_ARGS[@]}" "$LOG_DIR/mts_joint_pretraining_pi1m_v2.log"
fi
if [[ "$PRETRAIN_BENCHMARK_ONLY" == 1 ]]; then
  exit 0
fi
if [[ "$PRETRAIN_ONLY" == 1 ]]; then
  exit 0
fi
if [[ ! -f "$JOINT_CKPT" ]]; then
  echo "Missing MTS Joint Pretraining checkpoint: $JOINT_CKPT" >&2
  exit 2
fi

# MTS fine-tuning uses independent (seed, task, fold) shards.  Seed 42 is
# always completed and evaluated first; seeds 43/44 are admitted only after
# the configured promotion gate passes.
if [[ "$MTS_RUN_MULTI_SEED" == 1 ]]; then
  FINETUNE_SEEDS="42 43 44"
fi
read -r -a FINETUNE_SEED_LIST <<< "$FINETUNE_SEEDS"
if [[ " ${FINETUNE_SEED_LIST[*]} " != *" 42 "* ]]; then
  echo "MTS fine-tuning must include seed 42 as the promotion screen." >&2
  exit 2
fi

pids=()
declare -A PID_UNIT=()
declare -A PID_GPU=()
cleanup_stage3_children() {
  local status=$?
  trap - INT TERM EXIT
  for pid in "${pids[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      # The slot child is a subshell wrapping the train process; kill the
      # train process (direct child) first so no orphan survives the cleanup
      # (Plan §7 failure-reaping test).
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup_stage3_children INT TERM EXIT
mkdir -p "$RESULTS_DIR/shards" "$RESULTS_DIR/predictions" "$LOG_DIR"
STORE_PATH=$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
from scripts.audit_mips_trimer_cache import _specs
print(Path(_specs(Path.cwd())["topology"]["root"]).parents[1].joinpath(
    "validation", "store.json"
))
PY
)
CHECKPOINT_SHA256=$(sha256sum "$JOINT_CKPT" | awk '{print $1}')
CACHE_STORE_SHA256=$(sha256sum "$STORE_PATH" | awk '{print $1}')
TOPOLOGY_ARTIFACT_HASH=$(cat "$($PYTHON_BIN - <<'PY'
from pathlib import Path
from scripts.audit_mips_trimer_cache import _specs
print(Path(_specs(Path.cwd())["topology"]["root"]).joinpath(".done"))
PY
)")
TRIMER_ARTIFACT_HASH=$(cat "$($PYTHON_BIN - <<'PY'
from pathlib import Path
from scripts.audit_mips_trimer_cache import _specs
print(Path(_specs(Path.cwd())["trimer"]["root"]).joinpath(".done"))
PY
)")

validate_shard() {
  local shard=$1 prediction=$2 task=$3 fold=$4 fine_seed=$5 training_hash=$6
  if [[ ! -s "$shard" ]]; then
    return 1
  fi
  "$PYTHON_BIN" - "$shard" "$prediction" "$task" "$fold" "$fine_seed" "$CONFIG_HASH" \
  "$FEATURE_CONFIG_HASH" "$GRAPH_MODEL_CONFIG_HASH" \
    "$GEOMETRY_MODEL_CONFIG_HASH" "$SOURCE_GEOMETRY_MODEL_CONFIG_HASH" "$training_hash" \
    "$FINETUNE_CONFIG_HASH" "$FINETUNE_PROFILE_HASH" \
    "$CHECKPOINT_SHA256" "$CACHE_STORE_SHA256" "$TOPOLOGY_ARTIFACT_HASH" \
    "$TRIMER_ARTIFACT_HASH" <<'PY'
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
path, prediction, task = sys.argv[1:4]
fold, seed = int(sys.argv[4]), int(sys.argv[5])
config_hash, feature_hash, graph_hash, geometry_hash = sys.argv[6:10]
source_geometry_hash = sys.argv[10]
training_hash, finetune_hash, profile_hash = sys.argv[11:14]
checkpoint_sha, store_sha = sys.argv[14:16]
topology_artifact, trimer_artifact = sys.argv[16:18]
try:
    frame = pd.read_csv(path)
    if len(frame) != 1 or str(frame.iloc[0].get("task", "")) != task:
        raise ValueError
    row = frame.iloc[0]
    if str(row.get("config_hash", "")) != config_hash or str(row.get("resolved_config_hash", "")) != config_hash:
        raise ValueError("resolved config hash mismatch")
    if str(row.get("feature_config_hash", "")) != feature_hash:
        raise ValueError("feature config hash mismatch")
    if str(row.get("graph_model_config_hash", "")) != graph_hash:
        raise ValueError("graph model hash mismatch")
    if str(row.get("geometry_model_config_hash", "")) != geometry_hash:
        raise ValueError("geometry model hash mismatch")
    if str(row.get("source_geometry_model_config_hash", "")) != source_geometry_hash:
        raise ValueError("source geometry model hash mismatch")
    if str(row.get("training_config_hash", "")) != training_hash:
        raise ValueError("training config hash mismatch")
    if str(row.get("finetune_config_hash", "")) != finetune_hash:
        raise ValueError("finetune config hash mismatch")
    if str(row.get("finetune_profile_hash", "")) != profile_hash:
        raise ValueError("finetune profile hash mismatch")
    if int(row.get("seed", -1)) != seed:
        raise ValueError("seed mismatch")
    if str(row.get("checkpoint_sha256", "")) != checkpoint_sha:
        raise ValueError("checkpoint artifact hash mismatch")
    if str(row.get("cache_store_sha256", "")) != store_sha:
        raise ValueError("cache store hash mismatch")
    if str(row.get("topology_cache_artifact_hash", "")) != topology_artifact:
        raise ValueError("topology artifact hash mismatch")
    if str(row.get("trimer_cache_artifact_hash", "")) != trimer_artifact:
        raise ValueError("Trimer artifact hash mismatch")
    split_path = Path("data/splits/mips_shared5") / f"{task}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    if str(row.get("split_manifest_hash", "")) != split_hash:
        raise ValueError("split manifest hash mismatch")
    metrics = json.loads(str(frame.iloc[0]["per_fold_metrics"]))
    if len(metrics) != 1 or int(metrics[0].get("fold", -1)) != fold:
        raise ValueError
    pred_path = Path(prediction)
    if not pred_path.is_file():
        raise ValueError("missing prediction shard")
    digest = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if str(row.get("prediction_sha256", "")) != digest:
        raise ValueError("prediction hash mismatch")
    with np.load(pred_path, allow_pickle=False) as payload:
        meta = json.loads(str(np.asarray(payload["metadata"]).item()))
        if len(payload["y_true"]) != len(payload["y_pred"]):
            raise ValueError("prediction shape mismatch")
    if any((meta.get("task") != task, int(meta.get("fold", -1)) != fold,
            int(meta.get("seed", -1)) != seed,
            meta.get("finetune_config_hash") != finetune_hash,
            meta.get("finetune_profile_hash") != profile_hash,
            meta.get("checkpoint_sha256") != checkpoint_sha,
            meta.get("cache_store_sha256") != store_sha,
            meta.get("split_manifest_hash") != split_hash)):
        raise ValueError("prediction metadata mismatch")
except Exception:
    raise SystemExit(1)
PY
}

launch_stage3_unit() {
  local unit=$1 gpu=$2 fine_seed task fold shard prediction training_hash
  IFS='|' read -r fine_seed task fold <<< "$unit"
  shard="$RESULTS_DIR/shards/$fine_seed/$task/fold_${fold}.csv"
  prediction="$RESULTS_DIR/predictions/$fine_seed/$task/fold_${fold}.npz"
  training_hash=$(stage3_training_hash "$fine_seed")
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    if [[ "${MTS_FAKE_TRAIN:-0}" == "1" ]]; then
      # Test-only injection (Plan §7): a fake training command drives the real
      # four-slot dispatcher without GPU/Dataset.  It receives the unit and
      # output paths; the real train.py path is untouched when unset.
      "$PYTHON_BIN" "${MTS_FAKE_TRAIN_CMD}" \
        --task "$task" --fold "$fold" \
        --shard "$shard" --prediction "$prediction" \
        >> "$LOG_DIR/finetune_seed${fine_seed}_${task}_fold${fold}.log" 2>&1
    else
      "$PYTHON_BIN" scripts/train.py \
        "${COMMON[@]}" "${CACHE[@]}" "${TRAIN_G_FAMILY_ARGS[@]}" \
        "${TRAIN_IDENTITY_ARGS[@]}" \
        "${MTS_T1_INIT_ARGS[@]}" \
        "${MTS_FINETUNE_MODE_ARGS[@]}" \
        --cache_layers "$MTS_TRAIN_CACHE_LAYERS" \
        --tasks "$task" --fold_ids "$fold" \
        --pretrained_model_path "$JOINT_CKPT" \
        --checkpoint_seed "$SEED" --seed "$fine_seed" \
        --resolved_config_hash "$CONFIG_HASH" \
        --checkpoint_sha256 "$CHECKPOINT_SHA256" \
        --cache_store_sha256 "$CACHE_STORE_SHA256" \
        --training_config_hash "$training_hash" \
        --finetune_config_hash "$FINETUNE_CONFIG_HASH" \
        --finetune_profile "$FINETUNE_PROFILE" \
        --finetune_profile_hash "$FINETUNE_PROFILE_HASH" \
        --predictions_dir "$RESULTS_DIR/predictions/$fine_seed" \
        --checkpoint_pretraining_dataset "$PRETRAIN_DATASET" \
        --checkpoint_tier 1m \
        --epochs "$FINETUNE_EPOCHS" --patience "$FINETUNE_PATIENCE" --batch_size "$FINETUNE_BATCH_SIZE" \
        --eval_batch_size "$FINETUNE_EVAL_BATCH_SIZE" --amp_dtype "$FINETUNE_AMP_DTYPE" \
        --loader_workers "$FINETUNE_LOADER_WORKERS" \
        --evaluation_protocol "$EVALUATION_PROTOCOL" \
        --target_transform recommended \
        --regression_loss huber --huber_beta 0.5 --max_grad_norm 1.0 \
        --graph_lr 1e-5 --fusion_lr 1e-4 --head_lr 1e-4 --weight_decay 0.02 \
        --warmup_epochs 5 --head_dropout 0.25 \
        --results_dir "$shard" \
        >> "$LOG_DIR/finetune_seed${fine_seed}_${task}_fold${fold}.log" 2>&1
    fi
  ) &
  local pid=$!
  pids+=("$pid")
  PID_UNIT["$pid"]="$unit"
  PID_GPU["$pid"]="$gpu"
}

run_finetune_seeds() {
  local -a requested=("$@") queue=() free_gpus=()
  IFS=',' read -r -a free_gpus <<< "$FINETUNE_GPU_IDS"
  local fine_seed task fold shard prediction training_hash
  local -a dispatch_tasks=("${TASK_LIST[@]}") dispatch_folds=("${FOLD_LIST[@]}")
  if [[ "$MTS_FINETUNE_SCHEDULE" == "lpt_v1" ]]; then
    for task in "${TASK_LIST[@]}"; do
      if ! printf '%s\n' "${LPT_V1_TASK_ORDER[@]}" | grep -qx "$task"; then
        echo "[finetune] lpt_v1 schedule cannot order unknown task '$task'" >&2
        exit 2
      fi
    done
    for fold in "${FOLD_LIST[@]}"; do
      if ! printf '%s\n' "${LPT_V1_FOLD_ORDER[@]}" | grep -qx "$fold"; then
        echo "[finetune] lpt_v1 fold order is 0 1 2 3 4; got '$fold'" >&2
        exit 2
      fi
    done
    # Reorder only the requested tasks by the fixed LPT sequence (longest task
    # first); the fold order stays as requested (0..4 for the full 8x5 run).
    # A partial smoke run therefore dispatches exactly its requested subset.
    dispatch_tasks=()
    for task in "${LPT_V1_TASK_ORDER[@]}"; do
      if printf '%s\n' "${TASK_LIST[@]}" | grep -qx "$task"; then
        dispatch_tasks+=("$task")
      fi
    done
    dispatch_folds=("${FOLD_LIST[@]}")
  else
    dispatch_tasks=("${TASK_LIST[@]}")
    dispatch_folds=("${FOLD_LIST[@]}")
  fi
  for fine_seed in "${requested[@]}"; do
    training_hash=$(stage3_training_hash "$fine_seed")
    for task in "${dispatch_tasks[@]}"; do
      mkdir -p "$RESULTS_DIR/shards/$fine_seed/$task" "$RESULTS_DIR/predictions/$fine_seed/$task"
      for fold in "${dispatch_folds[@]}"; do
        shard="$RESULTS_DIR/shards/$fine_seed/$task/fold_${fold}.csv"
        prediction="$RESULTS_DIR/predictions/$fine_seed/$task/fold_${fold}.npz"
        if validate_shard "$shard" "$prediction" "$task" "$fold" "$fine_seed" "$training_hash"; then
          echo "[finetune] resume: verified seed=$fine_seed task=$task fold=$fold"
          continue
        fi
        for invalid in "$shard" "$prediction"; do
          if [[ -e "$invalid" ]]; then
            mv "$invalid" "${invalid}.invalid.$(date +%s)"
          fi
        done
        queue+=("$fine_seed|$task|$fold")
      done
    done
  done

  local next_unit=0 gpu finished_pid status finished_unit finished_gpu
  while (( next_unit < ${#queue[@]} || ${#pids[@]} > 0 )); do
    while (( next_unit < ${#queue[@]} && ${#free_gpus[@]} > 0 )); do
      gpu="${free_gpus[0]}"
      free_gpus=("${free_gpus[@]:1}")
      launch_stage3_unit "${queue[$next_unit]}" "$gpu"
      next_unit=$((next_unit + 1))
    done
    (( ${#pids[@]} > 0 )) || continue
    set +e
    wait -n -p finished_pid
    status=$?
    set -e
    finished_unit="${PID_UNIT[$finished_pid]:-unknown}"
    finished_gpu="${PID_GPU[$finished_pid]:-}"
    unset "PID_UNIT[$finished_pid]" "PID_GPU[$finished_pid]"
    remaining=()
    for pid in "${pids[@]}"; do
      [[ "$pid" != "$finished_pid" ]] && remaining+=("$pid")
    done
    pids=("${remaining[@]}")
    [[ -n "$finished_gpu" ]] && free_gpus+=("$finished_gpu")
    if (( status != 0 )); then
      echo "[finetune] failed unit=${finished_unit} status=${status}; stopping all slots" >&2
      exit "$status"
    fi
    echo "[finetune] completed unit=${finished_unit}; queued=$(( ${#queue[@]} - next_unit )) active=${#pids[@]}"
  done
}

run_finetune_seeds 42
if (( ${#TASK_LIST[@]} != 8 || ${#FOLD_LIST[@]} != 5 )); then
  echo "[finetune] partial task/fold run completed; formal summary requires 8 tasks x 5 folds."
  unset PID_UNIT
  trap - INT TERM EXIT
  exit 0
fi
GATE_BASELINE_ARGS=()
if [[ -f results/mts/mts_summary.csv ]]; then
  GATE_BASELINE_ARGS=(
    --gate-baseline-csv results/mts/mts_summary.csv
    --gate-output "$RESULTS_DIR/seed42_promotion.json"
  )
else
  echo "[finetune] historical baseline results/mts/mts_summary.csv missing; seed-42 promotion gate skipped (single-seed run unaffected)." >&2
fi
"$PYTHON_BIN" scripts/summarize_mips_trimer_scage.py \
  --results-root "$RESULTS_DIR" \
  --output-csv "$RESULTS_DIR/mts_seed42_summary.csv" \
  --output-md "$RESULTS_DIR/mts_seed42_summary.md" \
  --tasks "${TASK_LIST[@]}" --folds "${FOLD_LIST[@]}" --seeds 42 \
  --evaluation-protocol "$EVALUATION_PROTOCOL" \
  "${GATE_BASELINE_ARGS[@]}"

remaining_seeds=()
for fine_seed in "${FINETUNE_SEED_LIST[@]}"; do
  [[ "$fine_seed" != 42 ]] && remaining_seeds+=("$fine_seed")
done
if (( ${#remaining_seeds[@]} > 0 )); then
  if ! "$PYTHON_BIN" - "$RESULTS_DIR/seed42_promotion.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 1)
PY
  then
    echo "Seed-42 promotion gate failed; seeds 43/44 will not run." >&2
    exit 4
  fi
  run_finetune_seeds "${remaining_seeds[@]}"
fi
unset PID_UNIT
trap - INT TERM EXIT
if [[ "$RESOLVED_CONFIG_SCHEMA" == "mts-experiment-v3" ]]; then
  if [[ "$EVALUATION_PROTOCOL" == "nested5" ]]; then
    FORMAL_SUMMARY_CSV="$RESULTS_DIR/nested_summary.csv"
    FORMAL_SUMMARY_MD="$RESULTS_DIR/final_report.md"
  else
    FORMAL_SUMMARY_CSV="$RESULTS_DIR/historical_summary.csv"
    FORMAL_SUMMARY_MD="$RESULTS_DIR/final_report.md"
  fi
else
  FORMAL_SUMMARY_CSV="$RESULTS_DIR/mts_finetune_summary.csv"
  FORMAL_SUMMARY_MD="$RESULTS_DIR/mts_finetune_summary.md"
fi
"$PYTHON_BIN" scripts/summarize_mips_trimer_scage.py \
  --results-root "$RESULTS_DIR" \
  --output-csv "$FORMAL_SUMMARY_CSV" \
  --output-md "$FORMAL_SUMMARY_MD" \
  --tasks "${TASK_LIST[@]}" --folds "${FOLD_LIST[@]}" \
  --evaluation-protocol "$EVALUATION_PROTOCOL" \
  --seeds "${FINETUNE_SEED_LIST[@]}"
