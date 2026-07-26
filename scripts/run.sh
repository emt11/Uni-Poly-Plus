#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PYTHON_BIN=${PYTHON_BIN:-/root/anaconda3/envs/Uni-Poly/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python
fi

# Only expose experiment-level switches here. Architecture defaults live in
# train.py / pretrain.py so this entry point stays readable.
GRAPH_GEOM_EPOCHS=${GRAPH_GEOM_EPOCHS:-}
ALIGN_EPOCHS=${ALIGN_EPOCHS:-10}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}
BASELINE=${BASELINE:-scage_parallel}           # flat4 | scage_parallel
FP_MODE=${FP_MODE:-ecfp}                       # ecfp | mixfp
CONFORMER_PROFILE=${CONFORMER_PROFILE:-quality}   # fast | full | quality
REBUILD_FEATURE_CACHE=${REBUILD_FEATURE_CACHE:-0}
GEOM_INPUT=${GEOM_INPUT:-}                        # optional explicit override
PRETRAIN_DATASET=${PRETRAIN_DATASET:-PI1M_50k}
PRETRAIN_NPROC=${PRETRAIN_NPROC:-4}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-0}       # safe for 64 MB /dev/shm
PRETRAIN_ONLY=${PRETRAIN_ONLY:-0}
PRETRAIN_AMP=${PRETRAIN_AMP:-bf16}
MODEL_VERSION=${MODEL_VERSION:-}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-32}
STAGE1_GRAD_ACCUM=${STAGE1_GRAD_ACCUM:-4}
STAGE1_LR=${STAGE1_LR:-1e-4}
STAGE1_DYNAMIC_WARMUP=${STAGE1_DYNAMIC_WARMUP:-1600}
STAGE1_DYNAMIC_WINDOW=${STAGE1_DYNAMIC_WINDOW:-160}
STAGE1_MASK_WEIGHT=${STAGE1_MASK_WEIGHT:-1.0}
STAGE1_SP_WEIGHT=${STAGE1_SP_WEIGHT:-0.5}
STAGE1_GEOMETRY_WEIGHT=${STAGE1_GEOMETRY_WEIGHT:-0.75}
STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE:-64}
STAGE3_BATCH_SIZE=${STAGE3_BATCH_SIZE:-32}
SCAGE_LOCALITY_MODE=${SCAGE_LOCALITY_MODE:-soft}
SCAGE_PERIODIC_IMAGE_MODE=${SCAGE_PERIODIC_IMAGE_MODE:-explicit_images}
SCAGE_PERIODIC_IMAGE_CAP=${SCAGE_PERIODIC_IMAGE_CAP:-1}
SCAGE_FORCE_TOPOLOGY_ONLY=${SCAGE_FORCE_TOPOLOGY_ONLY:-0}
SCAGE_USE_PBC_DISTANCE=${SCAGE_USE_PBC_DISTANCE:-1}
MAX_SMILES_LENGTH=${MAX_SMILES_LENGTH:-}
STAGE3_ONLY=${STAGE3_ONLY:-0}
STAGE2_ONLY=${STAGE2_ONLY:-0}                  # skip Stage 1; run Stage 2 then Stage 3
EXPERIMENT_TAG=${EXPERIMENT_TAG:-}
SCAGE_TOPOLOGY_MAX_DISTANCE=${SCAGE_TOPOLOGY_MAX_DISTANCE:-20}

# Stage-2 alignment knobs. FP remains regularly dropped, but no longer loses
# most of its alignment signal before downstream fine-tuning.
ALIGN_TEMPERATURE=${ALIGN_TEMPERATURE:-0.07}
ALIGN_FP_DROPOUT=${ALIGN_FP_DROPOUT:-0.20}
ALIGN_SMILES_DROPOUT=${ALIGN_SMILES_DROPOUT:-0.15}
ALIGN_GRAPH_DROPOUT=${ALIGN_GRAPH_DROPOUT:-0.10}
ALIGN_FUSED_WEIGHT=${ALIGN_FUSED_WEIGHT:-1.0}
ALIGN_GRAPH_SMILES_WEIGHT=${ALIGN_GRAPH_SMILES_WEIGHT:-0.5}
ALIGN_GRAPH_FP_WEIGHT=${ALIGN_GRAPH_FP_WEIGHT:-0.50}
ALIGN_LOMO_WEIGHT=${ALIGN_LOMO_WEIGHT:-0.25}
ALIGN_POOLING_KL_WEIGHT=${ALIGN_POOLING_KL_WEIGHT:-0.01}
ALIGN_FUSED_MASK_WEIGHT=${ALIGN_FUSED_MASK_WEIGHT:-0.25}
ALIGN_ENCODER_LR=${ALIGN_ENCODER_LR:-1e-5}
ALIGN_FP_LR=${ALIGN_FP_LR:-2e-5}
ALIGN_PROJECTION_LR=${ALIGN_PROJECTION_LR:-1e-4}
ALIGN_FUSION_LR=${ALIGN_FUSION_LR:-1e-4}

# Stage-3 regularization/optimization knobs. The v3 run suppressed FP to about
# four percent mean pooling weight, so v4 restores it with low-rate unfreezing.
FP_BIT_DROPOUT=${FP_BIT_DROPOUT:-0.05}
FP_MODALITY_DROPOUT=${FP_MODALITY_DROPOUT:-0.10}
SMILES_MODALITY_DROPOUT=${SMILES_MODALITY_DROPOUT:-0.10}
GRAPH_MODALITY_DROPOUT=${GRAPH_MODALITY_DROPOUT:-0.05}
FP_UNFREEZE_EPOCH=${FP_UNFREEZE_EPOCH:-10}
SMILES_LR=${SMILES_LR:-5e-6}
GRAPH_LR=${GRAPH_LR:-}
FP_LR=${FP_LR:-5e-6}
FUSION_LR=${FUSION_LR:-5e-5}
HEAD_LR=${HEAD_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-}
DEEP_UNFREEZE_EPOCH=${DEEP_UNFREEZE_EPOCH:-10}
FREEZE_ENCODER_EPOCHS=${FREEZE_ENCODER_EPOCHS:-5}
REGRESSION_LOSS=${REGRESSION_LOSS:-}             # huber | mse
HUBER_BETA=${HUBER_BETA:-0.5}                  # used only by huber
HEAD_DROPOUT=${HEAD_DROPOUT:-0.30}
UNIMODAL_AUX_WEIGHT=${UNIMODAL_AUX_WEIGHT:-0.0}
CROSS_TASK_AUX_WEIGHT=${CROSS_TASK_AUX_WEIGHT:-0.0}
CROSS_TASK_AUX_TASKS=${CROSS_TASK_AUX_TASKS:-""}
read -r -a CROSS_TASK_AUX_TASK_LIST <<< "$CROSS_TASK_AUX_TASKS"
FUSION_PRIOR_KL_WEIGHT=${FUSION_PRIOR_KL_WEIGHT:-0.0}
FUSION_PRIOR=${FUSION_PRIOR:-"0.30 0.40 0.30"}
read -r -a FUSION_PRIOR_VALUES <<< "$FUSION_PRIOR"
REFIT_FULL_TRAIN=${REFIT_FULL_TRAIN:-0}
REFIT_FULL_TRAIN_ARGS=(--no-refit_full_train)
if [[ "$REFIT_FULL_TRAIN" == "1" ]]; then
    REFIT_FULL_TRAIN_ARGS=(--refit_full_train)
fi
SWA_START_EPOCH=${SWA_START_EPOCH:--1}
TARGET_TRANSFORM=${TARGET_TRANSFORM:-}             # recommended | standard | auto | log
RANDOM_SEED=${RANDOM_SEED:-42}
TASKS=${TASKS:-"eat eea egb egc ei eps nc xc"}
read -r -a TASK_LIST <<< "$TASKS"
FOLD_IDS=${FOLD_IDS:-"0 1 2 3 4"}
read -r -a FOLD_ID_LIST <<< "$FOLD_IDS"

case "$PRETRAIN_DATASET" in
    smi_all) PRETRAIN_TAG=smi ;;
    PI1M_v2) PRETRAIN_TAG=pi1m ;;
    PI1M_20k) PRETRAIN_TAG=pi1m20k ;;
    PI1M_50k) PRETRAIN_TAG=pi1m50k ;;
    PI1M_*k)
        if [[ ! -f "data/raw/${PRETRAIN_DATASET}.csv" ]]; then
            echo "Missing data/raw/${PRETRAIN_DATASET}.csv. Create it with scripts/create_pi1m_subset.py." >&2
            exit 2
        fi
        PRETRAIN_TAG=$(printf '%s' "$PRETRAIN_DATASET" | tr '[:upper:]' '[:lower:]' | tr -d '_')
        ;;
    *)
        echo "Unsupported PRETRAIN_DATASET=$PRETRAIN_DATASET. Use smi_all, PI1M_<N>k, or PI1M_v2." >&2
        exit 2
        ;;
esac
if [[ -z "$MAX_SMILES_LENGTH" && "$PRETRAIN_DATASET" == PI1M_* ]]; then
    MAX_SMILES_LENGTH=141
fi

if [[ "$BASELINE" == "mips" || "$BASELINE" == "scage_mips_ablation" ]]; then
    echo "BASELINE=$BASELINE has been removed. Use BASELINE=flat4 or BASELINE=scage_parallel." >&2
    exit 2
fi

case "$BASELINE" in
    flat4)
        # FP + SMILES + GIN Graph + PaiNN Geom: four modality tokens are fused together.
        GRAPH_ENCODER_TYPE=gin
        FUSION_TYPE=self_attention_pooling
        ;;
    scage_parallel)
        # SMILES, SCAGE structure, and FP are fused as peer modality tokens.
        GRAPH_ENCODER_TYPE=scage
        FUSION_TYPE=parallel_attention
        ;;
    *)
        echo "Unsupported BASELINE=$BASELINE. Use flat4 or scage_parallel." >&2
        exit 2
        ;;
esac

GRAPH_GEOM_EPOCHS=${GRAPH_GEOM_EPOCHS:-10}
GRAPH_LR=${GRAPH_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.02}
REGRESSION_LOSS=${REGRESSION_LOSS:-huber}
TARGET_TRANSFORM=${TARGET_TRANSFORM:-recommended}
MODEL_VERSION=${MODEL_VERSION:-v5}

if [[ -z "$GEOM_INPUT" ]]; then
    # SCAGE defaults to deterministic PolyGen-inspired fractional PBC.
    if [[ "$BASELINE" == "scage_parallel" ]]; then
        GEOM_INPUT=polygen_periodic
    else
        GEOM_INPUT=periodic_pbc
    fi
fi

case "$GEOM_INPUT" in
    repeat_unit|periodic_pbc|polygen_periodic|screw_periodic|smer_context) ;;
    *)
        echo "Unsupported GEOM_INPUT=$GEOM_INPUT." >&2
        exit 2
        ;;
esac
if [[ "$BASELINE" == "scage_parallel" && "$GEOM_INPUT" != "polygen_periodic" ]]; then
    echo "BASELINE=scage_parallel now requires GEOM_INPUT=polygen_periodic for MIPS-PBC LGA." >&2
    exit 2
fi
if [[ "$GEOM_INPUT" == "polygen_periodic" || "$GEOM_INPUT" == "screw_periodic" || "$GEOM_INPUT" == "smer_context" ]]; then
    if [[ "$GRAPH_ENCODER_TYPE" != "scage" ]]; then
        echo "GEOM_INPUT=$GEOM_INPUT requires BASELINE=scage_parallel." >&2
        exit 2
    fi
fi

case "$CONFORMER_PROFILE" in
    fast)
        # Bounded diversity for routine experiments: rank four ETKDG candidates
        # and retain the two lowest-energy conformers.
        CONFORMER_ARGS=(--conformer_profile fast --conformer_3d_count 4 --conformer_keep_count 2)
        ;;
    full)
        # Explicitly state the quality-oriented defaults instead of relying on
        # argparse defaults; this also makes cache provenance unambiguous.
        CONFORMER_ARGS=(--conformer_profile full --conformer_3d_count 8 --conformer_keep_count 4)
        ;;
    quality)
        # Use the same exhaustive profile as the standalone PBC quality audit.
        CONFORMER_ARGS=(--conformer_profile quality --conformer_3d_count 4 --conformer_keep_count 4)
        ;;
    *)
        echo "Unsupported CONFORMER_PROFILE=$CONFORMER_PROFILE. Use fast, full, or quality." >&2
        exit 2
        ;;
esac

if [[ "$BASELINE" == "scage_parallel" ]]; then
    case "$GEOM_INPUT" in
        polygen_periodic) GEOM_TAG=polygen ;;
        screw_periodic) GEOM_TAG=screw ;;
        smer_context) GEOM_TAG=smer ;;
        *) GEOM_TAG="${GEOM_INPUT}" ;;
    esac
    RUN_TAG="scage_${GEOM_TAG}_${FP_MODE}_${CONFORMER_PROFILE}_${PRETRAIN_TAG}"
else
    RUN_TAG="${BASELINE}_${GEOM_INPUT}_${FP_MODE}_${CONFORMER_PROFILE}_${PRETRAIN_TAG}"
fi
if [[ -n "$EXPERIMENT_TAG" ]]; then
    RUN_TAG="${RUN_TAG}_${EXPERIMENT_TAG}"
fi
RUN_TAG="${RUN_TAG}_${MODEL_VERSION}"
LOG_FILE="./logs/run_${RUN_TAG}.log"
GRAPH_GEOM_PRETRAIN_PATH="./pretrained_models/${RUN_TAG}_graph_geom.pth"
ALIGN_PRETRAIN_PATH="./pretrained_models/${RUN_TAG}_alignment.pth"
if [[ -n "${GRAPH_GEOM_CHECKPOINT:-}" ]]; then
    GRAPH_GEOM_PRETRAIN_PATH="$GRAPH_GEOM_CHECKPOINT"
fi
if [[ -n "${ALIGN_CHECKPOINT:-}" ]]; then
    ALIGN_PRETRAIN_PATH="$ALIGN_CHECKPOINT"
fi
RESULT_PATH="./results/${RUN_TAG}.csv"
MODEL_DIR="./saved_models/${RUN_TAG}"

mkdir -p logs pretrained_models results
# mkdir -p "$MODEL_DIR"  # Downstream saved_models output is disabled in train.py.
: > "$LOG_FILE"

if [[ "$PRETRAIN_DATASET" == "PI1M_50k" && ! -f data/raw/PI1M_50k.csv ]]; then
    PI1M_PARTIAL_CACHE=${PI1M_PARTIAL_CACHE:-data/processed/scage/feature_cache_PI1M_v2_scage-starlink-backbone-input-v1-m4p-v1_geom-screw-periodic-forward-screw-v2-energytop2_cand4_quality_fp-ecfp_tok141.pt.partial}
    if [[ ! -f "$PI1M_PARTIAL_CACHE" ]]; then
        echo "PI1M partial cache not found: $PI1M_PARTIAL_CACHE" >&2
        exit 2
    fi
    "$PYTHON_BIN" scripts/materialize_partial_feature_cache.py \
        --partial-cache "$PI1M_PARTIAL_CACHE" \
        --source-csv data/raw/PI1M_v2.csv \
        --output-dataset PI1M_50k \
        --sample-size 50000 \
        --seed 42 \
        --workers 16 2>&1 | tee -a "$LOG_FILE"
fi

if [[ "$STAGE2_ONLY" == "1" && "$STAGE3_ONLY" == "1" ]]; then
    echo "STAGE2_ONLY=1 and STAGE3_ONLY=1 are mutually exclusive." >&2
    exit 2
fi
if [[ "$STAGE2_ONLY" == "1" && ! -f "$GRAPH_GEOM_PRETRAIN_PATH" ]]; then
    echo "Stage-1 checkpoint not found: $GRAPH_GEOM_PRETRAIN_PATH" >&2
    exit 2
fi

# Common model/data configuration. All omitted values use the CLI defaults:
# PaiNN, 4 graph layers, 256 dimensions, GraphNorm, residual GIN, 8/4/8
# conformer generation, and AdamW downstream parameter groups.
COMMON_ARGS=(
    --seed "$RANDOM_SEED"
    --graph_input star_linking
    --geom_input "$GEOM_INPUT"
    --graph_encoder_type "$GRAPH_ENCODER_TYPE"
    --fp_mode "$FP_MODE"
    --fusion_type "$FUSION_TYPE"
    "${CONFORMER_ARGS[@]}"
)
if [[ -n "$MAX_SMILES_LENGTH" ]]; then
    COMMON_ARGS+=(--max_smiles_length "$MAX_SMILES_LENGTH")
fi

if [[ "$BASELINE" == "scage_parallel" ]]; then
    STAGE1_MODALITIES=(graph)
    STAGE23_MODALITIES=(smiles graph fp)
else
    STAGE1_MODALITIES=(smiles graph fp geom)
    STAGE23_MODALITIES=(smiles graph fp geom)
fi

if [[ "$BASELINE" == "scage_parallel" ]]; then
    # Sparse MIPS-PBC Graph Transformer; the legacy dense SCAGE encoder remains
    # in source only and is not instantiated by this route.
    COMMON_ARGS+=(
        --graph_num_layers 6
        --graph_emb_dim 512
        --scage_num_heads 8
        --scage_ffn_hidden_dim 2048
        --scage_num_kernels 128
        --scage_attention_dropout 0.1
        --scage_distance_mode bias
        --scage_distance_rbf 64
        --scage_distance_cutoff 12.0
        --no-scage_use_descriptors
        --parallel_attention_layers 1
        --fusion_dropout 0.2
    )
fi

CACHE_WORKERS=${CACHE_WORKERS:-16}
if [[ "$PRETRAIN_DATASET" == PI1M_* ]]; then
    CACHE_PARTIAL_EVERY=5000
else
    CACHE_PARTIAL_EVERY=50
fi
CACHE_ARGS=(
    --feature_cache_workers "$CACHE_WORKERS"
    --feature_cache_chunksize 2
    --feature_cache_partial_every "$CACHE_PARTIAL_EVERY"
    --feature_cache_item_timeout 45
)
STAGE3_CACHE_PARTIAL_EVERY=${STAGE3_CACHE_PARTIAL_EVERY:-500}

# Random-range coordinate denoising is non-default and shared by both stages.
GEOM_NOISE_ARGS=(
    --geom_noise_std_min 0.05
    --geom_noise_std_max 0.2
)

run_stage() {
    local stage_name="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} 开始" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} 完成" | tee -a "$LOG_FILE"
}

PRETRAIN_LAUNCH=("$PYTHON_BIN")
if [[ "$PRETRAIN_NPROC" -gt 1 ]]; then
    PRETRAIN_LAUNCH=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$PRETRAIN_NPROC")
fi

# Build geometry/graph features in a single CPU-only process before torchrun.
# This keeps fork-based RDKit workers fast and prevents inherited CUDA state.
if [[ "$REBUILD_FEATURE_CACHE" == "1" && "$STAGE3_ONLY" != "1" && "$STAGE2_ONLY" != "1" ]]; then
    if [[ "$BASELINE" == "scage_parallel" ]]; then
        CACHE_STAGE=scage_m4p
    else
        CACHE_STAGE=graph_geom
    fi
    run_stage "Feature cache CPU 预构建" \
        env CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/pretrain.py \
            --dataset_name "$PRETRAIN_DATASET" \
            --modalities "${STAGE1_MODALITIES[@]}" \
            "${COMMON_ARGS[@]}" \
            "${CACHE_ARGS[@]}" \
            --pretrain_stage "$CACHE_STAGE" \
            --cache_only \
            --rebuild_feature_cache
fi

if [[ "$STAGE3_ONLY" != "1" && "$BASELINE" == "scage_parallel" ]]; then
    if [[ "$STAGE2_ONLY" != "1" ]]; then
        run_stage "Stage 1/3 Polymer-SCAGE M4P 预训练" \
            "${PRETRAIN_LAUNCH[@]}" scripts/pretrain.py \
                --dataset_name "$PRETRAIN_DATASET" \
                --modalities "${STAGE1_MODALITIES[@]}" \
                "${COMMON_ARGS[@]}" \
                "${CACHE_ARGS[@]}" \
                --pretrain_stage scage_m4p \
                --pretrain_profile mips24h \
                --epochs "$GRAPH_GEOM_EPOCHS" \
                --batch_size "$STAGE1_BATCH_SIZE" \
                --gradient_accumulation_steps "$STAGE1_GRAD_ACCUM" \
                --loader_workers "$DATALOADER_WORKERS" \
                --amp_dtype "$PRETRAIN_AMP" \
                --lr "$STAGE1_LR" \
                --graph_mask_ratio 0.30 \
                --dynamic_pretrain_loss \
                --dynamic_loss_warmup_steps "$STAGE1_DYNAMIC_WARMUP" \
                --dynamic_loss_recent_window "$STAGE1_DYNAMIC_WINDOW" \
                --dynamic_loss_temperature 1.0 \
                --scage_mips_mask_weight "$STAGE1_MASK_WEIGHT" \
                --scage_periodic_sp_weight "$STAGE1_SP_WEIGHT" \
                --scage_periodic_geometry_weight "$STAGE1_GEOMETRY_WEIGHT" \
                --scage_geometry_max_pairs 32 \
                --save_path "$GRAPH_GEOM_PRETRAIN_PATH"
    fi

    run_stage "Stage 2/3 SCAGE 语义对齐预训练" \
        "${PRETRAIN_LAUNCH[@]}" scripts/pretrain.py \
            --dataset_name "$PRETRAIN_DATASET" \
            --modalities "${STAGE23_MODALITIES[@]}" \
            "${COMMON_ARGS[@]}" \
            --pretrain_stage alignment \
            --pretrained_model_path "$GRAPH_GEOM_PRETRAIN_PATH" \
            --epochs "$ALIGN_EPOCHS" \
            --batch_size "$STAGE2_BATCH_SIZE" \
            --gradient_accumulation_steps 1 \
            --loader_workers "$DATALOADER_WORKERS" \
            --amp_dtype "$PRETRAIN_AMP" \
            --temperature "$ALIGN_TEMPERATURE" \
            --alignment_fp_drop "$ALIGN_FP_DROPOUT" \
            --alignment_smiles_drop "$ALIGN_SMILES_DROPOUT" \
            --alignment_graph_drop "$ALIGN_GRAPH_DROPOUT" \
            --alignment_fused_weight "$ALIGN_FUSED_WEIGHT" \
            --alignment_graph_smiles_weight "$ALIGN_GRAPH_SMILES_WEIGHT" \
            --alignment_graph_fp_weight "$ALIGN_GRAPH_FP_WEIGHT" \
            --alignment_lomo_weight "$ALIGN_LOMO_WEIGHT" \
            --alignment_pooling_kl_weight "$ALIGN_POOLING_KL_WEIGHT" \
            --alignment_fused_mask_weight "$ALIGN_FUSED_MASK_WEIGHT" \
            --alignment_graph_lr "$ALIGN_ENCODER_LR" \
            --alignment_smiles_lr "$ALIGN_ENCODER_LR" \
            --alignment_fp_lr "$ALIGN_FP_LR" \
            --alignment_projection_lr "$ALIGN_PROJECTION_LR" \
            --alignment_fusion_lr "$ALIGN_FUSION_LR" \
            --no-dynamic_pretrain_loss \
            --save_path "$ALIGN_PRETRAIN_PATH"
elif [[ "$STAGE3_ONLY" != "1" ]]; then
    if [[ "$STAGE2_ONLY" != "1" ]]; then
        run_stage "Stage 1/3 Graph + Geom 单模态预训练" \
            "$PYTHON_BIN" scripts/pretrain.py \
                --dataset_name "$PRETRAIN_DATASET" \
                --modalities "${STAGE1_MODALITIES[@]}" \
                "${COMMON_ARGS[@]}" \
                "${CACHE_ARGS[@]}" \
                "${GEOM_NOISE_ARGS[@]}" \
                --pretrain_stage graph_geom \
                --epochs "$GRAPH_GEOM_EPOCHS" \
                --no-dynamic_pretrain_loss \
                --graph_shortest_path_weight 0.2 \
                --graph_angle_weight 0.2 \
                --save_path "$GRAPH_GEOM_PRETRAIN_PATH"
    fi

    run_stage "Stage 2/3 多模态对齐预训练" \
        "$PYTHON_BIN" scripts/pretrain.py \
            --dataset_name "$PRETRAIN_DATASET" \
            --modalities "${STAGE23_MODALITIES[@]}" \
            "${COMMON_ARGS[@]}" \
            "${GEOM_NOISE_ARGS[@]}" \
            --pretrain_stage alignment \
            --pretrained_model_path "$GRAPH_GEOM_PRETRAIN_PATH" \
            --epochs "$ALIGN_EPOCHS" \
            --dynamic_pretrain_loss \
            --dynamic_loss_warmup_steps 200 \
            --graph_pretrain_weight 0.1 \
            --geom_denoise_weight 0.1 \
            --graph_periodic_aug_weight 0.1 \
            --graph_shortest_path_weight 0.05 \
            --graph_angle_weight 0.05 \
            --save_path "$ALIGN_PRETRAIN_PATH"
fi

if [[ "$PRETRAIN_ONLY" == "1" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 24h pretraining scope complete: $ALIGN_PRETRAIN_PATH" | tee -a "$LOG_FILE"
    exit 0
fi

run_stage "Stage 3/3 下游 5-fold 监督训练" \
    "$PYTHON_BIN" scripts/train.py \
        --modalities "${STAGE23_MODALITIES[@]}" \
        "${COMMON_ARGS[@]}" \
        "${CACHE_ARGS[@]}" \
        --feature_cache_partial_every "$STAGE3_CACHE_PARTIAL_EVERY" \
        --tasks "${TASK_LIST[@]}" \
        --fold_ids "${FOLD_ID_LIST[@]}" \
        --loader_workers "$DATALOADER_WORKERS" \
        --pretrained_model_path "$ALIGN_PRETRAIN_PATH" \
        --epochs "$TRAIN_EPOCHS" \
        --batch_size "$STAGE3_BATCH_SIZE" \
        --smiles_lr "$SMILES_LR" \
        --graph_lr "$GRAPH_LR" \
        --fp_lr "$FP_LR" \
        --fusion_lr "$FUSION_LR" \
        --head_lr "$HEAD_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --fp_bit_dropout "$FP_BIT_DROPOUT" \
        --fp_modality_dropout "$FP_MODALITY_DROPOUT" \
        --smiles_modality_dropout "$SMILES_MODALITY_DROPOUT" \
        --graph_modality_dropout "$GRAPH_MODALITY_DROPOUT" \
        --fp_unfreeze_epoch "$FP_UNFREEZE_EPOCH" \
        --freeze_smiles_epochs "$FREEZE_ENCODER_EPOCHS" \
        --deep_unfreeze_epoch "$DEEP_UNFREEZE_EPOCH" \
        --head_dropout "$HEAD_DROPOUT" \
        --regression_loss "$REGRESSION_LOSS" \
        --huber_beta "$HUBER_BETA" \
        --unimodal_aux_weight "$UNIMODAL_AUX_WEIGHT" \
        --cross_task_aux_weight "$CROSS_TASK_AUX_WEIGHT" \
        --cross_task_aux_tasks "${CROSS_TASK_AUX_TASK_LIST[@]}" \
        --fusion_prior_kl_weight "$FUSION_PRIOR_KL_WEIGHT" \
        --fusion_prior "${FUSION_PRIOR_VALUES[@]}" \
        --swa_start_epoch "$SWA_START_EPOCH" \
        --target_transform "$TARGET_TRANSFORM" \
        "${REFIT_FULL_TRAIN_ARGS[@]}" \
        --results_dir "$RESULT_PATH" \
        --models_dir "$MODEL_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${BASELINE} 三阶段流程完成。结果：$RESULT_PATH" | tee -a "$LOG_FILE"
echo "总日志：$LOG_FILE"
