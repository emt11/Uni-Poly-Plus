#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

LOG_FILE="./logs/run_starlink_schnet_multistage_8tasks.log"
GRAPH_GEOM_PRETRAIN_PATH="./pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth"
ALIGN_PRETRAIN_PATH="./pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth"
RESULT_PATH="./results/results_starlink_schnet_multistage_8tasks.csv"
MODEL_DIR="./saved_models_starlink_schnet_multistage_8tasks"

GRAPH_GEOM_EPOCHS=${GRAPH_GEOM_EPOCHS:-20}
ALIGN_EPOCHS=${ALIGN_EPOCHS:-10}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}

mkdir -p logs pretrained_models results "$MODEL_DIR"
: > "$LOG_FILE"

run_stage() {
    local stage_name="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} 开始" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} 完成" | tee -a "$LOG_FILE"
}

COMMON_PRETRAIN_ARGS=(
    --dataset_name smi_all
    --modalities smiles graph fp geom
    --geometry_encoder schnet
    --geom_model_name ""
    --graph_input star_linking
    --graph_num_layers 6
    --graph_emb_dim 256
    --graph_dropout 0.1
    --graph_pooling attention
    --joint_embedding_dim 256
    --geom_noise_std 0.2
    --geom_distance_loss_weight 1.0
    --graph_mask_atom_weight 1.0
    --graph_mask_ratio 0.15
)

run_stage "Stage 1/3 Graph + Geom 单模态预训练：star-linking/backbone + schnet denoising" \
    python scripts/pretrain.py \
        "${COMMON_PRETRAIN_ARGS[@]}" \
        --pretrain_stage graph_geom \
        --epochs "$GRAPH_GEOM_EPOCHS" \
        --geom_denoise_weight 1.0 \
        --graph_pretrain_weight 1.0 \
        --graph_starlink_consistency_weight 0.5 \
        --rebuild_feature_cache \
        --save_path "$GRAPH_GEOM_PRETRAIN_PATH"

run_stage "Stage 2/3 多模态对齐预训练：加载 Graph/Geom 权重后进行 contrastive alignment" \
    python scripts/pretrain.py \
        "${COMMON_PRETRAIN_ARGS[@]}" \
        --pretrain_stage alignment \
        --pretrained_model_path "$GRAPH_GEOM_PRETRAIN_PATH" \
        --epochs "$ALIGN_EPOCHS" \
        --geom_denoise_weight 0.1 \
        --graph_pretrain_weight 0.1 \
        --graph_starlink_consistency_weight 0.1 \
        --save_path "$ALIGN_PRETRAIN_PATH"

run_stage "Stage 3/3 下游 5-fold 监督训练：加载 alignment 权重" \
    python scripts/train.py \
        --modalities smiles graph fp geom \
        --geometry_encoder schnet \
        --geom_model_name "" \
        --graph_input star_linking \
        --tasks eat eea egb egc ei eps nc xc \
        --pretrained_model_path "$ALIGN_PRETRAIN_PATH" \
        --graph_num_layers 6 \
        --graph_emb_dim 256 \
        --graph_dropout 0.1 \
        --graph_pooling attention \
        --joint_embedding_dim 256 \
        --epochs "$TRAIN_EPOCHS" \
        --patience 10 \
        --results_dir "$RESULT_PATH" \
        --models_dir "$MODEL_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Star-linking + Backbone / SchNet 多阶段流程完成。结果：$RESULT_PATH" | tee -a "$LOG_FILE"
echo "总日志：$LOG_FILE"
