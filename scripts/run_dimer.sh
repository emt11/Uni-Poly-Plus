#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

mkdir -p logs pretrained_models saved_models_dimer results

LOG_FILE="./logs/run_dimer.log"
: > "$LOG_FILE"

run_stage() {
    local stage_name="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} 开始" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${stage_name} 完成" | tee -a "$LOG_FILE"
}

run_stage "Stage 1/2 Uni-Poly Dimer 多模态预训练：smi_all" \
    python scripts/pretrain.py \
        --dataset_name smi_all \
        --modalities smiles graph fp geom \
        --geometry_encoder schnet \
        --geom_model_name ./pretrained_models/encoders/schnet_qm9_gap.pth \
        --graph_input dimer \
        --joint_embedding_dim 256 \
        --epochs 20 \
        --save_path ./pretrained_models/saved_pretrained_model_dimer_all.pth

run_stage "Stage 2/2 Dimer 全部任务下游训练" \
    python scripts/train.py \
        --modalities smiles graph fp geom \
        --geometry_encoder schnet \
        --geom_model_name ./pretrained_models/encoders/schnet_qm9_gap.pth \
        --graph_input dimer \
        --tasks eat eea egb egc ei eps nc tg xc \
        --pretrained_model_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
        --joint_embedding_dim 256 \
        --epochs 100 \
        --patience 10 \
        --results_dir ./results/results_dimer_all.csv \
        --models_dir ./saved_models_dimer

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dimer 流程完成。结果：./results/results_dimer_all.csv" | tee -a "$LOG_FILE"
echo "总日志：$LOG_FILE"
