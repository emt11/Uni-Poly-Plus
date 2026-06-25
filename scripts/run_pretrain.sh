#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs pretrained_models

# dimer graph 多模态预训练：
# - graph_input=dimer：使用二聚体局部链段图，失败自动回退 repeat_unit
nohup python scripts/pretrain.py \
    --dataset_name smi_all \
    --modalities smiles graph fp geom \
    --geometry_encoder schnet \
    --geom_model_name ./pretrained_models/encoders/schnet_qm9_gap.pth \
    --graph_input dimer \
    --joint_embedding_dim 256 \
    --epochs 20 \
    --save_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
    > ./logs/pretrain_dimer_all.log 2>&1 &

echo "已启动 dimer graph 多模态预训练。日志：./logs/pretrain_dimer_all.log"
echo "查看进度：tail -f ./logs/pretrain_dimer_all.log"
