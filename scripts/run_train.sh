#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs saved_models_starlink_schnet_multistage_8tasks results

# 运行全部任务下游训练，使用 star-linking graph。
# 预训练权重应由 scripts/run_pretrain.sh 生成：
# ./pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth
nohup python scripts/train.py \
    --modalities smiles graph fp geom \
    --geometry_encoder schnet \
    --geom_model_name "" \
    --graph_input star_linking \
    --tasks eat eea egb egc ei eps nc xc \
    --pretrained_model_path ./pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth \
    --graph_num_layers 6 \
    --graph_emb_dim 256 \
    --graph_dropout 0.1 \
    --graph_pooling attention \
    --joint_embedding_dim 256 \
    --epochs 100 \
    --patience 10 \
    --results_dir ./results/results_starlink_schnet_multistage_8tasks.csv \
    --models_dir ./saved_models_starlink_schnet_multistage_8tasks \
    > ./logs/train_starlink_schnet_multistage_8tasks.log 2>&1 &

echo "已启动全部任务 star-linking + SchNet 下游训练。日志：./logs/train_starlink_schnet_multistage_8tasks.log"
echo "查看进度：tail -f ./logs/train_starlink_schnet_multistage_8tasks.log"
