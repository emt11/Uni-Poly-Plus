#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs saved_models_dimer results

# 运行全部任务下游训练，使用 dimer graph。
# 预训练权重应由 scripts/run_pretrain.sh 生成：
# ./pretrained_models/saved_pretrained_model_dimer_all.pth
nohup python scripts/train.py \
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
    --models_dir ./saved_models_dimer \
    > ./logs/train_dimer_all.log 2>&1 &

echo "已启动全部任务 dimer graph 下游训练。日志：./logs/train_dimer_all.log"
echo "查看进度：tail -f ./logs/train_dimer_all.log"
