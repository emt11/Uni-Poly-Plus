#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PYTHON_BIN=${PYTHON_BIN:-/root/anaconda3/envs/Uni-Poly/bin/python}

TAG=${TAG:-scage_polygen_ecfp_quality_pi1m50k}
ALIGN_CHECKPOINT=${ALIGN_CHECKPOINT:-"./pretrained_models/${TAG}_alignment.pth"}
mkdir -p logs results "saved_models/${TAG}"

"$PYTHON_BIN" scripts/train.py \
    --modalities smiles graph fp \
    --graph_encoder_type scage \
    --graph_input star_linking \
    --geom_input polygen_periodic \
    --fp_mode ecfp \
    --fusion_type parallel_attention \
    --parallel_attention_layers 1 \
    --graph_num_layers 6 \
    --graph_emb_dim 512 \
    --scage_num_heads 16 \
    --scage_ffn_hidden_dim 256 \
    --scage_num_kernels 128 \
    --scage_distance_mode mips_dual \
    --scage_distance_rbf 32 \
    --scage_distance_cutoff 12.0 \
    --scage_topology_bias \
    --scage_topology_max_distance 20 \
    --scage_topology_locality_mode soft \
    --scage_periodic_image_mode explicit_images \
    --scage_periodic_image_cap 1 \
    --conformer_profile quality \
    --conformer_3d_count 4 \
    --conformer_keep_count 4 \
    --tasks eat eea egb egc ei eps nc xc \
    --pretrained_model_path "$ALIGN_CHECKPOINT" \
    --epochs 100 \
    --patience 10 \
    --results_dir "./results/${TAG}.csv" \
    --models_dir "./saved_models/${TAG}" \
    2>&1 | tee "./logs/train_${TAG}.log"
