#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1
# Define the command to run the training script with desired parameters
nohup python scripts/pretrain.py \
    --dataset_name smi_all \
    --modalities smiles graph fp geom kg \
    --geometry_encoder schnet \
    > ./logs/pretrain.log 2>&1 &


