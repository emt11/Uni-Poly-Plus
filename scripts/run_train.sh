#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1
# Define the command to run the training script with desired parameters
# --tasks tg eat eea egb egc ei eps nc xc \
nohup python scripts/train.py \
    --modalities smiles graph fp geom kg \
    --geometry_encoder schnet \
    --tasks tg eat eea egb egc ei eps nc xc \
    --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
    > ./logs/train.log 2>&1 &
