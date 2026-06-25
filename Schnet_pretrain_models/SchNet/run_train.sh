#!/bin/bash
# SchNet 在 QM9 上的预训练脚本
# 使用方法: bash run_train.sh [property_name]

# 设置默认参数
PROPERTY=${1:-"cv"}
EPOCHS=${2:-1000}
BATCH_SIZE=${3:-32}
LEARNING_RATE=${4:-1e-4}
PATIENCE=${5:-100}

echo "=========================================="
echo "SchNet Pretraining on QM9"
echo "=========================================="
echo "Property: $PROPERTY"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Patience: $PATIENCE"
echo "=========================================="

# 运行训练
python pretrain_models/SchNet/train_schnet.py \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --patience $PATIENCE \
    --device cuda \
    --num_workers 4 \
    --save_dir ./pretrained_models/encoders \
    --seed 42

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Model saved to: ./pretrained_models/encoders/schnet_qm9_${PROPERTY}.pth"
echo ""
echo "Next steps:"
echo "1. Update Uni-Poly config:"
echo "   geom_model_name='./pretrained_models/encoders/schnet_qm9_${PROPERTY}.pth'"
echo "2. Run Uni-Poly training"
echo "=========================================="
