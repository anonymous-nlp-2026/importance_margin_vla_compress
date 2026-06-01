#!/bin/bash
# M=128 control: cross-attention pooling without compression (128→128)
# Purpose: verify if SR > baseline comes from cross-attention architecture vs compression ratio
# DO NOT START — wait for Director approval after M=96/M=16 finish

cd /root/autodl-tmp/importance_margin_vla_compress
source /root/miniconda3/bin/activate

CUDA_VISIBLE_DEVICES=X HF_HOME=/root/autodl-tmp/.hf_cache HF_HUB_OFFLINE=1 WANDB_MODE=offline \
nohup python -u train_smart_compress.py \
  --config configs/smart_compress_m128.yaml \
  --wandb_run smart_compress_m128 \
  > train_M128.log 2>&1 &
echo "M=128 PID: $!"
