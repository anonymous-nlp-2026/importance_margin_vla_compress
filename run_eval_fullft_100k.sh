#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/importance_margin_vla_compress
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

CKPT="checkpoints/libero_full_ft/step_099999/checkpoint.pt"
CFG="configs/libero_full_ft.yaml"

echo "=== libero_object ==="
python -u eval_success_rate_libero.py \
    --checkpoint "$CKPT" --config "$CFG" \
    --suite libero_object --num_episodes 10 --gpu 0 \
    --output eval_results/fullft_100k_object.json

echo "=== libero_spatial ==="
python -u eval_success_rate_libero.py \
    --checkpoint "$CKPT" --config "$CFG" \
    --suite libero_spatial --num_episodes 10 --gpu 0 \
    --output eval_results/fullft_100k_spatial.json

echo "=== libero_goal ==="
python -u eval_success_rate_libero.py \
    --checkpoint "$CKPT" --config "$CFG" \
    --suite libero_goal --num_episodes 10 --gpu 0 \
    --output eval_results/fullft_100k_goal.json

echo "=== ALL DONE $(date) ==="
