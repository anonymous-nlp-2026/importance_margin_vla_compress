#!/bin/bash
# activate your conda environment
conda activate base
cd .
source /etc/network_turbo
export HF_HOME=./cache
export HF_HUB_DISABLE_XET=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export CUDA_VISIBLE_DEVICES=1

mkdir -p eval_results

echo "=== Eval 1: v8 standard (500 batches) ==="
python3 -u evaluate.py \
    --checkpoint checkpoints/imm_anchor_v8/step_029999 \
    --config configs/imm_anchor_v8.yaml \
    --max_batches 500 --seed 42 \
    --output eval_results/eval_imm_v8_fixed.json

echo "=== Eval 2: v8 bypass-ACIS (500 batches) ==="
python3 -u evaluate.py \
    --checkpoint checkpoints/imm_anchor_v8/step_029999 \
    --config configs/imm_anchor_v8.yaml \
    --max_batches 500 --seed 42 --bypass_acis \
    --output eval_results/eval_imm_v8_bypass_fixed.json

echo "=== Eval 3: baseline (500 batches) ==="
python3 -u evaluate.py \
    --checkpoint checkpoints/baseline/step_029999 \
    --config configs/baseline.yaml \
    --max_batches 500 --seed 42 \
    --output eval_results/eval_baseline_v5_fixed.json

echo "=== ALL DONE $(date) ==="
