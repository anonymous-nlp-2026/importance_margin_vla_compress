#!/bin/bash
set -e
cd /root/autodl-tmp/importance_margin_vla_compress
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
source /etc/network_turbo 2>/dev/null || true
export PYTHONUNBUFFERED=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export CUDA_VISIBLE_DEVICES=1

echo "$(date) === Standard eval step_028000 ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_028000 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 100 --seed 42 \
  --output eval_results/trajectory/v8_step_028000_standard.json

echo "$(date) === Bypass eval step_028000 ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_028000 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 100 --seed 42 \
  --bypass_acis \
  --output eval_results/trajectory/v8_step_028000_bypass.json

echo "$(date) === Trajectory standard eval step_029999 ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_029999 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 100 --seed 42 \
  --output eval_results/trajectory/v8_step_029999_standard.json

echo "$(date) === Trajectory bypass eval step_029999 ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_029999 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 100 --seed 42 \
  --bypass_acis \
  --output eval_results/trajectory/v8_step_029999_bypass.json

echo "$(date) === 500-batch bypass eval step_029999 ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_029999 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 500 --seed 42 \
  --bypass_acis \
  --output eval_results/eval_imm_v8_bypass_final.json

echo "$(date) === ALL TRAJECTORY + BYPASS DONE ==="
