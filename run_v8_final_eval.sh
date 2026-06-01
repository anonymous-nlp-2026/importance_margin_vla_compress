#!/bin/bash
set -e
GPU=${1:-1}

cd .
# activate your conda environment
source /etc/network_turbo 2>/dev/null || true
export PYTHONUNBUFFERED=1
export HF_HOME=./cache
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HUB_DISABLE_XET=1

echo "=== Phase 1: v8 Standard Eval (500 batches) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_029999 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 500 --seed 42 \
  --output eval_results/eval_imm_v8_final.json

echo "=== Phase 2: v8 Bypass-ACIS Eval (500 batches) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/step_029999 \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 500 --seed 42 \
  --bypass_acis \
  --output eval_results/eval_imm_v8_bypass_final.json

echo "=== Phase 3: Gate Analysis (D024 revised) ==="
python3 analyze_gate.py \
  --baseline eval_results/eval_baseline_v5_final.json \
  --imm eval_results/eval_imm_v8_final.json \
  --bypass eval_results/eval_imm_v8_bypass_final.json \
  --output eval_results/gate_report_v8_final.json

echo "=== DONE ==="
cat eval_results/gate_report_v8_final.json | python3 -m json.tool
