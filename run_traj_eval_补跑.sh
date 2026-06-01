#!/bin/bash
set -e
cd .
# activate your conda environment
source /etc/network_turbo 2>/dev/null || true
export PYTHONUNBUFFERED=1
export HF_HOME=./cache
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export CUDA_VISIBLE_DEVICES=1

for STEP in 022000 024000 026000; do
  echo "$(date) === Standard eval step_${STEP} ==="
  python3 evaluate.py \
    --checkpoint checkpoints/imm_anchor_v8/step_${STEP} \
    --config configs/imm_anchor_v8.yaml \
    --gpu 0 --max_batches 100 --seed 42 \
    --output eval_results/trajectory/v8_step_${STEP}_standard.json

  echo "$(date) === Bypass eval step_${STEP} ==="
  python3 evaluate.py \
    --checkpoint checkpoints/imm_anchor_v8/step_${STEP} \
    --config configs/imm_anchor_v8.yaml \
    --gpu 0 --max_batches 100 --seed 42 \
    --bypass_acis \
    --output eval_results/trajectory/v8_step_${STEP}_bypass.json

  echo "$(date) === Done step_${STEP} ==="
done
echo "$(date) === ALL TRAJECTORY EVALS COMPLETE ==="
