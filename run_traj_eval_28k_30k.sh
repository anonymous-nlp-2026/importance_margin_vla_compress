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

# Wait for 28K checkpoint
echo "$(date) Waiting for step_028000..."
while [ ! -d checkpoints/imm_anchor_v8/step_028000 ]; do sleep 10; done
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

# Wait for 30K checkpoint (final)
echo "$(date) Waiting for step_030000 or final..."
while true; do
  if [ -d checkpoints/imm_anchor_v8/step_030000 ]; then
    FINAL_STEP=step_030000
    break
  elif [ -d checkpoints/imm_anchor_v8/step_029999 ]; then
    FINAL_STEP=step_029999
    break
  fi
  # Check if training is done (exit file exists)
  if [ -f /tmp/agent-ml-exit/mvp_imm_anchor_v8.exit ]; then
    # Training finished, find the latest checkpoint
    FINAL_STEP=$(ls -1d checkpoints/imm_anchor_v8/step_* | sort | tail -1 | xargs basename)
    break
  fi
  sleep 10
done

echo "$(date) === Trajectory eval ${FINAL_STEP} ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/${FINAL_STEP} \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 100 --seed 42 \
  --output eval_results/trajectory/v8_${FINAL_STEP}_standard.json

echo "$(date) === Bypass eval ${FINAL_STEP} ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/${FINAL_STEP} \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 100 --seed 42 \
  --bypass_acis \
  --output eval_results/trajectory/v8_${FINAL_STEP}_bypass.json

echo "$(date) === FULL 500-batch eval ${FINAL_STEP} ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/${FINAL_STEP} \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 500 --seed 42 \
  --output eval_results/eval_imm_v8_final.json

echo "$(date) === FULL 500-batch bypass eval ${FINAL_STEP} ==="
python3 evaluate.py \
  --checkpoint checkpoints/imm_anchor_v8/${FINAL_STEP} \
  --config configs/imm_anchor_v8.yaml \
  --gpu 0 --max_batches 500 --seed 42 \
  --bypass_acis \
  --output eval_results/eval_imm_v8_bypass_final.json

echo "$(date) === Gate Analysis ==="
python3 analyze_gate.py \
  --baseline eval_results/eval_baseline_v5_final.json \
  --imm eval_results/eval_imm_v8_final.json \
  --bypass eval_results/eval_imm_v8_bypass_final.json \
  --output eval_results/gate_report_v8_final.json

echo "$(date) === ALL DONE ==="
