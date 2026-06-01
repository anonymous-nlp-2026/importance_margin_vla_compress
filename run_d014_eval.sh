#!/bin/bash
# D014 Final Gate Evaluation Protocol
# v7 final (500 batches), baseline final (500 batches),
# baseline mid-checkpoints (50 batches each)

set -e
# activate your conda environment
cd .

CKPT_BASE="checkpoints/baseline"
CKPT_V7="checkpoints/imm_anchor"
CFG_BASE="configs/baseline.yaml"
CFG_V7="configs/imm_anchor.yaml"
GPU=${1:-0}

mkdir -p results

# Determine v7 final checkpoint: prefer latest/, fall back to highest step
V7_FINAL="${CKPT_V7}/latest/checkpoint.pt"
if [ ! -f "$V7_FINAL" ]; then
    V7_FINAL=$(ls -d ${CKPT_V7}/step_*/checkpoint.pt | sort -t_ -k2 -n | tail -1)
fi
echo "v7 final checkpoint: $V7_FINAL"

# Baseline final checkpoint
BL_FINAL="${CKPT_BASE}/step_029999/checkpoint.pt"
if [ ! -f "$BL_FINAL" ]; then
    BL_FINAL="${CKPT_BASE}/latest/checkpoint.pt"
fi
echo "baseline final checkpoint: $BL_FINAL"

# Phase 1: v7 final eval (500 batches)
echo "=== Phase 1: v7 final eval (500 batches) ==="
python evaluate.py \
    --checkpoint "$V7_FINAL" \
    --config "$CFG_V7" \
    --gpu "$GPU" \
    --max_batches 500 \
    --output results/eval_imm_v7_final.json

# Phase 2: baseline final eval (500 batches)
echo "=== Phase 2: baseline final eval (500 batches) ==="
python evaluate.py \
    --checkpoint "$BL_FINAL" \
    --config "$CFG_BASE" \
    --gpu "$GPU" \
    --max_batches 500 \
    --output results/eval_baseline_v5_final.json

# Phase 3a: baseline step_005000 (50 batches)
echo "=== Phase 3a: baseline step_005000 (50 batches) ==="
python evaluate.py \
    --checkpoint "${CKPT_BASE}/step_005000/checkpoint.pt" \
    --config "$CFG_BASE" \
    --gpu "$GPU" \
    --max_batches 50 \
    --output results/eval_baseline_v5_step005000.json

# Phase 3b: baseline step_010000 (50 batches)
echo "=== Phase 3b: baseline step_010000 (50 batches) ==="
python evaluate.py \
    --checkpoint "${CKPT_BASE}/step_010000/checkpoint.pt" \
    --config "$CFG_BASE" \
    --gpu "$GPU" \
    --max_batches 50 \
    --output results/eval_baseline_v5_step010000.json

echo "=== D014 eval complete ==="
echo "Results:"
ls -la results/eval_*.json
