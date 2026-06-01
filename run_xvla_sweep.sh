#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/autodl-tmp/importance_margin_vla_compress
export MUJOCO_GL=egl

SUITE="libero_object"
NUM_EP=50

run_one() {
    local k=$1
    local gpu=$2
    echo "[GPU${gpu}] Starting k=${k} at $(date)"
    CUDA_VISIBLE_DEVICES=${gpu} python eval_xvla_pruning_libero.py \
        --keep_ratio ${k} \
        --num_episodes ${NUM_EP} \
        --gpu 0 \
        --suite ${SUITE} \
        --output eval_results/xvla_prune_k${k}_${SUITE}.json \
        2>&1 | tee eval_results/log_k${k}.txt
    echo "[GPU${gpu}] Finished k=${k} at $(date)"
}

mkdir -p eval_results

# Batch 1: 4 k values in parallel on 4 GPUs
echo "=== BATCH 1 START $(date) ==="
run_one 1.0  0 &
run_one 0.95 1 &
run_one 0.9  2 &
run_one 0.7  3 &
wait
echo "=== BATCH 1 DONE $(date) ==="

# Batch 2: 4 more k values
echo "=== BATCH 2 START $(date) ==="
run_one 0.5 0 &
run_one 0.3 1 &
run_one 0.2 2 &
run_one 0.1 3 &
wait
echo "=== BATCH 2 DONE $(date) ==="

echo "ALL SWEEPS COMPLETE at $(date)"

# Print summary
echo ""
echo "========== SUMMARY =========="
for k in 1.0 0.95 0.9 0.7 0.5 0.3 0.2 0.1; do
    f="eval_results/xvla_prune_k${k}_${SUITE}.json"
    if [ -f "$f" ]; then
        sr=$(python -c "import json; d=json.load(open('$f')); print(f'k={d[\"keep_ratio\"]:.2f}: SR={d[\"overall_success_rate\"]*100:.1f}% ({d[\"total_successes\"]}/{d[\"total_episodes\"]})')")
        echo "$sr"
    else
        echo "k=${k}: MISSING"
    fi
done
