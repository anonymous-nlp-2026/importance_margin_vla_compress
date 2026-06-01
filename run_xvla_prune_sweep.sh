#!/bin/bash
set -e

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./cache

# activate your conda environment
conda activate base

cd .

MODEL_PATH="./X-VLA-Libero"
SUITE="libero_object"
EPISODES=50
SEED=42
DOMAIN_ID=0

K_VALUES="1.0 0.95 0.9 0.7 0.5 0.3 0.2 0.1"

echo "===== X-VLA Pruning Sweep ====="
echo "Suite: $SUITE | Episodes/task: $EPISODES | Seed: $SEED"
echo "K values: $K_VALUES"
echo ""

for K in $K_VALUES; do
    echo ">>> Starting keep_ratio=$K at $(date)"
    python eval_xvla_pruning_libero.py \
        --model_path "$MODEL_PATH" \
        --suite "$SUITE" \
        --keep_ratio "$K" \
        --num_episodes "$EPISODES" \
        --seed "$SEED" \
        --domain_id "$DOMAIN_ID" \
        --gpu 0 \
        --output "eval_results/xvla_prune_k${K}_${SUITE}.json" \
        2>&1 | tee "eval_results/xvla_prune_k${K}_${SUITE}.log"
    echo ">>> Finished keep_ratio=$K at $(date)"
    echo ""
done

echo "===== All sweeps complete ====="
echo "SWEEP_DONE" > eval_results/xvla_sweep_sentinel.txt
