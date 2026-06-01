"""Scan all checkpoints for mode collapse boundary."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mode_collapse import (
    load_config, load_dataset_batches, preprocess_batch,
    sample_with_different_noise
)
from train import load_smolvla_policy, get_tokenizer_from_policy

DEVICE = torch.device("cuda:0")
NUM_NOISE_SAMPLES = 16
NUM_OBS_TRIALS = 5


def test_checkpoint(policy, ckpt_path, batches, tokenizer):
    """Load checkpoint weights and measure noise diversity std."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    policy.eval()
    step = ckpt.get("step", "?")

    noise_stds = []
    for trial in range(min(NUM_OBS_TRIALS, len(batches))):
        batch = preprocess_batch(dict(batches[trial]), tokenizer, DEVICE)
        single_batch = {k: v[:1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        actions = sample_with_different_noise(policy, single_batch, NUM_NOISE_SAMPLES)
        std_per_dim = actions.std(dim=0).mean().item()
        noise_stds.append(std_per_dim)

    avg_std = np.mean(noise_stds)
    return step, avg_std


def scan_checkpoints(config_path, checkpoint_dir, label):
    """Scan all checkpoints in a directory."""
    config = load_config(config_path)
    policy = load_smolvla_policy(config, DEVICE)
    tokenizer = get_tokenizer_from_policy(policy)

    print(f"\nLoading dataset...", flush=True)
    batches = load_dataset_batches(config, num_batches=NUM_OBS_TRIALS, batch_size=8)
    print(f"Dataset loaded ({len(batches)} batches)", flush=True)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_files = sorted(ckpt_dir.glob("step_*/checkpoint.pt"))

    print(f"\n{'='*60}")
    print(f"Scanning {label}: {len(ckpt_files)} checkpoints")
    print(f"{'='*60}", flush=True)

    results = []
    for ckpt_path in ckpt_files:
        step, avg_std = test_checkpoint(policy, str(ckpt_path), batches, tokenizer)
        verdict = "PASS" if avg_std >= 0.05 else ("WARN" if avg_std >= 0.01 else "FAIL")
        step_name = ckpt_path.parent.name
        print(f"  {label} {step_name}: std={avg_std:.6f} [{verdict}]", flush=True)
        results.append((step_name, avg_std, verdict))

    return results


if __name__ == "__main__":
    print("="*60)
    print("=== cuda:1 Early Checkpoints ===")
    print("="*60)

    # Scan baseline_v6_alpha16 first (this is the alpha=64 variant based on task context)
    r2 = scan_checkpoints(
        "configs/baseline_v6_alpha16.yaml",
        "checkpoints/baseline_v6_alpha16",
        "v6_alpha16"
    )

    # Free memory
    torch.cuda.empty_cache()

    # Scan baseline
    r1 = scan_checkpoints(
        "configs/baseline.yaml",
        "checkpoints/baseline",
        "baseline"
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, results in [("baseline", r1), ("v6_alpha16", r2)]:
        print(f"\n--- {label} ---")
        for step_name, std, verdict in results:
            print(f"  {step_name}: std={std:.6f} [{verdict}]")

        # Find collapse boundary
        for i in range(1, len(results)):
            prev_std = results[i-1][1]
            curr_std = results[i][1]
            if prev_std >= 0.01 and curr_std < 0.01:
                print(f"  Collapse boundary: {results[i-1][0]} (std={prev_std:.4f}) -> {results[i][0]} (std={curr_std:.4f})")
                break
        else:
            if all(r[1] < 0.01 for r in results):
                print(f"  All checkpoints collapsed!")
            elif all(r[1] >= 0.01 for r in results):
                print(f"  No collapse detected in any checkpoint")
