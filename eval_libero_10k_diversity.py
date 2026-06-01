"""Diversity check for LIBERO 10K checkpoints on cuda:2."""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch

from test_mode_collapse import load_model, sample_with_different_noise
from train import get_tokenizer_from_policy, preprocess_batch


def create_libero_batch(seed=42):
    """Synthetic LIBERO observation: 2 cameras (256x256), 8D state."""
    torch.manual_seed(seed)
    return {
        "observation.images.image": torch.randn(1, 3, 256, 256),
        "observation.images.image2": torch.randn(1, 3, 256, 256),
        "observation.state": torch.randn(1, 8) * 0.1,
        "task": ["Pick up the red cup and place it on the plate.\n"],
    }


def test_diversity(ckpt_path, config_path, num_noise=16, num_obs=5):
    device = torch.device("cuda:0")
    
    policy, _ = load_model(str(ckpt_path), config_path, device)
    tokenizer = get_tokenizer_from_policy(policy)
    
    stds = []
    for i in range(num_obs):
        raw = create_libero_batch(seed=42 + i)
        batch = preprocess_batch(raw, tokenizer, device=device)
        actions = sample_with_different_noise(policy, batch, num_noise)
        std = actions.std(dim=0).mean().item()
        stds.append(std)
        print(f"  Obs {i}: std={std:.6f}")
    
    avg = np.mean(stds)
    verdict = "FAIL" if avg < 0.01 else ("WARN" if avg < 0.05 else "PASS")
    
    del policy
    torch.cuda.empty_cache()
    return avg, verdict


if __name__ == "__main__":
    configs = {
        "Full FT 10K": (
            PROJECT_DIR / "checkpoints/libero_full_ft/step_010000/checkpoint.pt",
            str(PROJECT_DIR / "configs/libero_full_ft.yaml"),
        ),
        "LoRA 10K": (
            PROJECT_DIR / "checkpoints/libero_baseline/step_010000/checkpoint.pt",
            str(PROJECT_DIR / "configs/libero_baseline.yaml"),
        ),
    }
    
    results = {}
    for name, (ckpt, cfg) in configs.items():
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        avg_std, verdict = test_diversity(ckpt, cfg)
        results[name] = {"diversity_std": avg_std, "verdict": verdict}
        print(f"  => diversity_std={avg_std:.6f} [{verdict}]")
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, r in results.items():
        print(f"  {name}: diversity_std={r['diversity_std']:.6f} [{r['verdict']}]")
