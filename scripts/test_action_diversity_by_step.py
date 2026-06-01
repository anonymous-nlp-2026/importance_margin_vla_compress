"""
Batch action diversity test across all baseline checkpoints.
Uses synthetic observations with correct format.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch

from test_mode_collapse import load_model, load_pretrained, load_config, sample_with_different_noise
from train import get_tokenizer_from_policy, preprocess_batch


def create_synthetic_batch(device, state_dim=14, img_size=224, seed=42):
    """Create a synthetic observation batch with correct structure for SmolVLA."""
    torch.manual_seed(seed)
    batch = {
        "observation.images.top": torch.randn(1, 3, img_size, img_size),
        "observation.state": torch.randn(1, state_dim) * 0.1,
        "task": ["Insert the peg into the hole.\n"],
    }
    return batch


def test_checkpoint_diversity(policy, tokenizer, device, num_noise_samples=16, num_obs=5):
    """Test noise diversity across multiple synthetic observations."""
    stds = []
    for obs_idx in range(num_obs):
        raw_batch = create_synthetic_batch(device, seed=42 + obs_idx)
        batch = preprocess_batch(raw_batch, tokenizer, device=device)
        actions = sample_with_different_noise(policy, batch, num_noise_samples)
        std = actions.std(dim=0).mean().item()
        stds.append(std)
        print(f"  Obs {obs_idx}: noise std = {std:.6f}")
    return stds


def main():
    device = torch.device("cuda:0")
    config_path = str(PROJECT_DIR / "configs" / "baseline.yaml")
    pretrained_path = "./cache/lerobot/smolvla_base"
    num_noise_samples = 16
    num_obs = 5
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    checkpoints = sorted(PROJECT_DIR.glob("checkpoints/baseline/step_*/checkpoint.pt"))
    print(f"Found {len(checkpoints)} checkpoints: {[str(c.parent.name) for c in checkpoints]}")
    
    results = {}
    
    # Test pretrained
    print("\n" + "="*60)
    print("PRETRAINED (reference)")
    print("="*60)
    policy_pt, _ = load_pretrained(pretrained_path, config_path, device)
    tokenizer_pt = get_tokenizer_from_policy(policy_pt)
    
    pt_stds = test_checkpoint_diversity(policy_pt, tokenizer_pt, device, num_noise_samples, num_obs)
    pretrained_avg = np.mean(pt_stds)
    print(f"  Average: {pretrained_avg:.6f}")
    results["pretrained"] = {"noise_diversity_std": round(pretrained_avg, 6), "per_obs": [round(s,6) for s in pt_stds]}
    
    del policy_pt
    torch.cuda.empty_cache()
    
    # Test each checkpoint
    for ckpt_path in checkpoints:
        step_name = ckpt_path.parent.name
        print(f"\n{'='*60}")
        print(f"Checkpoint: {step_name}")
        print("="*60)
        
        try:
            policy, _ = load_model(str(ckpt_path), config_path, device)
            tokenizer = get_tokenizer_from_policy(policy)
            
            ckpt_stds = test_checkpoint_diversity(policy, tokenizer, device, num_noise_samples, num_obs)
            avg_std = np.mean(ckpt_stds)
            ratio = avg_std / pretrained_avg if pretrained_avg > 1e-8 else 0.0
            verdict = "FAIL" if avg_std < 0.01 else ("WARN" if avg_std < 0.05 else "PASS")
            
            print(f"  Average: {avg_std:.6f} (ratio: {ratio:.4f}) [{verdict}]")
            results[step_name] = {
                "noise_diversity_std": round(avg_std, 6),
                "ratio_to_pretrained": round(ratio, 4),
                "verdict": verdict,
                "per_obs": [round(s,6) for s in ckpt_stds],
            }
            
            del policy
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[step_name] = {"error": str(e)}
    
    # Save
    out_dir = PROJECT_DIR / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "action_diversity_by_step.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Step':<15} {'Noise Std':<12} {'Ratio':<8} {'Verdict'}")
    print("-"*50)
    print(f"{'pretrained':<15} {pretrained_avg:<12.6f} {'1.0000':<8} REF")
    for ckpt_path in checkpoints:
        step_name = ckpt_path.parent.name
        r = results.get(step_name, {})
        if "error" in r:
            print(f"{step_name:<15} ERROR")
        else:
            print(f"{step_name:<15} {r['noise_diversity_std']:<12.6f} {r['ratio_to_pretrained']:<8.4f} {r['verdict']}")

if __name__ == "__main__":
    main()
