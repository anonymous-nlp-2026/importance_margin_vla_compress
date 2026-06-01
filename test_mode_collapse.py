"""
test_mode_collapse.py — Quick action diversity check for flow matching ODE.

Detects mode collapse by measuring action std across batches of observations.
Two tests:
  1. Cross-obs: different observations → different actions? (std across batch)
  2. Same-obs: same observation, different noise seeds → different actions? (std across samples)

Usage:
    python test_mode_collapse.py --checkpoint_path checkpoints/baseline_v6_alpha16/latest/checkpoint.pt \
        --config_path configs/baseline_v6_alpha16.yaml
    python test_mode_collapse.py --checkpoint_path checkpoints/baseline/latest/checkpoint.pt \
        --config_path configs/baseline.yaml --pretrained_path ./cache/lerobot/smolvla_base
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

ARTIFACTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACTS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Quick mode collapse diagnostic for flow matching ODE"
    )
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to checkpoint.pt")
    p.add_argument("--config_path", type=str, required=True,
                   help="Path to YAML config")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_batches", type=int, default=10,
                   help="Number of dataset batches for cross-obs test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_noise_samples", type=int, default=16,
                   help="Number of noise samples per obs for same-obs test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Pretrained model path (for comparison baseline)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading (same as train.py)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    import copy
    import yaml
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        for k, v in config.items():
            if k in base_config and isinstance(base_config[k], dict) and isinstance(v, dict):
                base_config[k].update(v)
            else:
                base_config[k] = v
        return base_config
    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    from train import load_smolvla_policy
    config = load_config(config_path)
    policy = load_smolvla_policy(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    policy.eval()
    log.info("Loaded checkpoint: %s (step %s)", checkpoint_path, ckpt.get("step", "?"))
    return policy, config


def load_pretrained(pretrained_path: str, config_path: str, device: torch.device):
    """Load pretrained model without LoRA checkpoint (as reference)."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    config = load_config(config_path)
    policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    from train import _adapt_projections
    features = config.get("dataset", {}).get("features", {})
    if features:
        _adapt_projections(policy, features)
    policy.to(device)
    policy.eval()
    log.info("Loaded pretrained: %s", pretrained_path)
    return policy, config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_batches(config, num_batches, batch_size):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from torch.utils.data import DataLoader

    ds_cfg = config["dataset"]
    repo_id = ds_cfg["repo_id"]
    chunk_size = 50

    meta = LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {"action": [i / meta.fps for i in range(chunk_size)]}

    dataset = LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=delta_timestamps,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )

    batches = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# Preprocess batch (tokenize task text)
# ---------------------------------------------------------------------------

def preprocess_batch(batch, tokenizer, device):
    from train import preprocess_batch as _preprocess
    return _preprocess(batch, tokenizer, device=device)


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_actions_from_batch(policy, batch):
    """Returns actions of shape (B, chunk_size, action_dim)."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    actions = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state
    )
    action_dim = policy.config.action_feature.shape[0]
    return actions[:, :, :action_dim]  # (B, chunk_size, action_dim)


@torch.no_grad()
def sample_with_different_noise(policy, batch, num_samples):
    """Sample actions from the same obs with different noise seeds.
    Returns (num_samples, chunk_size, action_dim)."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    # Repeat single obs num_samples times
    def repeat_t(t, n):
        if isinstance(t, torch.Tensor):
            return t.expand(n, *(-1,) * (t.dim() - 1)).contiguous()
        if isinstance(t, list):
            return [x.expand(n, *(-1,) * (x.dim() - 1)).contiguous() for x in t]
        return t

    images_r = repeat_t(images, num_samples)
    img_masks_r = repeat_t(img_masks, num_samples)
    state_r = state.expand(num_samples, -1).contiguous()
    lang_tokens_r = lang_tokens.expand(num_samples, -1).contiguous()
    lang_masks_r = lang_masks.expand(num_samples, -1).contiguous()

    actions = policy.model.sample_actions(
        images_r, img_masks_r, lang_tokens_r, lang_masks_r, state_r
    )
    action_dim = policy.config.action_feature.shape[0]
    return actions[:, :, :action_dim]  # (num_samples, chunk_size, action_dim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_test(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info("Loading finetuned model...")
    policy, config = load_model(args.checkpoint_path, args.config_path, device)

    from train import get_tokenizer_from_policy
    tokenizer = get_tokenizer_from_policy(policy)

    log.info("Loading dataset batches...")
    batches = load_dataset_batches(config, args.num_batches, args.batch_size)
    log.info("Loaded %d batches of size %d", len(batches), args.batch_size)

    # ---- Test 1: Cross-obs diversity (different obs → different actions?) ----
    log.info("")
    log.info("=" * 60)
    log.info("TEST 1: Cross-observation action diversity")
    log.info("=" * 60)

    cross_stds = []
    for i, batch in enumerate(batches):
        batch = preprocess_batch(dict(batch), tokenizer, device)
        actions = sample_actions_from_batch(policy, batch)  # (B, chunk, action_dim)
        std_per_dim = actions.std(dim=0).mean()  # std across batch, then average
        cross_stds.append(std_per_dim.item())
        log.info("  Batch %2d: action std (cross-obs) = %.6f", i + 1, std_per_dim.item())

    avg_cross_std = np.mean(cross_stds)
    log.info("  Average cross-obs std: %.6f", avg_cross_std)

    # ---- Test 2: Same-obs noise diversity (same obs, different noise → same action?) ----
    log.info("")
    log.info("=" * 60)
    log.info("TEST 2: Same-observation noise diversity")
    log.info("=" * 60)

    first_batch = preprocess_batch(dict(batches[0]), tokenizer, device)
    # Use first sample from batch
    single_batch = {}
    for k, v in first_batch.items():
        if isinstance(v, torch.Tensor):
            single_batch[k] = v[:1]

    noise_stds = []
    for trial in range(min(5, len(batches))):
        if trial > 0:
            tb = preprocess_batch(dict(batches[trial]), tokenizer, device)
            single_batch = {k: v[:1] for k, v in tb.items() if isinstance(v, torch.Tensor)}

        actions = sample_with_different_noise(
            policy, single_batch, args.num_noise_samples
        )  # (N, chunk, action_dim)
        std_per_dim = actions.std(dim=0).mean()
        noise_stds.append(std_per_dim.item())
        log.info("  Obs %d: action std (noise diversity) = %.6f", trial + 1, std_per_dim.item())

    avg_noise_std = np.mean(noise_stds)
    log.info("  Average noise diversity std: %.6f", avg_noise_std)

    # ---- Optional: pretrained reference ----
    pretrained_noise_std = None
    if args.pretrained_path:
        log.info("")
        log.info("=" * 60)
        log.info("REFERENCE: Pretrained model noise diversity")
        log.info("=" * 60)

        policy_pt, _ = load_pretrained(args.pretrained_path, args.config_path, device)
        tokenizer_pt = get_tokenizer_from_policy(policy_pt)

        ref_batch = preprocess_batch(dict(batches[0]), tokenizer_pt, device)
        ref_single = {k: v[:1] for k, v in ref_batch.items() if isinstance(v, torch.Tensor)}
        ref_actions = sample_with_different_noise(
            policy_pt, ref_single, args.num_noise_samples
        )
        pretrained_noise_std = ref_actions.std(dim=0).mean().item()
        log.info("  Pretrained noise diversity std: %.6f", pretrained_noise_std)

        del policy_pt
        torch.cuda.empty_cache()

    # ---- Verdict ----
    log.info("")
    log.info("=" * 60)
    log.info("VERDICT")
    log.info("=" * 60)
    log.info("  Cross-obs std:     %.6f", avg_cross_std)
    log.info("  Noise diversity:   %.6f", avg_noise_std)
    if pretrained_noise_std is not None:
        log.info("  Pretrained ref:    %.6f", pretrained_noise_std)
        ratio = avg_noise_std / pretrained_noise_std if pretrained_noise_std > 1e-8 else 0.0
        log.info("  Ratio (ft/pt):     %.4f", ratio)

    if avg_noise_std < 0.01:
        log.info("  >>> FAIL: mode collapse detected (noise diversity std < 0.01)")
        verdict = "FAIL"
    elif avg_noise_std < 0.05:
        log.info("  >>> WARN: low action diversity (noise diversity std < 0.05)")
        verdict = "WARN"
    else:
        log.info("  >>> PASS: action diversity restored (noise diversity std > 0.05)")
        verdict = "PASS"

    return {
        "verdict": verdict,
        "cross_obs_std": round(avg_cross_std, 6),
        "noise_diversity_std": round(avg_noise_std, 6),
        "pretrained_noise_std": round(pretrained_noise_std, 6) if pretrained_noise_std else None,
        "checkpoint": args.checkpoint_path,
    }


if __name__ == "__main__":
    args = parse_args()
    result = run_test(args)
    log.info("Result: %s", result)
