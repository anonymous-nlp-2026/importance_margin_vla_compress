"""MF3 Baseline Control: fresh lr schedule on libero_full_ft checkpoint.

Control experiment to test whether Smart Compress 15pp gain comes from
re-encoding (compressor) or simply from a fresh lr schedule.

Setup:
  - Load libero_full_ft step_099999 checkpoint
  - Freeze SigLIP + connector + LLM backbone
  - Train only action_head/expert (same as Smart Compress)
  - No VisionTokenCompressor: 128 vision tokens go directly to LLM
  - lr=1e-4 cosine schedule, 100K steps, batch_size=4

Usage:
    python train_baseline_control.py --config configs/baseline_control.yaml
    python train_baseline_control.py --config configs/baseline_control.yaml --dry-run
"""

from __future__ import annotations

import faulthandler
faulthandler.enable()
import argparse
import copy
import json
import logging
import os
import signal
import sys
import time as time_module
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent


# ---- config loading ----

def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        base_config = deep_merge(base_config, config)
        return base_config
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- model loading ----

def load_smolvla_policy(config: Dict[str, Any], device: torch.device):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    model_cfg = config["model"]
    ds_cfg = config.get("dataset", {})

    pretrained_path = model_cfg.get("pretrained")
    if pretrained_path:
        policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    else:
        policy = SmolVLAPolicy(SmolVLAConfig())

    features = ds_cfg.get("features", {})
    state_dim = features.get("state_dim")
    action_dim = features.get("action_dim")
    cameras = features.get("cameras")

    if state_dim is not None:
        policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        )
    if action_dim is not None:
        policy.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(action_dim,)
        )
    if cameras:
        new_inputs = {}
        for key, feat in policy.config.input_features.items():
            if feat.type == FeatureType.VISUAL:
                continue
            new_inputs[key] = feat
        for cam_key in cameras:
            new_inputs[cam_key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=list(policy.config.resize_imgs_with_padding or [3, 256, 256]),
            )
        policy.config.input_features = new_inputs

    policy.to(device)
    return policy


def load_init_checkpoint(policy, ckpt_path: str, device: torch.device):
    """Load model weights only from a checkpoint (no optimizer/scheduler)."""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint.pt"

    log.info("Loading init checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    log.info("Checkpoint loaded (step %d)", ckpt.get("step", -1))
    if missing:
        log.warning("Missing keys: %s", missing[:10])
    if unexpected:
        log.warning("Unexpected keys: %s", unexpected[:10])

    return ckpt.get("step", -1)


def freeze_for_control(policy):
    """Freeze SigLIP + connector + LLM backbone. Train only action expert + projections."""
    expert_keywords = (
        "lm_expert", "state_proj", "action_in_proj",
        "action_out_proj", "action_time_mlp_in", "action_time_mlp_out",
    )

    for name, param in policy.named_parameters():
        param.requires_grad = any(kw in name for kw in expert_keywords)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    log.info(
        "Freeze config: trainable=%d / total=%d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )
    return trainable


# ---- dataset loading ----

def load_dataset(config: Dict[str, Any]):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    ds_cfg = config["dataset"]
    repo_id = ds_cfg["repo_id"]
    chunk_size = 50

    meta = LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {"action": [i / meta.fps for i in range(chunk_size)]}

    dataset = LeRobotDataset(
        repo_id=repo_id,
        episodes=ds_cfg.get("episodes", None),
        delta_timestamps=delta_timestamps,
    )
    return dataset


def make_dataloader(dataset, config: Dict[str, Any]):
    from torch.utils.data import DataLoader
    train_cfg = config["training"]
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )


def get_tokenizer_from_policy(policy):
    try:
        return policy.model.vlm_with_expert.processor.tokenizer
    except Exception:
        return None


def preprocess_batch(batch: Dict[str, Any], tokenizer, max_length: int = 48, device=None) -> Dict[str, Any]:
    tasks = batch.pop("task", None)
    if tasks is not None:
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = [t if t.endswith("\n") else t + "\n" for t in tasks]
        encoded = tokenizer(
            tasks,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = encoded["input_ids"]
        batch["observation.language.attention_mask"] = encoded["attention_mask"].bool()
    elif "observation.language.tokens" not in batch:
        B = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
        batch["observation.language.tokens"] = torch.ones(B, 1, dtype=torch.long)
        batch["observation.language.attention_mask"] = torch.ones(B, 1, dtype=torch.bool)

    for key in list(batch.keys()):
        if not isinstance(batch[key], torch.Tensor):
            del batch[key]

    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}

    return batch


# ---- optimizer + scheduler ----

def make_optimizer(model, config: Dict[str, Any]):
    train_cfg = config["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=train_cfg.get("lr", 1e-4),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=train_cfg.get("weight_decay", 1e-10),
    )


def make_scheduler(optimizer, config: Dict[str, Any]):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    train_cfg = config["training"]
    total_steps = train_cfg.get("total_steps", 100000)
    warmup_steps = train_cfg.get("warmup_steps", 1000)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=train_cfg.get("decay_lr", 2.5e-6),
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


# ---- checkpointing ----

def save_checkpoint(model, optimizer, scheduler, step, config, info, output_dir):
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / f"step_{step:06d}"
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        }
        if info:
            state["info"] = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in info.items()
            }
        torch.save(state, ckpt_dir / "checkpoint.pt")

        latest_link = output_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_dir.name)
        log.info("Checkpoint saved: %s (step %d)", ckpt_dir, step)
    except Exception as e:
        log.warning("Failed to save checkpoint at step %d: %s", step, e)


# ---- dry run ----

def dry_run(model, dataloader, device, config, tokenizer=None):
    log.info("=" * 60)
    log.info("DRY RUN: shape + gradient verification")
    log.info("=" * 60)

    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if tokenizer is not None:
        batch = preprocess_batch(batch, tokenizer, device=device)

    model.train()
    loss, loss_dict = model.forward(batch)
    log.info("Forward OK. Loss=%.4f", loss.item())

    loss.backward()

    log.info("--- Gradient Check ---")
    expert_grad = False
    frozen_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if any(kw in name for kw in ("lm_expert", "action_in_proj", "action_out_proj")):
                expert_grad = True
                if "lm_expert.layers.0" in name and "q_proj" in name:
                    log.info("  [EXPERT] %s: grad_norm=%.6f", name, grad_norm)
        elif param.requires_grad and param.grad is None:
            log.warning("  [NO GRAD] %s: requires_grad=True but grad is None!", name)
        elif not param.requires_grad and param.grad is not None:
            frozen_ok = False
            log.warning("  [FROZEN BUT HAS GRAD] %s", name)

    log.info("Expert receives gradient: %s", expert_grad)
    log.info("Frozen params have no gradient: %s", frozen_ok)

    log.info("--- Parameter Counts ---")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("  Total trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    log.info("--- Vision Tokens ---")
    log.info("  Vision tokens: 128 (no compression)")

    log.info("=" * 60)
    log.info("DRY RUN PASSED")
    log.info("=" * 60)

    model.zero_grad()


# ---- evaluation ----

@torch.no_grad()
def evaluate(model, dataloader, step, config, device, tokenizer=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    max_batches = config.get("evaluation", {}).get("max_eval_batches", 50)

    for batch in dataloader:
        if n_batches >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if tokenizer is not None:
            batch = preprocess_batch(batch, tokenizer, device=device)
        loss, _ = model.forward(batch)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return {"eval/action_loss": avg_loss}


# ---- graceful exit ----

class GracefulExit:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        log.info("Received signal %d, will exit after current step.", signum)
        self.should_exit = True


# ---- main ----

def parse_args():
    parser = argparse.ArgumentParser(description="MF3 Baseline Control: fresh lr on full_ft checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Resume training from control checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="smart-compress-vla")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run shape/gradient verification only")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    train_cfg = config["training"]
    model_cfg = config["model"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load base model
    log.info("Loading SmolVLA policy...")
    policy = load_smolvla_policy(config, device)

    # Load libero_full_ft checkpoint weights (model only, fresh optimizer/scheduler)
    init_ckpt = model_cfg.get("init_checkpoint")
    if init_ckpt:
        init_step = load_init_checkpoint(policy, init_ckpt, device)
        log.info("Initialized from checkpoint at step %d (optimizer/scheduler are fresh)", init_step)

    # Freeze everything except action expert
    freeze_for_control(policy)

    model = policy

    # Dataset
    log.info("Loading dataset...")
    dataset = load_dataset(config)
    dataloader = make_dataloader(dataset, config)

    tokenizer = get_tokenizer_from_policy(policy)

    # Dry run
    if args.dry_run:
        dry_run(model, dataloader, device, config, tokenizer=tokenizer)
        return

    # Fresh optimizer + scheduler
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)

    total_steps = args.max_steps or train_cfg.get("total_steps", 100000)
    log_every = train_cfg.get("log_every", 50)
    eval_every = train_cfg.get("eval_every", 5000)
    save_every = train_cfg.get("save_every", 10000)
    grad_clip = train_cfg.get("grad_clip", 10.0)
    output_dir = train_cfg.get("output_dir", "checkpoints/baseline_control")

    start_step = 0

    # Resume (from control checkpoint, not init)
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "checkpoint.pt"
        log.info("Resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        log.info("Resumed at step %d", start_step)

    # W&B
    import wandb
    run_name = args.wandb_run or "baseline_control_fresh_lr"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=config,
        resume="allow",
    )

    # Eval dataloader
    eval_dataloader = make_dataloader(dataset, {
        **config,
        "training": {**train_cfg, "batch_size": train_cfg.get("eval_batch_size", 1)},
    })

    # Training loop
    graceful = GracefulExit()
    model.train()
    data_iter = iter(dataloader)
    t0 = time_module.time()
    last_info = {}

    log.info("Starting training: %d -> %d steps", start_step, total_steps)

    for step in range(start_step, total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = preprocess_batch(batch, tokenizer, device=device)

        loss, loss_dict = model.forward(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        ).item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        last_info = {"action_loss": loss.detach()}

        if step % log_every == 0:
            action_l = loss.item()
            log_dict = {
                "train/action_loss": action_l,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
            }
            wandb.log(log_dict, step=step)

            elapsed = time_module.time() - t0
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            log.info(
                "step=%d/%d  loss=%.4f  lr=%.2e  grad=%.2f  sps=%.1f",
                step, total_steps, action_l,
                optimizer.param_groups[0]["lr"], grad_norm, sps,
            )

        if step > 0 and step % eval_every == 0:
            eval_metrics = evaluate(model, eval_dataloader, step, config, device, tokenizer=tokenizer)
            wandb.log(eval_metrics, step=step)
            for k, v in eval_metrics.items():
                log.info("  %s: %.4f", k, v)

        if step > 0 and step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, config, last_info, output_dir)

        if graceful.should_exit:
            log.info("Graceful exit at step %d", step)
            save_checkpoint(model, optimizer, scheduler, step, config, last_info, output_dir)
            break

    if not graceful.should_exit:
        save_checkpoint(model, optimizer, scheduler, total_steps - 1, config, last_info, output_dir)

    wandb.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
