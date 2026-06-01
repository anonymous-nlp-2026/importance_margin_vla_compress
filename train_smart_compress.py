"""Smart Compress Training: VisionTokenCompressor + SmolVLA on LIBERO.

Usage:
    python train_smart_compress.py --config configs/smart_compress.yaml
    python train_smart_compress.py --config configs/smart_compress.yaml --dry-run
    python train_smart_compress.py --config configs/smart_compress.yaml --resume checkpoints/smart_compress_m32/latest

Trainable: VisionTokenCompressor + action expert + projections.
Frozen: SigLIP + connector + VLM text model.
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
sys.path.insert(0, str(ARTIFACTS_DIR))

from smart_compress_module import SmartCompressWrapper, VisionTokenCompressor


# ---- config loading (reused from train.py) ----

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
    """Load SmolVLAPolicy, adapt projections for target dataset."""
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


def wrap_with_compressor(policy, config: Dict[str, Any]) -> SmartCompressWrapper:
    compress_cfg = config.get("compressor", {})
    wrapper = SmartCompressWrapper(policy, compress_config=compress_cfg)
    return wrapper


def freeze_for_compress_training(wrapper: SmartCompressWrapper):
    """Freeze SigLIP + connector + VLM text model. Train compressor + expert + projections."""
    # freeze everything first
    for param in wrapper.policy.parameters():
        param.requires_grad = False

    # unfreeze action expert + projection layers
    expert_keywords = (
        "lm_expert", "state_proj", "action_in_proj",
        "action_out_proj", "action_time_mlp_in", "action_time_mlp_out",
    )
    for name, param in wrapper.policy.named_parameters():
        if any(kw in name for kw in expert_keywords):
            param.requires_grad = True

    # compressor is already trainable (nn.Module default)
    for param in wrapper.compressor.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    log.info(
        "Freeze config: trainable=%d / total=%d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )

    # breakdown
    comp_params = sum(p.numel() for p in wrapper.compressor.parameters())
    expert_params = sum(
        p.numel() for n, p in wrapper.policy.named_parameters()
        if p.requires_grad
    )
    log.info("  Compressor params: %d", comp_params)
    log.info("  Expert+projection trainable params: %d", expert_params)


# ---- dataset loading (reused from train.py) ----

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
        bm = SmartCompressWrapper._unwrap_to_vla(policy.model)
        return bm.vlm_with_expert.processor.tokenizer
    except Exception:
        return None




def preprocess_batch(batch: Dict[str, Any], tokenizer, max_length: int = 48, device=None) -> Dict[str, Any]:
    """Tokenize task strings and move batch to device."""
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
    from torch.optim.lr_scheduler import CosineAnnealingLR
    train_cfg = config["training"]
    total_steps = train_cfg.get("total_steps", 100000)
    warmup_steps = train_cfg.get("warmup_steps", 1000)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=train_cfg.get("decay_lr", 2.5e-6),
    )

    from torch.optim.lr_scheduler import LinearLR, SequentialLR
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
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "info": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in (info or {}).items()},
    }
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)

    latest_dir = output_dir.parent / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, latest_dir / "checkpoint.pt")
    log.info("Checkpoint saved: %s (step %d)", ckpt_path, step)


# ---- dry run ----

def dry_run(model, dataloader, device, config, tokenizer=None):
    """Verify shapes, gradient flow, and parameter counts."""
    log.info("=" * 60)
    log.info("DRY RUN: shape + gradient verification")
    log.info("=" * 60)

    model.eval()
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if tokenizer is not None:
        batch = preprocess_batch(batch, tokenizer, device=device)

    # forward
    model.train()
    loss, info = model.forward(batch, step=0)
    log.info("Forward OK. Loss=%.4f", loss.item())
    log.info("Info: %s", {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in info.items()})

    # backward
    loss.backward()

    # check gradients
    log.info("--- Gradient Check ---")
    comp_grad = False
    expert_grad = False
    vlm_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "compressor" in name:
                comp_grad = True
                if "queries" in name or "output_norm" in name:
                    log.info("  [COMPRESSOR] %s: grad_norm=%.6f", name, grad_norm)
            elif any(kw in name for kw in ("lm_expert", "action_in_proj", "action_out_proj")):
                expert_grad = True
        elif param.requires_grad and param.grad is None:
            log.warning("  [NO GRAD] %s: requires_grad=True but grad is None!", name)

    log.info("Compressor receives gradient: %s", comp_grad)
    log.info("Expert receives gradient: %s", expert_grad)

    # check frozen params have no grad
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if "vision_model" in name or "text_model" in name or "connector" in name:
                assert param.grad is None, f"Frozen param {name} has gradient!"

    vlm_frozen_ok = all(
        p.grad is None
        for n, p in model.named_parameters()
        if not p.requires_grad and ("vision_model" in n or "text_model" in n)
    )
    log.info("VLM+SigLIP frozen (no grad): %s", vlm_frozen_ok)

    # parameter counts
    log.info("--- Parameter Counts ---")
    comp_total = sum(p.numel() for p in model.compressor.parameters())
    comp_train = sum(p.numel() for p in model.compressor.parameters() if p.requires_grad)
    policy_train = sum(p.numel() for n, p in model.policy.named_parameters() if p.requires_grad)
    all_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_total = sum(p.numel() for p in model.parameters())

    log.info("  Compressor: %d params (all trainable)", comp_total)
    log.info("  Policy trainable (expert+proj): %d params", policy_train)
    log.info("  Total trainable: %d / %d (%.2f%%)", all_train, all_total, 100 * all_train / all_total)

    # compression info
    M = model.compressor.num_queries
    log.info("--- Compression ---")
    log.info("  Original vision tokens: 128")
    log.info("  Compressed tokens (M): %d", M)
    log.info("  Compression ratio: %.1fx (%.0f%% reduction)", 128 / M, 100 * (1 - M / 128))

    # action output shape
    with torch.no_grad():
        x_out = model.forward_inference(batch)
    log.info("--- Inference Output ---")
    log.info("  Action output shape: %s", list(x_out.shape))
    log.info("  Expected: [B=%d, chunk=%d, max_action_dim=%d]",
             batch["action"].shape[0],
             model.base_model.config.chunk_size,
             model.base_model.config.max_action_dim)

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
        loss, _ = model.forward(batch, step=step)
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
    parser = argparse.ArgumentParser(description="Smart Compress training for SmolVLA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
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

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # load model
    log.info("Loading SmolVLA policy...")
    policy = load_smolvla_policy(config, device)

    log.info("Wrapping with VisionTokenCompressor...")
    model = wrap_with_compressor(policy, config)
    model.to(device)

    freeze_for_compress_training(model)

    # dataset
    log.info("Loading dataset...")
    dataset = load_dataset(config)
    dataloader = make_dataloader(dataset, config)

    # tokenizer for batch preprocessing
    tokenizer = get_tokenizer_from_policy(policy)

    # dry run
    if args.dry_run:
        dry_run(model, dataloader, device, config, tokenizer=tokenizer)
        return

    # optimizer + scheduler
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)

    total_steps = args.max_steps or train_cfg.get("total_steps", 100000)
    log_every = train_cfg.get("log_every", 50)
    eval_every = train_cfg.get("eval_every", 5000)
    save_every = train_cfg.get("save_every", 10000)
    grad_clip = train_cfg.get("grad_clip", 10.0)
    output_dir = train_cfg.get("output_dir", "checkpoints/smart_compress")

    start_step = 0

    # resume
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
    compress_cfg = config.get("compressor", {})
    run_name = args.wandb_run or f"smart_compress_m{compress_cfg.get('num_queries', 32)}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=config,
        resume="allow",
    )

    # eval dataloader
    eval_dataloader = make_dataloader(dataset, {
        **config,
        "training": {**train_cfg, "batch_size": train_cfg.get("eval_batch_size", 1)},
    })

    # training loop
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

        loss, info = model.forward(batch, step=step)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        ).item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        last_info = info

        if step % log_every == 0:
            action_l = info["action_loss"].item() if isinstance(info["action_loss"], torch.Tensor) else info["action_loss"]
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
