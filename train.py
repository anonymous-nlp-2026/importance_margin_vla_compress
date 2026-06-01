"""IMM Training Pipeline for SmolVLA + LIBERO.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/imm_anchor.yaml
    python train.py --config configs/imm_anchor.yaml --resume checkpoints/imm_anchor/latest

Two modes:
  - Baseline: L_action only (flow matching MSE), no ACIS/margin loss.
  - IMM: L_action + lambda * L_margin with ACIS scoring + token selection + warmup.

Depends: lerobot, torch, wandb, peft.
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
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import wandb
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="IMM training for SmolVLA on LIBERO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (use CUDA_VISIBLE_DEVICES for physical mapping)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="imm-vla-compress", help="W&B project name")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name (auto-generated if None)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total_steps (for dry-run)")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config with _base_ inheritance support."""
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
    """Recursively merge override into base."""
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


def _adapt_projections(policy, features: Dict[str, Any]):
    """Update SmolVLAConfig features for target dataset dimensions."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    state_dim = features.get("state_dim")
    action_dim = features.get("action_dim")
    cameras = features.get("cameras")

    if state_dim is not None:
        old_shape = None
        if policy.config.input_features and "observation.state" in policy.config.input_features:
            old_shape = policy.config.input_features["observation.state"].shape
        policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        )
        log.info("Updated state feature shape: %s -> (%d,)", old_shape, state_dim)

    if action_dim is not None:
        old_shape = None
        if policy.config.output_features and "action" in policy.config.output_features:
            old_shape = policy.config.output_features["action"].shape
        policy.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(action_dim,)
        )
        log.info("Updated action feature shape: %s -> (%d,)", old_shape, action_dim)

    if cameras is not None:
        image_shape = tuple(features.get("image_shape", (3, 256, 256)))
        old_cam_keys = [
            k for k, v in policy.config.input_features.items()
            if v.type == FeatureType.VISUAL
        ]
        for k in old_cam_keys:
            del policy.config.input_features[k]
        for cam_name in cameras:
            policy.config.input_features[cam_name] = PolicyFeature(
                type=FeatureType.VISUAL, shape=image_shape
            )
        log.info("Updated cameras: %s -> %s (shape=%s)", old_cam_keys, cameras, image_shape)


def load_smolvla_policy(config: Dict[str, Any], device: torch.device):
    """Load SmolVLAPolicy via LeRobot API, adapt to dataset, and apply LoRA."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_cfg = config["model"]
    ds_cfg = config.get("dataset", {})

    pretrained_path = model_cfg.get("pretrained")
    if pretrained_path:
        policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    else:
        policy_config = SmolVLAConfig()
        policy = SmolVLAPolicy(policy_config)

    features = ds_cfg.get("features", {})
    if features:
        _adapt_projections(policy, features)

    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("rank", 0) > 0:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg.get("alpha", lora_cfg["rank"] * 2),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get(
                "target_modules_regex",
                r"(vlm_with_expert\.lm_expert\..*\.(q|v)_proj"
                r"|(state_proj|action_in_proj|action_out_proj"
                r"|action_time_mlp_in|action_time_mlp_out))",
            ),
            bias="none",
        )
        policy.model = get_peft_model(policy.model, peft_config)
        log.info(
            "LoRA applied: rank=%d, alpha=%d, trainable params: %s",
            lora_cfg["rank"],
            peft_config.lora_alpha,
            sum(p.numel() for p in policy.model.parameters() if p.requires_grad),
        )

    # Full expert FT: freeze VLM+vision, train action expert + projections
    train_expert_only = model_cfg.get("train_expert_only", False)
    if train_expert_only and lora_cfg.get("rank", 0) == 0:
        expert_keywords = (
            "lm_expert", "state_proj", "action_in_proj",
            "action_out_proj", "action_time_mlp_in", "action_time_mlp_out",
        )
        for name, param in policy.named_parameters():
            param.requires_grad = any(kw in name for kw in expert_keywords)
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy.parameters())
        log.info(
            "Full expert FT: trainable=%d/%d (%.1f%%)",
            trainable, total, 100 * trainable / total,
        )

    policy.to(device)
    return policy


def load_dataset(config: Dict[str, Any]):
    """Load dataset via LeRobot API with delta_timestamps for action chunking."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    ds_cfg = config["dataset"]
    repo_id = ds_cfg["repo_id"]
    chunk_size = 50  # SmolVLA default

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
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )


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
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    total_steps = train_cfg.get("total_steps", 30000)
    decay_lr = train_cfg.get("decay_lr", 2.5e-6)

    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=decay_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def get_tokenizer_from_policy(policy):
    """Extract tokenizer from SmolVLAPolicy, handling PeftModel wrapping."""
    model = policy.model
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model
    except ImportError:
        pass
    return model.vlm_with_expert.processor.tokenizer


def preprocess_batch(batch: Dict[str, Any], tokenizer, max_length: int = 48, device=None) -> Dict[str, Any]:
    """Tokenize task strings and move batch to device for SmolVLA."""
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



def baseline_forward(policy, batch):
    """Baseline forward: flow matching loss without action_is_pad masking.

    Bypasses SmolVLAPolicy.forward() which dilutes loss via pad masking.
    Uses the same approach as IMMSmolVLAWrapper.forward() for consistency.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)

    losses = policy.model.forward(
        images, img_masks, lang_tokens, lang_masks, state, actions, None, None
    )

    action_dim = policy.config.action_feature.shape[0]
    loss = losses[:, :, :action_dim].mean()
    return loss, {"action_loss": loss.detach()}




def compute_k_ratio(step: int, config: Dict[str, Any]) -> Optional[float]:
    """Compute dynamic k_ratio based on annealing schedule in config."""
    imm_cfg = config.get("imm", {})
    schedule = imm_cfg.get("k_ratio_schedule")
    if schedule is None:
        return None

    sched_type = schedule.get("type", "linear")
    if sched_type == "piecewise":
        breakpoints = schedule["breakpoints"]
        ratio = breakpoints[0][1]
        for bp_step, bp_ratio in breakpoints:
            if step >= bp_step:
                ratio = bp_ratio
        return ratio

    start = schedule["start"]
    end = schedule["end"]
    warmup_steps = schedule.get("warmup_steps", 0)
    anneal_steps = schedule.get("anneal_steps", 20000)

    if step < warmup_steps:
        return start

    anneal_progress = min(1.0, (step - warmup_steps) / anneal_steps)

    if sched_type == "cosine":
        import math as _math
        ratio = start + (end - start) * (1 - _math.cos(_math.pi * anneal_progress)) / 2
    else:  # linear
        ratio = start + (end - start) * anneal_progress

    return ratio


def wrap_with_imm(policy, config: Dict[str, Any]):
    """Wrap SmolVLAPolicy.model with IMMSmolVLAWrapper."""
    sys.path.insert(0, str(ARTIFACTS_DIR))
    from imm.smolvla_wrapper import IMMSmolVLAWrapper

    imm_cfg = config["imm"]
    acis_cfg = config.get("acis", {})

    imm_kwargs = {
        "lambda_margin": imm_cfg.get("lambda_margin", 0.1),
        "delta": imm_cfg.get("delta", 1.0),
        "warmup_steps": imm_cfg.get("warmup_steps", 1000),
        "temperature": imm_cfg.get("temperature", 1.0),
        "method": imm_cfg.get("soft_topk_method", "sigmoid"),
        "k_ratio": imm_cfg.get("k_ratio", 0.5),
        "gradient_isolation": imm_cfg.get("gradient_isolation", False),
    }

    wrapper = IMMSmolVLAWrapper(
        smolvla_policy=policy,
        acis_config=acis_cfg,
        imm_config=imm_kwargs,
    )
    return wrapper


def setup_twostage(model, config, device):
    """Load baseline LoRA weights and freeze everything except ACIS for two-stage Stage 1."""
    ts_cfg = config["twostage"]
    baseline_ckpt = ts_cfg.get("baseline_checkpoint", "checkpoints/baseline/step_029999/checkpoint.pt")

    ckpt_path = Path(baseline_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ARTIFACTS_DIR / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.policy.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded baseline LoRA weights from %s (step %d)", ckpt_path, ckpt.get("step", -1))

    acis_param_ids = {id(p) for p in model.acis.parameters()}
    n_frozen = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        if id(param) in acis_param_ids:
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
            n_frozen += param.numel()

    log.info("Two-stage Stage 1: frozen=%d, trainable(ACIS)=%d", n_frozen, n_trainable)


def setup_stage2_recovery(model, config, device):
    s2_cfg = config["stage2_recovery"]
    ckpt_path = Path(s2_cfg["checkpoint"])
    if not ckpt_path.is_absolute():
        ckpt_path = ARTIFACTS_DIR / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded stage1 checkpoint from %s (step %d)", ckpt_path, ckpt.get("step", -1))

    acis_param_ids = {id(p) for p in model.acis.parameters()}
    n_frozen = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        if id(param) in acis_param_ids:
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            param.requires_grad = True
            n_trainable += param.numel()

    log.info("Stage2 recovery: frozen(ACIS)=%d, trainable(expert)=%d", n_frozen, n_trainable)



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    config: Dict[str, Any],
    info: Optional[Dict] = None,
    output_dir: Optional[str] = None,
):
    """Save checkpoint. Non-fatal on failure."""
    if output_dir is None:
        output_dir = config["training"].get("output_dir", "checkpoints")
    ckpt_dir = Path(output_dir) / f"step_{step:06d}"
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
            state["margin_stats"] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in info.items()
                if k in ("margin", "s_k", "s_k_plus_1", "lambda", "action_loss", "margin_loss")
            }

        torch.save(state, ckpt_dir / "checkpoint.pt")

        latest_link = Path(output_dir) / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_dir.name)

        log.info("Checkpoint saved: %s", ckpt_dir)
    except Exception as e:
        log.warning("Failed to save checkpoint at step %d: %s", step, e)


def load_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    device: torch.device,
) -> int:
    """Load checkpoint and return the step to resume from."""
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint.pt"

    log.info("Resuming from %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])

    start_step = state["step"] + 1
    log.info("Resumed at step %d", start_step)
    return start_step


@torch.no_grad()
def evaluate(model, dataloader, step: int, config: Dict[str, Any], device: torch.device, imm_enabled: bool, tokenizer=None):
    """Run validation pass and return metrics dict."""
    sys.path.insert(0, str(ARTIFACTS_DIR))
    from imm.metrics import compute_margin, margin_statistics, topk_preservation_rate, l_eps_delta_ratio

    model.eval()
    eval_cfg = config.get("evaluation", {})
    epsilons = eval_cfg.get("perturbations", [0.0, 0.01, 0.02, 0.05])
    pruning_ratio = eval_cfg.get("pruning_ratio", 0.5)
    max_batches = eval_cfg.get("max_eval_batches", 50)

    results = {}
    total_action_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        if tokenizer is not None:
            batch = preprocess_batch(batch, tokenizer, device=device)
        else:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if imm_enabled:
            _, info = model(batch, step=step)
            total_action_loss += info["action_loss"].item()

            scores = info["importance_scores"]
            k = info["k"]

            for eps in epsilons:
                key = f"eps_{eps}"
                if key not in results:
                    results[key] = {"margins": [], "preservation": []}

                if eps > 0:
                    batch_noisy = {k_: v.clone() if isinstance(v, torch.Tensor) else v for k_, v in batch.items()}
                    for img_key in [k_ for k_ in batch_noisy if "image" in k_.lower()]:
                        if isinstance(batch_noisy[img_key], torch.Tensor) and batch_noisy[img_key].is_floating_point():
                            batch_noisy[img_key] = (batch_noisy[img_key] + eps * torch.randn_like(batch_noisy[img_key])).clamp(0, 1)
                    _, info_noisy = model(batch_noisy, step=step)
                    scores_noisy = info_noisy["importance_scores"]
                else:
                    scores_noisy = scores

                margin = compute_margin(scores, k)
                pres = topk_preservation_rate(scores, scores_noisy, k)
                results[key]["margins"].append(margin.cpu())
                results[key]["preservation"].append(pres.cpu())
        else:
            loss, info = model.forward_baseline(batch) if hasattr(model, "forward_baseline") else (model(batch), {})
            if isinstance(loss, tuple):
                loss = loss[0]
            total_action_loss += (info.get("action_loss", loss) if isinstance(info, dict) else loss).item()

        n_batches += 1

    metrics = {"eval/action_loss": total_action_loss / max(n_batches, 1)}

    if imm_enabled:
        for eps_key, data in results.items():
            margins = torch.cat(data["margins"])
            pres = torch.cat(data["preservation"])
            metrics[f"eval/{eps_key}/margin_mean"] = margins.mean().item()
            metrics[f"eval/{eps_key}/margin_median"] = margins.median().item()
            metrics[f"eval/{eps_key}/margin_std"] = margins.std().item()
            metrics[f"eval/{eps_key}/preservation_rate"] = pres.mean().item()

    model.train()
    return metrics


class GracefulExit:
    """Handle SIGINT/SIGTERM for graceful checkpoint saving."""

    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        log.info("Received signal %d, will exit after current step", signum)
        self.should_exit = True


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_cfg = config["training"]
    total_steps = train_cfg.get("total_steps", 30000)
    eval_every = train_cfg.get("eval_every", 2000)
    save_every = train_cfg.get("save_every", 5000)
    log_every = train_cfg.get("log_every", 50)
    grad_clip = train_cfg.get("grad_clip", 10.0)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    lip_monitor_every = train_cfg.get("lip_monitor_every", 500)

    imm_enabled = config.get("imm", {}).get("enabled", False)

    output_dir = train_cfg.get("output_dir", f"checkpoints/{'imm' if imm_enabled else 'baseline'}")
    config["training"]["output_dir"] = output_dir

    # Model
    log.info("Loading SmolVLA policy...")
    policy = load_smolvla_policy(config, device)

    tokenizer = get_tokenizer_from_policy(policy)

    if imm_enabled:
        log.info("Wrapping with IMMSmolVLAWrapper (delta=%.2f, lambda=%.2f)",
                 config["imm"].get("delta", 1.0), config["imm"].get("lambda_margin", 0.1))
        model = wrap_with_imm(policy, config)
        model.to(device)
    else:
        model = policy

    # Two-stage setup (v8b)
    is_twostage = imm_enabled and "twostage" in config
    if is_twostage:
        ts_cfg = config["twostage"]
        setup_twostage(model, config, device)
        total_steps = ts_cfg["stage1_steps"]
        config = copy.deepcopy(config)
        config["training"].update({
            "lr": ts_cfg.get("stage1_lr", 3e-4),
            "warmup_steps": ts_cfg.get("stage1_warmup_steps", 500),
            "total_steps": total_steps,
        })
        log.info("Two-stage mode: Stage 1 for %d steps, lr=%.2e", total_steps, config["training"]["lr"])

    # Stage2 recovery mode
    is_stage2_recovery = imm_enabled and "stage2_recovery" in config and not is_twostage
    if is_stage2_recovery:
        setup_stage2_recovery(model, config, device)
        log.info("Stage2 recovery mode: lr=%.2e, total_steps=%d", config["training"]["lr"], total_steps)

    # --max_steps override
    if args.max_steps is not None:
        total_steps = args.max_steps
        config["training"]["total_steps"] = total_steps
        log.info("Overriding total_steps to %d (dry-run)", total_steps)


    # Dataset
    log.info("Loading dataset...")
    dataset = load_dataset(config)
    dataloader = make_dataloader(dataset, config)
    dataloader_iter = iter(dataloader)

    eval_dataset = load_dataset(config)
    eval_dataloader = make_dataloader(eval_dataset, {**config, "training": {**train_cfg, "batch_size": train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 4))}})

    # Optimizer / Scheduler
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    # W&B
    run_name = args.wandb_run or (f"{'imm' if imm_enabled else 'baseline'}_seed{args.seed}")
    wandb.init(project=args.wandb_project, name=run_name, config=config, resume="allow")

    # Training loop
    graceful = GracefulExit()
    model.train()
    t0 = time.time()
    last_info = None

    log.info("Starting training from step %d to %d (grad_accum=%d, effective_batch=%d)",
             start_step, total_steps, grad_accum_steps, train_cfg.get("batch_size", 4) * grad_accum_steps)

    for step in range(start_step, total_steps):
        # Dynamic k_ratio annealing (per optimizer step)
        if imm_enabled and hasattr(model, 'k_ratio'):
            new_k_ratio = compute_k_ratio(step, config)
            if new_k_ratio is not None:
                model.k_ratio = new_k_ratio

        optimizer.zero_grad()
        accum_loss = 0.0

        for _accum_i in range(grad_accum_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            batch = preprocess_batch(batch, tokenizer, device=device)

            if imm_enabled:
                if is_twostage:
                    total_loss, info = model.forward_acis_only(batch, step=step)
                else:
                    total_loss, info = model(batch, step=step)
                last_info = info
            else:
                total_loss, info = baseline_forward(policy, batch)
                last_info = info

            if grad_accum_steps > 1:
                total_loss = total_loss / grad_accum_steps
            total_loss.backward()
            accum_loss += total_loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()
        scheduler.step()

        # Memory monitoring
        if step % 500 == 0:
            with open("/proc/meminfo") as _memf:
                _memlines = _memf.readlines()
            _mem_total = int(_memlines[0].split()[1]) / 1024 / 1024
            _mem_avail = int(_memlines[2].split()[1]) / 1024 / 1024
            log.info("[MEM] step=%d total=%.1fGB avail=%.1fGB used=%.1fGB", step, _mem_total, _mem_avail, _mem_total - _mem_avail)

        # Logging
        if step % log_every == 0:
            log_dict = {
                "train/total_loss": accum_loss,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": step,
            }

            if imm_enabled and isinstance(last_info, dict):
                log_dict.update({
                    "train/action_loss": last_info.get("action_loss", torch.tensor(0.0)).item(),
                    "train/margin_loss": last_info.get("margin_loss", torch.tensor(0.0)).item(),
                    "train/lambda": last_info.get("lambda", torch.tensor(0.0)).item(),
                })
                if hasattr(model, 'k_ratio'):
                    log_dict["train/k_ratio"] = model.k_ratio
                if "margin" in last_info:
                    margins = last_info["margin"]
                    log_dict.update({
                        "margin/mean": margins.mean().item(),
                        "margin/median": margins.median().item(),
                        "margin/std": margins.std().item(),
                        "margin/min": margins.min().item(),
                        "margin/max": margins.max().item(),
                    })
                    log_dict["margin/histogram"] = wandb.Histogram(margins.cpu().numpy())
                if "importance_scores" in last_info:
                    scores = last_info["importance_scores"]
                    k = last_info.get("k", int(scores.shape[1] * 0.5))
                    topk_vals, _ = torch.topk(scores, k, dim=-1)
                    log_dict["scores/topk_mean"] = topk_vals.mean().item()
                if "lipschitz_estimate" in last_info:
                    L = last_info["lipschitz_estimate"]
                    if "margin" in last_info:
                        eps_test = 0.01
                        k_val = last_info.get("k", 64)
                        ratio = L * eps_test * k_val / margins.clamp(min=1e-8)
                        log_dict["gate/L_eps_delta_ratio_mean"] = ratio.mean().item()
                        log_dict["gate/lipschitz_estimate"] = L
            else:
                if isinstance(last_info, dict) and "action_loss" in last_info:
                    log_dict["train/action_loss"] = last_info["action_loss"].item() if isinstance(last_info["action_loss"], torch.Tensor) else last_info["action_loss"]

            wandb.log(log_dict, step=step)

            elapsed = time.time() - t0
            steps_done = step - start_step + 1
            sps = steps_done / elapsed if elapsed > 0 else 0
            if imm_enabled and isinstance(last_info, dict):
                action_l = last_info.get("action_loss", torch.tensor(0.0)).item()
                margin_l = last_info.get("margin_loss", torch.tensor(0.0)).item()
                margin_mean = last_info.get("margin", torch.zeros(1)).mean().item() if "margin" in last_info else 0.0
                log.info(
                    "step=%d/%d  loss=%.4f  action=%.4f  margin_loss=%.4f  Δ=%.4f  lr=%.2e  grad=%.2f  sps=%.1f",
                    step, total_steps, accum_loss,
                    action_l, margin_l, margin_mean,
                    optimizer.param_groups[0]["lr"], grad_norm, sps,
                )
            else:
                log.info(
                    "step=%d/%d  loss=%.4f  lr=%.2e  grad=%.2f  sps=%.1f",
                    step, total_steps, accum_loss,
                    optimizer.param_groups[0]["lr"], grad_norm, sps,
                )

        # Eval
        if step > 0 and step % eval_every == 0:
            log.info("Running evaluation at step %d...", step)
            eval_metrics = evaluate(model, eval_dataloader, step, config, device, imm_enabled, tokenizer=tokenizer)
            wandb.log(eval_metrics, step=step)
            for k, v in eval_metrics.items():
                log.info("  %s: %.4f", k, v)

        # Checkpoint
        if step > 0 and step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, config, last_info, output_dir)

        # Empirical Lipschitz monitoring
        if imm_enabled and hasattr(model, "acis") and step > 0 and step % lip_monitor_every == 0:
            try:
                with torch.no_grad():
                    images, _ = model.policy.prepare_images(batch)
                    vis_tokens, _ = model._get_visual_tokens(images)
                    actions = model.policy.prepare_action(batch)
                    bm = model.base_model
                    _noise = bm.sample_noise(actions.shape, actions.device)
                    _time = bm.sample_time(actions.shape[0], actions.device)
                    _time_expanded = _time[:, None, None]
                    _x_t = _time_expanded * _noise + (1 - _time_expanded) * actions
                    a_query, _, _ = bm.embed_suffix(_x_t, _time)

                    from imm.metrics import compute_empirical_lipschitz
                    L_emp = compute_empirical_lipschitz(
                        model.acis, vis_tokens, a_query,
                        num_samples=10, epsilon=0.001,
                    )
                    sn = model.acis.get_lipschitz_estimate()

                if L_emp > 0.5:
                    log.warning("[LIP_MONITOR] step=%d, empirical_L=%.4f, spectral_norm=%.4f \u2014 WARNING: L > 0.5", step, L_emp, sn)
                else:
                    log.info("[LIP_MONITOR] step=%d, empirical_L=%.4f, spectral_norm=%.4f", step, L_emp, sn)
            except Exception as e:
                log.warning("[LIP_MONITOR] step=%d, failed: %s", step, str(e))

        if graceful.should_exit:
            log.info("Graceful exit at step %d", step)
            save_checkpoint(model, optimizer, scheduler, step, config, last_info, output_dir)
            break

    # Final checkpoint
    if not graceful.should_exit:
        save_checkpoint(model, optimizer, scheduler, total_steps - 1, config, last_info, output_dir)
        if is_twostage and ts_cfg.get('checkpoint_after_stage1', True):
            stage1_dir = str(Path(output_dir) / 'stage1_done')
            save_checkpoint(model, optimizer, scheduler, total_steps - 1, config, last_info, stage1_dir)
            log.info("Stage 1 complete. Checkpoint: %s", stage1_dir)

    wandb.finish()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
