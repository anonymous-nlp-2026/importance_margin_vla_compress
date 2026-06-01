"""Two-stage IMM training: ACIS-first, then LoRA.

Stage 1: Freeze LoRA, train ACIS only with margin_loss.
         ACIS learns importance scoring on fixed backbone visual features.
         Skips VLM forward entirely — only runs visual encoder + ACIS + margin_loss.
Stage 2: Freeze ACIS, train LoRA only with action_loss.
         Uses gradient_isolation to ensure only action_loss gradient reaches LoRA.

Usage:
    python train_twostage.py --config configs/imm_anchor_v8b_twostage.yaml
    python train_twostage.py --config configs/imm_anchor_v8b_twostage.yaml --resume_stage2 checkpoints/.../stage1_done
    python train_twostage.py --config configs/imm_anchor_v8b_twostage.yaml --max_steps 5 --dry_run

Depends: lerobot, torch, wandb, peft, imm (local package).
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from train import (
    load_smolvla_policy,
    load_dataset,
    make_dataloader,
    wrap_with_imm,
    get_tokenizer_from_policy,
    preprocess_batch,
    save_checkpoint,
    GracefulExit,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage IMM training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_stage2", type=str, default=None,
                        help="Path to stage1 checkpoint to skip stage1 and start stage2 directly")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="imm-vla-compress")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override total steps per stage (for dry-run)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry-run mode: skip wandb, minimal steps, verify gradients")
    return parser.parse_args()


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


# ─── Parameter group utilities ─────────────────────────────────────────────────

def get_lora_params(model) -> List[torch.nn.Parameter]:
    params = []
    for name, p in model.named_parameters():
        if "lora_" in name:
            params.append(p)
    return params


def get_acis_params(model) -> List[torch.nn.Parameter]:
    return list(model.acis.parameters())


def get_expert_params(model) -> List[torch.nn.Parameter]:
    """Non-ACIS trainable params for Full FT mode. Call before stage freeze/unfreeze."""
    acis_param_ids = {id(p) for p in model.acis.parameters()}
    return [p for p in model.parameters() if id(p) not in acis_param_ids and p.requires_grad]


def freeze_params(params: List[torch.nn.Parameter]):
    for p in params:
        p.requires_grad = False


def unfreeze_params(params: List[torch.nn.Parameter]):
    for p in params:
        p.requires_grad = True


def verify_grad_state(model, stage: str, dry_run: bool = False,
                     train_mode: str = "twostage", expert_params=None):
    acis_params = get_acis_params(model)
    acis_trainable = sum(p.requires_grad for p in acis_params)

    if train_mode == "twostage_fullft":
        expert_trainable = sum(p.requires_grad for p in expert_params) if expert_params else 0
        log.info("[%s] Expert params: %d total, %d trainable", stage, len(expert_params or []), expert_trainable)
        log.info("[%s] ACIS params: %d total, %d trainable", stage, len(acis_params), acis_trainable)
        if dry_run:
            if stage == "stage1":
                assert expert_trainable == 0, f"Stage1: expert should be frozen but {expert_trainable} are trainable"
                assert acis_trainable == len(acis_params), f"Stage1: all ACIS params should be trainable"
            elif stage == "stage2":
                assert acis_trainable == 0, f"Stage2: ACIS should be frozen but {acis_trainable} are trainable"
                assert expert_trainable == len(expert_params), f"Stage2: all expert params should be trainable"
    else:
        lora_params = get_lora_params(model)
        lora_trainable = sum(p.requires_grad for p in lora_params)
        log.info("[%s] LoRA params: %d total, %d trainable", stage, len(lora_params), lora_trainable)
        log.info("[%s] ACIS params: %d total, %d trainable", stage, len(acis_params), acis_trainable)
        if dry_run:
            if stage == "stage1":
                assert lora_trainable == 0, f"Stage1: LoRA should be frozen but {lora_trainable} are trainable"
                assert acis_trainable == len(acis_params), f"Stage1: all ACIS params should be trainable"
            elif stage == "stage2":
                assert acis_trainable == 0, f"Stage2: ACIS should be frozen but {acis_trainable} are trainable"
                assert lora_trainable == len(lora_params), f"Stage2: all LoRA params should be trainable"


def verify_gradients_after_backward(model, stage: str, train_mode: str = "twostage", expert_params=None):
    acis_params = get_acis_params(model)
    if train_mode == "twostage_fullft":
        if stage == "stage1":
            expert_with_grad = sum(1 for p in (expert_params or []) if p.grad is not None and p.grad.abs().sum() > 0)
            acis_with_grad = sum(1 for p in acis_params if p.grad is not None and p.grad.abs().sum() > 0)
            log.info("[stage1 grad check] Expert with grad: %d (expect 0), ACIS with grad: %d",
                     expert_with_grad, acis_with_grad)
            assert expert_with_grad == 0, f"Stage1: expert got gradients ({expert_with_grad} params)"
        elif stage == "stage2":
            acis_with_grad = sum(1 for p in acis_params if p.grad is not None and p.grad.abs().sum() > 0)
            expert_with_grad = sum(1 for p in (expert_params or []) if p.grad is not None and p.grad.abs().sum() > 0)
            log.info("[stage2 grad check] ACIS with grad: %d (expect 0), Expert with grad: %d",
                     acis_with_grad, expert_with_grad)
            assert acis_with_grad == 0, f"Stage2: ACIS got gradients ({acis_with_grad} params)"
    else:
        lora_params = get_lora_params(model)
        if stage == "stage1":
            lora_with_grad = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().sum() > 0)
            acis_with_grad = sum(1 for p in acis_params if p.grad is not None and p.grad.abs().sum() > 0)
            log.info("[stage1 grad check] LoRA with grad: %d (expect 0), ACIS with grad: %d",
                     lora_with_grad, acis_with_grad)
            assert lora_with_grad == 0, f"Stage1: LoRA got gradients ({lora_with_grad} params)"
        elif stage == "stage2":
            acis_with_grad = sum(1 for p in acis_params if p.grad is not None and p.grad.abs().sum() > 0)
            lora_with_grad = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().sum() > 0)
            log.info("[stage2 grad check] ACIS with grad: %d (expect 0), LoRA with grad: %d",
                     acis_with_grad, lora_with_grad)
            assert acis_with_grad == 0, f"Stage2: ACIS got gradients ({acis_with_grad} params)"


# ─── Stage-specific forward functions ──────────────────────────────────────────

def stage1_forward(model, batch, step):
    """Stage 1: ACIS-only forward. Skips VLM — computes margin_loss directly.

    Detaches vis_tokens and suffix_embs so gradient only flows through ACIS params.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    bm = model.base_model
    images, img_masks = model.policy.prepare_images(batch)
    vis_tokens, _ = model._get_visual_tokens(images)
    actions = model.policy.prepare_action(batch)

    noise = bm.sample_noise(actions.shape, actions.device)
    t = bm.sample_time(actions.shape[0], actions.device)
    t_expanded = t[:, None, None]
    x_t = t_expanded * noise + (1 - t_expanded) * actions
    suffix_embs, _, _ = bm.embed_suffix(x_t, t)

    # Detach backbone outputs so gradient only reaches ACIS
    importance_scores = model.acis(suffix_embs.detach(), vis_tokens.detach())

    N_vis = vis_tokens.shape[1]
    k = max(1, int(N_vis * model.k_ratio))
    margin_loss, margin_info = model.imm_loss.margin_loss(importance_scores, k)

    info = {
        **margin_info,
        "margin_loss": margin_loss.detach(),
        "importance_scores": importance_scores.detach(),
        "k": k,
    }
    return margin_loss, info


def stage2_forward(model, batch, step):
    """Stage 2: Full forward with gradient_isolation=True.

    gradient_isolation ensures action_loss gradient doesn't reach ACIS,
    and margin_loss gradient doesn't reach backbone.
    ACIS is frozen, so only LoRA gets updated via action_loss.
    """
    old_gi = model.gradient_isolation
    model.gradient_isolation = True
    try:
        total_loss, info = model(batch, step=step)
    finally:
        model.gradient_isolation = old_gi
    return total_loss, info


# ─── Training loop ─────────────────────────────────────────────────────────────

def make_stage_optimizer(params, lr, betas=(0.9, 0.95), weight_decay=1e-10):
    trainable = [p for p in params if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, betas=betas, weight_decay=weight_decay)


def make_stage_scheduler(optimizer, warmup_steps, total_steps, decay_lr=2.5e-6):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=decay_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def run_stage(
    model, dataloader, stage, forward_fn, optimizer, scheduler,
    total_steps, config, device, output_dir, dry_run=False, graceful=None,
    train_mode="twostage", expert_params=None,
):
    grad_clip = config["training"].get("grad_clip", 10.0)
    log_every = config["training"].get("log_every", 50)
    save_every = config["training"].get("save_every", 2000)

    dataloader_iter = iter(dataloader)
    tokenizer = get_tokenizer_from_policy(model.policy)
    model.train()
    t0 = time.time()

    log.info("=== Starting %s: %d steps ===", stage, total_steps)

    for step in range(total_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = preprocess_batch(batch, tokenizer, device=device)

        loss, info = forward_fn(model, batch, step)

        if dry_run and step == 0:
            model.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()
        loss.backward()

        if dry_run and step == 0:
            verify_gradients_after_backward(model, stage, train_mode=train_mode, expert_params=expert_params)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            if stage == "stage1":
                ml = info.get("margin_loss", torch.tensor(0.0))
                ml = ml.item() if isinstance(ml, torch.Tensor) else ml
                log.info("[%s] step=%d/%d  margin_loss=%.4f  lr=%.2e  grad=%.2f  sps=%.1f",
                         stage, step, total_steps, ml,
                         optimizer.param_groups[0]["lr"], grad_norm, sps)
            else:
                al = info.get("action_loss", torch.tensor(0.0))
                al = al.item() if isinstance(al, torch.Tensor) else al
                tl = loss.item()
                log.info("[%s] step=%d/%d  total=%.4f  action=%.4f  lr=%.2e  grad=%.2f  sps=%.1f",
                         stage, step, total_steps, tl, al,
                         optimizer.param_groups[0]["lr"], grad_norm, sps)

            if not dry_run:
                import wandb
                wandb.log({
                    f"{stage}/loss": loss.item(),
                    f"{stage}/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    f"{stage}/lr": optimizer.param_groups[0]["lr"],
                })

        if not dry_run and step > 0 and step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, config, info,
                           str(Path(output_dir) / stage))

        if graceful and graceful.should_exit:
            log.info("Graceful exit at %s step %d", stage, step)
            save_checkpoint(model, optimizer, scheduler, step, config, info,
                           str(Path(output_dir) / stage))
            return False

    log.info("=== %s complete ===", stage)
    return True


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    twostage_cfg = config.get("twostage", {})
    stage1_steps = twostage_cfg.get("stage1_steps", 10000)
    stage2_steps = twostage_cfg.get("stage2_steps", 20000)
    stage1_lr = twostage_cfg.get("stage1_lr", 3e-4)
    stage2_lr = twostage_cfg.get("stage2_lr", 1e-4)
    stage1_warmup = twostage_cfg.get("stage1_warmup_steps", 500)
    stage2_warmup = twostage_cfg.get("stage2_warmup_steps", 1000)
    train_mode = twostage_cfg.get("train_mode", "twostage")

    if args.max_steps:
        if stage1_steps > 0:
            stage1_steps = args.max_steps
        if stage2_steps > 0:
            stage2_steps = args.max_steps

    output_dir = config["training"].get("output_dir", "checkpoints/twostage")

    log.info("Loading SmolVLA policy...")
    policy = load_smolvla_policy(config, device)
    tokenizer = get_tokenizer_from_policy(policy)

    log.info("Wrapping with IMMSmolVLAWrapper...")
    model = wrap_with_imm(policy, config)
    model.to(device)

    # Load baseline checkpoint (converged backbone+LoRA)
    baseline_ckpt = twostage_cfg.get("baseline_checkpoint")
    if baseline_ckpt and not args.resume_stage2:
        ckpt_path = Path(baseline_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = ARTIFACTS_DIR / ckpt_path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.policy.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded baseline weights from %s (step %d)", ckpt_path, ckpt.get("step", -1))

    # For fullft mode, capture expert params before any stage freeze/unfreeze
    if train_mode == "twostage_fullft":
        expert_params = get_expert_params(model)
        log.info("Full FT mode: %d expert params, %d ACIS params",
                 len(expert_params), sum(1 for _ in model.acis.parameters()))
    else:
        expert_params = None

    log.info("Loading dataset...")
    dataset = load_dataset(config)
    dataloader = make_dataloader(dataset, config)

    if not args.dry_run:
        import wandb
        run_name = args.wandb_run or f"v8b_twostage_seed{args.seed}"
        wandb.init(project=args.wandb_project, name=run_name, config=config, resume="allow")

    graceful = GracefulExit()

    # ─── Stage 1: Train ACIS only ──────────────────────────────────────────────
    skip_stage1 = args.resume_stage2 is not None

    if not skip_stage1:
        log.info("=" * 60)
        if train_mode == "twostage_fullft":
            log.info("STAGE 1: Freeze expert, train ACIS (margin_loss only) [Full FT]")
        else:
            log.info("STAGE 1: Freeze LoRA, train ACIS (margin_loss only)")
        log.info("=" * 60)

        acis_params = get_acis_params(model)
        if train_mode == "twostage_fullft":
            freeze_params(expert_params)
        else:
            lora_params = get_lora_params(model)
            freeze_params(lora_params)
        unfreeze_params(acis_params)
        verify_grad_state(model, "stage1", dry_run=args.dry_run,
                         train_mode=train_mode, expert_params=expert_params)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        log.info("Param counts: total=%d, trainable(ACIS)=%d (%.4f%%), frozen=%d",
                 total_params, trainable_params, 100.0 * trainable_params / total_params, frozen_params)

        optimizer1 = make_stage_optimizer(acis_params, lr=stage1_lr)
        scheduler1 = make_stage_scheduler(optimizer1, stage1_warmup, stage1_steps)

        completed = run_stage(
            model=model, dataloader=dataloader, stage="stage1",
            forward_fn=stage1_forward, optimizer=optimizer1, scheduler=scheduler1,
            total_steps=stage1_steps, config=config, device=device,
            output_dir=output_dir, dry_run=args.dry_run, graceful=graceful,
            train_mode=train_mode, expert_params=expert_params,
        )

        if not completed:
            log.info("Exiting after stage1 interruption")
            if not args.dry_run:
                import wandb; wandb.finish()
            return

        stage1_ckpt_dir = Path(output_dir) / "stage1_done"
        stage1_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "stage": "stage1_complete",
        }, stage1_ckpt_dir / "checkpoint.pt")
        log.info("Stage 1 checkpoint saved: %s", stage1_ckpt_dir)

    else:
        log.info("Resuming from stage1 checkpoint: %s", args.resume_stage2)
        ckpt_path = Path(args.resume_stage2)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "checkpoint.pt"
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        log.info("Loaded stage1 checkpoint, proceeding to stage2")

    # ─── Stage 2: Train LoRA only ──────────────────────────────────────────────
    if stage2_steps == 0:
        log.info("stage2_steps=0, skipping Stage 2 (ACIS-only mode)")
        if not args.dry_run:
            import wandb; wandb.finish()
        if args.dry_run:
            log.info("=" * 60)
            log.info("DRY RUN PASSED — ACIS-only training executed successfully")
            log.info("=" * 60)
        log.info("Two-stage training complete (ACIS-only mode).")
        return

    if graceful.should_exit:
        if not args.dry_run:
            import wandb; wandb.finish()
        return

    log.info("=" * 60)
    if train_mode == "twostage_fullft":
        log.info("STAGE 2: Freeze ACIS, train expert (action_loss only) [Full FT]")
    else:
        log.info("STAGE 2: Freeze ACIS, train LoRA (action_loss only)")
    log.info("=" * 60)

    # Clear stale gradients from stage1
    model.zero_grad(set_to_none=True)

    acis_params = get_acis_params(model)
    if train_mode == "twostage_fullft":
        unfreeze_params(expert_params)
        freeze_params(acis_params)
        verify_grad_state(model, "stage2", dry_run=args.dry_run,
                         train_mode=train_mode, expert_params=expert_params)
        optimizer2 = make_stage_optimizer(expert_params, lr=stage2_lr)
    else:
        lora_params = get_lora_params(model)
        unfreeze_params(lora_params)
        freeze_params(acis_params)
        verify_grad_state(model, "stage2", dry_run=args.dry_run)
        optimizer2 = make_stage_optimizer(lora_params, lr=stage2_lr)
    scheduler2 = make_stage_scheduler(optimizer2, stage2_warmup, stage2_steps)

    completed = run_stage(
        model=model, dataloader=dataloader, stage="stage2",
        forward_fn=stage2_forward, optimizer=optimizer2, scheduler=scheduler2,
        total_steps=stage2_steps, config=config, device=device,
        output_dir=output_dir, dry_run=args.dry_run, graceful=graceful,
        train_mode=train_mode, expert_params=expert_params,
    )

    if completed and not args.dry_run:
        final_dir = Path(output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "stage": "complete",
        }, final_dir / "checkpoint.pt")
        log.info("Final checkpoint saved: %s", final_dir)

    if not args.dry_run:
        import wandb; wandb.finish()

    if args.dry_run:
        log.info("=" * 60)
        log.info("DRY RUN PASSED — both stages executed successfully")
        log.info("=" * 60)

    log.info("Two-stage training complete.")


if __name__ == "__main__":
    main()
