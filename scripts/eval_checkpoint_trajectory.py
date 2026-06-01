#!/usr/bin/env python3
"""Evaluate v8 checkpoints across training trajectory.

Runs standard (ACIS) and bypass evals at eps=0.0 for each checkpoint.
Outputs per-checkpoint JSONs to eval_results/trajectory/.
"""

import json
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import numpy as np
import random

def main():
    config_path = "configs/imm_anchor_v8.yaml"
    checkpoint_dir = Path("checkpoints/imm_anchor_v8")
    output_dir = Path("eval_results/trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    max_batches = 100
    gpu = 0  # CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 to cuda:0

    from evaluate import load_config, load_model_and_checkpoint, evaluate_full, estimate_lipschitz
    config = load_config(config_path)

    # Only eps=0.0 to save time
    config.setdefault("evaluation", {})["perturbations"] = [0.0]

    device = torch.device(f"cuda:{gpu}")

    # Load dataset once
    from train import load_dataset, make_dataloader, get_tokenizer_from_policy
    dataset = load_dataset(config)
    dl_config = {
        **config,
        "training": {**config.get("training", {}),
                     "batch_size": config.get("training", {}).get("eval_batch_size", 4)},
    }
    dataloader = make_dataloader(dataset, dl_config)

    # Find checkpoints
    ckpts = sorted(checkpoint_dir.glob("step_*"))
    print(f"Found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")
    print(f"Device: {device}, max_batches: {max_batches}")

    for ckpt in ckpts:
        step_name = ckpt.name
        standard_out = output_dir / f"v8_{step_name}_standard.json"
        bypass_out = output_dir / f"v8_{step_name}_bypass.json"

        if standard_out.exists() and bypass_out.exists():
            print(f"\n[SKIP] {step_name}: both results exist")
            continue

        print(f"\n{'='*60}")
        print(f"[EVAL] {step_name}")
        print(f"{'='*60}")

        t0 = time.time()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model, imm_enabled = load_model_and_checkpoint(config, str(ckpt), device)
        policy_for_tok = model.policy if hasattr(model, "policy") else model
        tokenizer = get_tokenizer_from_policy(policy_for_tok)

        # Standard eval (with ACIS)
        if not standard_out.exists():
            torch.manual_seed(seed)
            lipschitz_const = 0.0
            if imm_enabled and hasattr(model, "acis"):
                lipschitz_const = estimate_lipschitz(
                    model, dataloader, device, max_batches=10, num_samples=100
                )

            results = evaluate_full(
                model, dataloader, config, device, imm_enabled,
                lipschitz_const, max_batches=max_batches, seed=seed,
                tokenizer=tokenizer, bypass_acis=False,
            )
            results["_meta"] = {
                "checkpoint": str(ckpt), "bypass_acis": False,
                "max_batches": max_batches, "step": step_name,
            }
            with open(standard_out, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Standard saved: {standard_out}")

        # Bypass eval (no ACIS, random prune for pruneTrue)
        if not bypass_out.exists():
            torch.manual_seed(seed)
            results = evaluate_full(
                model, dataloader, config, device, imm_enabled,
                0.0, max_batches=max_batches, seed=seed,
                tokenizer=tokenizer, bypass_acis=True,
            )
            results["_meta"] = {
                "checkpoint": str(ckpt), "bypass_acis": True,
                "max_batches": max_batches, "step": step_name,
            }
            with open(bypass_out, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Bypass saved: {bypass_out}")

        del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"  {step_name} done in {elapsed:.0f}s")

    print("\n" + "="*60)
    print("ALL EVALUATIONS COMPLETE")
    print("="*60)

    # Quick summary
    for ckpt in ckpts:
        step_name = ckpt.name
        sf = output_dir / f"v8_{step_name}_standard.json"
        bf = output_dir / f"v8_{step_name}_bypass.json"
        if sf.exists() and bf.exists():
            with open(sf) as f:
                sd = json.load(f)
            with open(bf) as f:
                bd = json.load(f)
            s_loss = sd.get("eps0.0_pruneFalse", {}).get("action_loss", "N/A")
            s_prune = sd.get("eps0.0_pruneTrue", {}).get("action_loss", "N/A")
            b_loss = bd.get("eps0.0_pruneFalse", {}).get("action_loss", "N/A")
            b_prune = bd.get("eps0.0_pruneTrue", {}).get("action_loss", "N/A")
            print(f"{step_name}: std={s_loss}, std_prune={s_prune}, bypass={b_loss}, bypass_random_prune={b_prune}")


if __name__ == "__main__":
    main()
