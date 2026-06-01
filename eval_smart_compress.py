"""Offline evaluation for Smart Compress models.

Evaluates action prediction quality (flow matching MSE) on held-out data,
comparing compressed vs uncompressed (baseline) forward passes.

Usage:
    python eval_smart_compress.py --checkpoint checkpoints/smart_compress_m32/checkpoint.pt \
        --config configs/smart_compress.yaml
    python eval_smart_compress.py --checkpoint checkpoints/smart_compress_m32/checkpoint.pt \
        --config configs/smart_compress.yaml --compare-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import yaml

ARTIFACTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACTS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval Smart Compress")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Also compute baseline (uncompressed) loss for comparison")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    import copy
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


def main():
    args = parse_args()
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # load model
    from train_smart_compress import (
        load_smolvla_policy, wrap_with_compressor, freeze_for_compress_training,
        load_dataset, make_dataloader,
    )

    log.info("Loading model...")
    policy = load_smolvla_policy(config, device)
    model = wrap_with_compressor(policy, config)
    model.to(device)
    freeze_for_compress_training(model)

    # load checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint.pt"
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ckpt_step = ckpt.get("step", "?")
    log.info("Checkpoint step: %s", ckpt_step)

    # dataset
    dataset = load_dataset(config)
    eval_cfg = {**config, "training": {**config.get("training", {}), "batch_size": 1}}
    dataloader = make_dataloader(dataset, eval_cfg)

    # evaluate compressed
    log.info("Evaluating compressed model (M=%d)...", model.compressor.num_queries)
    compressed_losses = []
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            if n >= args.max_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, info = model.forward(batch)
            compressed_losses.append(loss.item())
            n += 1

    compressed_mean = sum(compressed_losses) / len(compressed_losses)
    compressed_std = (sum((x - compressed_mean)**2 for x in compressed_losses) / len(compressed_losses)) ** 0.5
    log.info("Compressed loss: %.6f +/- %.6f (n=%d)", compressed_mean, compressed_std, n)

    results = {
        "compressed": {
            "loss_mean": compressed_mean,
            "loss_std": compressed_std,
            "num_queries": model.compressor.num_queries,
            "compression_ratio": 128.0 / model.compressor.num_queries,
            "n_batches": n,
        },
        "_meta": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": ckpt_step,
            "config": str(args.config),
        },
    }

    # baseline comparison
    if args.compare_baseline:
        log.info("Evaluating baseline (uncompressed)...")
        baseline_losses = []
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if n >= args.max_batches:
                    break
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, _ = model.forward_baseline(batch)
                baseline_losses.append(loss.item())
                n += 1

        baseline_mean = sum(baseline_losses) / len(baseline_losses)
        baseline_std = (sum((x - baseline_mean)**2 for x in baseline_losses) / len(baseline_losses)) ** 0.5
        log.info("Baseline loss: %.6f +/- %.6f (n=%d)", baseline_mean, baseline_std, n)

        delta = compressed_mean - baseline_mean
        pct = 100 * delta / baseline_mean if baseline_mean > 0 else float("inf")
        log.info("Delta (compressed - baseline): %.6f (%.2f%%)", delta, pct)

        results["baseline"] = {
            "loss_mean": baseline_mean,
            "loss_std": baseline_std,
            "n_batches": n,
        }
        results["comparison"] = {
            "delta": delta,
            "delta_pct": pct,
        }

    # save
    output_path = args.output
    if output_path is None:
        results_dir = Path("eval_results")
        results_dir.mkdir(exist_ok=True)
        m = model.compressor.num_queries
        output_path = results_dir / f"smart_compress_m{m}_step{ckpt_step}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", output_path)

    # summary
    log.info("=" * 50)
    log.info("SUMMARY")
    log.info("  Compression: 128 -> %d tokens (%.0f%% reduction)",
             model.compressor.num_queries,
             100 * (1 - model.compressor.num_queries / 128))
    log.info("  Compressed action loss: %.6f", compressed_mean)
    if args.compare_baseline:
        log.info("  Baseline action loss:   %.6f", baseline_mean)
        log.info("  Degradation:            %.2f%%", pct)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
