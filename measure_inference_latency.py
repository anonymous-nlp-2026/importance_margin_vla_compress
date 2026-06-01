"""Measure SmolVLA inference latency: baseline (128 tokens) vs Smart Compress M=32/M=64.

Usage:
    CUDA_VISIBLE_DEVICES=2 python measure_inference_latency.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")

import torch
import numpy as np

PROJ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

N_WARMUP = 10
N_RUNS = 100


def make_dummy_batch(policy, tokenizer, device):
    """Create a synthetic batch matching LIBERO observation shapes."""
    batch = {}
    # 2 cameras, 3x256x256
    batch["observation.images.image"] = torch.randn(1, 3, 256, 256, device=device)
    batch["observation.images.image2"] = torch.randn(1, 3, 256, 256, device=device)
    # state: 8-dim (eef_pos 3 + eef_aa 3 + gripper 2)
    batch["observation.state"] = torch.randn(1, 8, device=device)
    # language tokens
    text = "pick up the red cup\n"
    encoded = tokenizer(
        [text], padding="longest", max_length=48,
        truncation=True, return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"].to(device)
    batch["observation.language.attention_mask"] = encoded["attention_mask"].bool().to(device)
    return batch


def time_inference(fn, batch, n_warmup, n_runs, device):
    """Time inference function with proper GPU synchronization."""
    # Warmup
    for _ in range(n_warmup):
        fn(batch)
    torch.cuda.synchronize(device)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn(batch)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return np.array(latencies)


def get_gpu_memory_mb(device):
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def main():
    device = torch.device("cuda:0")
    log.info("Device: %s", device)
    log.info("GPU: %s", torch.cuda.get_device_name(device))

    # Load config and base policy
    from train_smart_compress import load_smolvla_policy, load_config
    from smart_compress_module import SmartCompressWrapper
    from train import get_tokenizer_from_policy

    config = load_config("configs/smart_compress.yaml")
    log.info("Loading SmolVLA policy...")
    policy = load_smolvla_policy(config, device)
    tokenizer = get_tokenizer_from_policy(policy)

    dummy_batch = make_dummy_batch(policy, tokenizer, device)
    log.info("Dummy batch created.")

    results = {"configs": {}, "device": torch.cuda.get_device_name(device), "n_warmup": N_WARMUP, "n_runs": N_RUNS}

    # ============================================================
    # 1. Baseline: 128 vision tokens (no compression)
    # ============================================================
    log.info("=" * 60)
    log.info("CONFIG: baseline_128 (no compression, 128 vision tokens)")
    log.info("=" * 60)

    policy.eval()
    torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def baseline_fn(batch):
        return policy.predict_action_chunk(batch)

    latencies_baseline = time_inference(baseline_fn, dummy_batch, N_WARMUP, N_RUNS, device)
    mem_baseline = get_gpu_memory_mb(device)
    total_params_baseline = sum(p.numel() for p in policy.parameters())

    results["configs"]["baseline_128"] = {
        "n_vision_tokens": 128,
        "mean_ms": round(float(latencies_baseline.mean()), 3),
        "std_ms": round(float(latencies_baseline.std()), 3),
        "median_ms": round(float(np.median(latencies_baseline)), 3),
        "min_ms": round(float(latencies_baseline.min()), 3),
        "max_ms": round(float(latencies_baseline.max()), 3),
        "gpu_memory_mb": round(mem_baseline, 1),
        "total_params": total_params_baseline,
    }
    log.info("Baseline 128: mean=%.2f ms, std=%.2f ms, median=%.2f ms",
             latencies_baseline.mean(), latencies_baseline.std(), np.median(latencies_baseline))

    # ============================================================
    # 2. Smart Compress M=32
    # ============================================================
    log.info("=" * 60)
    log.info("CONFIG: smart_compress_m32 (32 vision tokens)")
    log.info("=" * 60)

    compress_config_m32 = {"num_queries": 32, "num_heads": 12, "num_layers": 2, "dropout": 0.1}
    wrapper_m32 = SmartCompressWrapper(policy, compress_config=compress_config_m32)
    wrapper_m32.to(device)

    ckpt_m32 = torch.load("checkpoints/smart_compress_m32/checkpoint.pt", map_location=device, weights_only=False)
    wrapper_m32.load_state_dict(ckpt_m32["model_state_dict"], strict=False)
    wrapper_m32.eval()
    log.info("M=32 checkpoint loaded.")

    torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def m32_fn(batch):
        return wrapper_m32.forward_inference(batch)

    latencies_m32 = time_inference(m32_fn, dummy_batch, N_WARMUP, N_RUNS, device)
    mem_m32 = get_gpu_memory_mb(device)
    compressor_params_m32 = sum(p.numel() for p in wrapper_m32.compressor.parameters())

    results["configs"]["smart_compress_m32"] = {
        "n_vision_tokens": 32,
        "mean_ms": round(float(latencies_m32.mean()), 3),
        "std_ms": round(float(latencies_m32.std()), 3),
        "median_ms": round(float(np.median(latencies_m32)), 3),
        "min_ms": round(float(latencies_m32.min()), 3),
        "max_ms": round(float(latencies_m32.max()), 3),
        "gpu_memory_mb": round(mem_m32, 1),
        "total_params": total_params_baseline + compressor_params_m32,
        "compressor_params": compressor_params_m32,
    }
    log.info("M=32: mean=%.2f ms, std=%.2f ms, median=%.2f ms",
             latencies_m32.mean(), latencies_m32.std(), np.median(latencies_m32))

    # Free M=32 wrapper (keep policy)
    del wrapper_m32, ckpt_m32
    torch.cuda.empty_cache()

    # ============================================================
    # 3. Smart Compress M=64
    # ============================================================
    log.info("=" * 60)
    log.info("CONFIG: smart_compress_m64 (64 vision tokens)")
    log.info("=" * 60)

    compress_config_m64 = {"num_queries": 64, "num_heads": 12, "num_layers": 2, "dropout": 0.1}
    wrapper_m64 = SmartCompressWrapper(policy, compress_config=compress_config_m64)
    wrapper_m64.to(device)

    ckpt_m64 = torch.load("checkpoints/smart_compress_m64/checkpoint.pt", map_location=device, weights_only=False)
    wrapper_m64.load_state_dict(ckpt_m64["model_state_dict"], strict=False)
    wrapper_m64.eval()
    log.info("M=64 checkpoint loaded.")

    torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def m64_fn(batch):
        return wrapper_m64.forward_inference(batch)

    latencies_m64 = time_inference(m64_fn, dummy_batch, N_WARMUP, N_RUNS, device)
    mem_m64 = get_gpu_memory_mb(device)
    compressor_params_m64 = sum(p.numel() for p in wrapper_m64.compressor.parameters())

    results["configs"]["smart_compress_m64"] = {
        "n_vision_tokens": 64,
        "mean_ms": round(float(latencies_m64.mean()), 3),
        "std_ms": round(float(latencies_m64.std()), 3),
        "median_ms": round(float(np.median(latencies_m64)), 3),
        "min_ms": round(float(latencies_m64.min()), 3),
        "max_ms": round(float(latencies_m64.max()), 3),
        "gpu_memory_mb": round(mem_m64, 1),
        "total_params": total_params_baseline + compressor_params_m64,
        "compressor_params": compressor_params_m64,
    }
    log.info("M=64: mean=%.2f ms, std=%.2f ms, median=%.2f ms",
             latencies_m64.mean(), latencies_m64.std(), np.median(latencies_m64))

    # ============================================================
    # Speedups
    # ============================================================
    baseline_mean = results["configs"]["baseline_128"]["mean_ms"]
    m32_mean = results["configs"]["smart_compress_m32"]["mean_ms"]
    m64_mean = results["configs"]["smart_compress_m64"]["mean_ms"]

    results["speedup_m32_vs_baseline"] = round(baseline_mean / m32_mean, 4) if m32_mean > 0 else None
    results["speedup_m64_vs_baseline"] = round(baseline_mean / m64_mean, 4) if m64_mean > 0 else None

    # ============================================================
    # Save
    # ============================================================
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_latency_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("Baseline (128 tokens): %.2f ms (+-%.2f)", baseline_mean, results["configs"]["baseline_128"]["std_ms"])
    log.info("M=32  (32 tokens):     %.2f ms (+-%.2f)  speedup=%.2fx", m32_mean, results["configs"]["smart_compress_m32"]["std_ms"], results["speedup_m32_vs_baseline"])
    log.info("M=64  (64 tokens):     %.2f ms (+-%.2f)  speedup=%.2fx", m64_mean, results["configs"]["smart_compress_m64"]["std_ms"], results["speedup_m64_vs_baseline"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
