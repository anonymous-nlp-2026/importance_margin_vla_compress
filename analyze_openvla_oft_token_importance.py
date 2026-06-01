#!/usr/bin/env python
"""OpenVLA(-OFT) Token Importance Analysis on LIBERO-Object.

Pruning Curve (L2 norm zeroing) + Per-layer Attention Entropy.

Usage:
    CUDA_VISIBLE_DEVICES=1 python analyze_openvla_oft_token_importance.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "./cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

OFT_MODEL_PATH = "./openvla-oft-libero-object"
STD_MODEL_PATH = "./openvla-libero-object"
LORA_ADAPTER_PATH = "/tmp/oft_lora_locked"

LIBERO_SNAP = Path("./cache/lerobot/hub/"
                   "datasets--HuggingFaceVLA--libero/snapshots/"
                   "86958911c0f959db2bbbdb107eb3e17c5f9c798e")
DATA_DIR = LIBERO_SNAP / "data" / "chunk-000"

LIBERO_OBJECT_TASKS = range(20, 30)
FRAMES_PER_TASK = 10
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"
KEEP_RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
OUT_DIR = Path("./artifacts")


def load_libero_frames():
    tasks_meta = pq.read_table(LIBERO_SNAP / "meta" / "tasks.parquet").to_pandas()
    task_names = tasks_meta.index.tolist()

    per_task_rows = defaultdict(list)
    for pf in sorted(os.listdir(DATA_DIR)):
        tbl = pq.read_table(DATA_DIR / pf)
        for idx, tid in enumerate(tbl.column("task_index").to_pylist()):
            if tid in LIBERO_OBJECT_TASKS:
                per_task_rows[tid].append((pf, idx))

    frames = []
    for tid in sorted(per_task_rows.keys()):
        rows = per_task_rows[tid]
        step = max(1, len(rows) // FRAMES_PER_TASK)
        for pf, idx in rows[::step][:FRAMES_PER_TASK]:
            tbl = pq.read_table(DATA_DIR / pf)
            img = Image.open(io.BytesIO(
                tbl.column("observation.images.image")[idx]["bytes"].as_py()
            )).convert("RGB")
            action = np.array(tbl.column("action")[idx].as_py(), dtype=np.float32)
            frames.append({
                "image": img,
                "action": action,
                "task_index": tid,
                "task_desc": task_names[tid] if tid < len(task_names) else f"task_{tid}",
            })

    print(f"Loaded {len(frames)} frames from {len(set(f['task_index'] for f in frames))} tasks")
    return frames


def load_model(device):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Try OFT with LoRA first
    adapter_path = os.path.join(LORA_ADAPTER_PATH, "adapter_model.safetensors")
    lora_valid = False
    if os.path.exists(adapter_path):
        try:
            import safetensors.torch
            safetensors.torch.load_file(adapter_path)
            lora_valid = True
            print("LoRA adapter validated.")
        except Exception as e:
            print(f"LoRA adapter invalid: {e}")

    if lora_valid:
        print(f"Loading OFT base model from {OFT_MODEL_PATH}...")
        model = AutoModelForImageTextToText.from_pretrained(
            OFT_MODEL_PATH, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        try:
            from peft import PeftModel
            print("Merging LoRA adapter...")
            model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
            model = model.merge_and_unload()
            label = "OpenVLA-OFT (LoRA merged, LIBERO-Object)"
            print("LoRA merged!")
        except Exception as e:
            print(f"LoRA merge failed: {e}, using base model")
            label = "OpenVLA-7B (base, no LoRA)"
        model_path = OFT_MODEL_PATH
    else:
        print(f"No valid LoRA. Loading standard OpenVLA from {STD_MODEL_PATH}...")
        model = AutoModelForImageTextToText.from_pretrained(
            STD_MODEL_PATH, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        label = "OpenVLA-7B (full FT, LIBERO-Object)"
        model_path = STD_MODEL_PATH

    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    n = sum(p.numel() for p in model.parameters())
    print(f"Model: {label}, {n/1e9:.2f}B params")
    return model, processor, label


@torch.no_grad()
def pruning_curve_analysis(model, processor, frames, device):
    print("\n=== Pruning Curve Analysis ===")

    unnorm_key = list(model.norm_stats.keys())[0]
    action_dim = model.get_action_dim(unnorm_key)
    print(f"unnorm_key={unnorm_key}, action_dim={action_dim}")

    pruner_state = {"keep_ratio": 1.0, "norms_cache": None}

    def projector_hook(module, input, output):
        kr = pruner_state["keep_ratio"]
        pruner_state["norms_cache"] = output.norm(dim=-1).float().cpu()
        if kr >= 1.0:
            return output
        B, N, D = output.shape
        norms = output.norm(dim=-1)
        k = max(1, int(N * kr))
        _, bottom_idx = torch.topk(norms, N - k, dim=1, largest=False)
        mask = torch.ones(B, N, 1, device=output.device, dtype=output.dtype)
        mask.scatter_(1, bottom_idx.unsqueeze(-1), 0.0)
        return output * mask.expand_as(output)

    hook_handle = model.projector.register_forward_hook(projector_hook)

    all_results = {str(kr): [] for kr in KEEP_RATIOS}
    gt_all = []
    all_norms = []

    for fi, frame in enumerate(frames):
        if fi % 20 == 0:
            print(f"  Frame {fi}/{len(frames)}...")

        prompt = PROMPT_TEMPLATE.format(task=str(frame["task_desc"]).lower())
        inputs = processor(prompt, frame["image"]).to(device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        if input_ids[0, -1] != 29871:
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], device=device)], dim=1)

        gt_all.append(frame["action"])

        for kr in KEEP_RATIOS:
            pruner_state["keep_ratio"] = kr
            gen = model.generate(
                input_ids, pixel_values=inputs["pixel_values"],
                max_new_tokens=action_dim, do_sample=False,
            )
            pred_ids = gen[0, -action_dim:].cpu().numpy()
            disc = model.vocab_size - pred_ids
            disc = np.clip(disc - 1, 0, model.bin_centers.shape[0] - 1)
            pred_norm = model.bin_centers[disc]
            stats = model.get_action_stats(unnorm_key)
            mask_arr = np.array(stats.get("mask", np.ones(action_dim, dtype=bool)))
            q99, q01 = np.array(stats["q99"]), np.array(stats["q01"])
            pred_cont = np.where(mask_arr, 0.5*(pred_norm+1)*(q99-q01)+q01, pred_norm)
            all_results[str(kr)].append(pred_cont)

        if pruner_state["norms_cache"] is not None:
            all_norms.append(pruner_state["norms_cache"].numpy())

    hook_handle.remove()

    gt = np.array(gt_all)
    curve = {}
    bl = None
    for kr in KEEP_RATIOS:
        preds = np.array(all_results[str(kr)])
        l1 = float(np.mean(np.abs(preds - gt)))
        mse = float(np.mean((preds - gt)**2))
        if kr == 1.0:
            bl = l1
        curve[str(kr)] = {"l1_loss": round(l1, 6), "mse_loss": round(mse, 6)}

    for kr in KEEP_RATIOS:
        curve[str(kr)]["normalized_l1"] = round(curve[str(kr)]["l1_loss"]/(bl+1e-8), 4)

    norms_flat = np.concatenate(all_norms, axis=0).flatten() if all_norms else np.array([])
    norm_stats = {
        "mean": round(float(np.mean(norms_flat)), 4),
        "std": round(float(np.std(norms_flat)), 4),
        "median": round(float(np.median(norms_flat)), 4),
        "min": round(float(np.min(norms_flat)), 4),
        "max": round(float(np.max(norms_flat)), 4),
        "n_tokens_per_image": int(all_norms[0].shape[-1]) if all_norms else 0,
    } if len(norms_flat) > 0 else {}

    print("\nPruning Curve:")
    for kr in KEEP_RATIOS:
        d = curve[str(kr)]
        print(f"  k={kr:.2f}: L1={d['l1_loss']:.4f}, norm_L1={d['normalized_l1']:.4f}, MSE={d['mse_loss']:.6f}")

    return curve, norm_stats


@torch.no_grad()
def entropy_analysis(model, processor, frames, device, max_frames=50):
    print(f"\n=== Attention Entropy Analysis (N={min(max_frames, len(frames))}) ===")

    n_layers = model.language_model.config.num_hidden_layers
    per_layer_ent = defaultdict(list)
    per_layer_vfrac = defaultdict(list)
    n_vis = None

    for fi, frame in enumerate(frames[:max_frames]):
        if fi % 10 == 0:
            print(f"  Frame {fi}/{min(max_frames, len(frames))}...")

        prompt = PROMPT_TEMPLATE.format(task=str(frame["task_desc"]).lower())
        inputs = processor(prompt, frame["image"]).to(device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        if input_ids[0, -1] != 29871:
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], device=device)], dim=1)

        try:
            out = model(input_ids=input_ids, pixel_values=inputs["pixel_values"],
                       attention_mask=torch.ones_like(input_ids), output_attentions=True)
        except RuntimeError:
            torch.cuda.empty_cache()
            continue

        if n_vis is None:
            vis = model.projector(model.vision_backbone(inputs["pixel_values"]))
            n_vis = vis.shape[1]
            print(f"  Vision tokens: {n_vis}")

        seq_len = out.attentions[0].shape[2]
        text_start = 1 + n_vis

        if text_start >= seq_len:
            del out; continue

        for li, attn in enumerate(out.attentions):
            vis_attn = attn[0, :, text_start:, 1:1+n_vis]
            full_attn = attn[0, :, text_start:, :]
            v_frac = (vis_attn.sum(dim=-1) / full_attn.sum(dim=-1).clamp(min=1e-10)).mean().item()
            vis_sum = vis_attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            vis_n = vis_attn / vis_sum
            ent = -(vis_n * torch.log(vis_n + 1e-10)).sum(dim=-1)
            ent_r = (ent / math.log(n_vis)).mean().item()
            per_layer_ent[li].append(ent_r)
            per_layer_vfrac[li].append(v_frac)

        del out
        torch.cuda.empty_cache()

    result = {
        "num_llm_layers": n_layers,
        "vision_tokens": n_vis,
        "per_layer_entropy_ratio": {},
        "per_layer_entropy_std": {},
        "per_layer_vision_attention_fraction": {},
    }
    all_e = []
    for L in range(n_layers):
        if per_layer_ent[L]:
            result["per_layer_entropy_ratio"][str(L)] = round(float(np.mean(per_layer_ent[L])), 4)
            result["per_layer_entropy_std"][str(L)] = round(float(np.std(per_layer_ent[L])), 4)
            result["per_layer_vision_attention_fraction"][str(L)] = round(float(np.mean(per_layer_vfrac[L])), 4)
            all_e.extend(per_layer_ent[L])
    if all_e:
        result["overall_entropy_ratio"] = round(float(np.mean(all_e)), 4)
        result["overall_entropy_std"] = round(float(np.std(all_e)), 4)

    print(f"\nOverall entropy: {result.get('overall_entropy_ratio', 'N/A')}")
    for L in range(n_layers):
        if str(L) in result["per_layer_entropy_ratio"]:
            print(f"  L{L:2d}: ent={result['per_layer_entropy_ratio'][str(L)]:.4f}, vfrac={result['per_layer_vision_attention_fraction'][str(L)]:.4f}")

    return result


def main():
    device = torch.device("cuda:0")
    model, processor, label = load_model(device)
    frames = load_libero_frames()

    curve, nstats = pruning_curve_analysis(model, processor, frames, device)
    entropy = entropy_analysis(model, processor, frames, device, max_frames=50)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pruning_out = {
        "model": label,
        "dataset": "LIBERO-Object (task_index 20-29)",
        "n_frames": len(frames),
        "keep_ratios": KEEP_RATIOS,
        "pruning_method": "L2_norm_zeroing",
        "pruning_curve": curve,
        "vision_token_norm_stats": nstats,
    }
    with open(OUT_DIR / "openvla_oft_pruning_curve.json", "w") as f:
        json.dump(pruning_out, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'openvla_oft_pruning_curve.json'}")

    entropy_out = {
        "model": label,
        "dataset": "LIBERO-Object (task_index 20-29)",
        "n_images": min(50, len(frames)),
        **entropy,
    }
    with open(OUT_DIR / "openvla_oft_attention_entropy_perlayer.json", "w") as f:
        json.dump(entropy_out, f, indent=2)
    print(f"Saved: {OUT_DIR / 'openvla_oft_attention_entropy_perlayer.json'}")

    print(f"\n{'='*60}")
    print(f"Done: {label}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
