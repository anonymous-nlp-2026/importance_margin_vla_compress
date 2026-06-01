#!/usr/bin/env python
"""Measure SmolVLA per-layer attention entropy on LIBERO-Object images.

Fixed version: only samples from LIBERO-Object tasks (task_index 20-29),
with balanced sampling across all 10 tasks.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "./cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, ".")

import json
import math
import io
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize


def resize_with_pad(img_t, h, w, pad_value=0):
    _, oh, ow = img_t.shape
    scale = min(h / oh, w / ow)
    nh, nw = int(oh * scale), int(ow * scale)
    img_r = resize(img_t, [nh, nw], antialias=True)
    ph, pw = h - nh, w - nw
    img_p = F.pad(img_r, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2], value=pad_value)
    return img_p


LIBERO_OBJECT_TASKS = range(20, 30)
FRAMES_PER_TASK = 26


@torch.no_grad()
def main():
    device = torch.device("cuda:0")

    # ── Load SmolVLA ────────────────────────────────────────
    from _lerobot_compat import SmolVLAPolicy
    print("Loading SmolVLA model...")
    policy = SmolVLAPolicy.from_pretrained("./cache/lerobot/smolvla_base")
    policy.eval().to(device)

    flow = policy.model
    vlm = flow.vlm_with_expert
    tokenizer = vlm.processor.tokenizer

    n_layers = vlm.num_vlm_layers
    n_vis = 64 * 2
    self_every = vlm.self_attn_every_n_layers

    print(f"  {n_layers} VLM layers, self_attn_every={self_every}")
    print(f"  attention_mode={vlm.attention_mode}")

    # ── Load LIBERO-Object images (task_index 20-29) ────────
    snap = Path("./cache/lerobot/hub/"
                "datasets--HuggingFaceVLA--libero/snapshots/"
                "86958911c0f959db2bbbdb107eb3e17c5f9c798e")
    data_dir = snap / "data" / "chunk-000"

    tasks_df = pq.read_table(snap / "meta" / "tasks.parquet").to_pandas()
    task_names = tasks_df.index.tolist()

    # First pass: collect all row indices per task for LIBERO-Object
    per_task_rows = defaultdict(list)  # task_idx -> list of (parquet_file, row_idx)
    for pf in sorted(os.listdir(data_dir)):
        tbl = pq.read_table(data_dir / pf)
        task_col = tbl.column("task_index").to_pylist()
        for idx, tid in enumerate(task_col):
            if tid in LIBERO_OBJECT_TASKS:
                per_task_rows[tid].append((pf, idx))

    print(f"\nLIBERO-Object task coverage:")
    for tid in sorted(per_task_rows.keys()):
        tname = task_names[tid] if tid < len(task_names) else "unknown"
        print(f"  task {tid}: {len(per_task_rows[tid])} rows available -- {tname}")

    # Second pass: balanced sampling
    frames = []
    task_sample_counts = {}
    for tid in sorted(per_task_rows.keys()):
        rows = per_task_rows[tid]
        n_avail = len(rows)
        step = max(1, n_avail // FRAMES_PER_TASK)
        sampled = rows[::step][:FRAMES_PER_TASK]
        task_sample_counts[tid] = len(sampled)

        for pf, idx in sampled:
            tbl = pq.read_table(data_dir / pf)
            frames.append(dict(
                img1=Image.open(io.BytesIO(
                    tbl.column("observation.images.image")[idx]["bytes"].as_py()
                )).convert("RGB"),
                img2=Image.open(io.BytesIO(
                    tbl.column("observation.images.image2")[idx]["bytes"].as_py()
                )).convert("RGB"),
                task_idx=tbl.column("task_index")[idx].as_py(),
                state=torch.tensor(
                    tbl.column("observation.state")[idx].as_py(), dtype=torch.float32
                ),
            ))

    n_tasks = len({f["task_idx"] for f in frames})
    print(f"\n  {len(frames)} frames from {n_tasks} LIBERO-Object tasks")
    for tid in sorted(task_sample_counts.keys()):
        print(f"    task {tid}: {task_sample_counts[tid]} frames")

    # ── Attention hook ──────────────────────────────────────
    attn_caps = []

    def hooked_attn(attention_mask, batch_size, head_dim,
                    query_states, key_states, value_states):
        nh = vlm.num_attention_heads
        nkv = vlm.num_key_value_heads
        g = nh // nkv
        kv_len = key_states.shape[1]

        k_exp = (key_states[:, :, :, None, :]
                 .expand(batch_size, kv_len, nkv, g, head_dim)
                 .reshape(batch_size, kv_len, nh, head_dim))
        v_exp = (value_states[:, :, :, None, :]
                 .expand(batch_size, kv_len, nkv, g, head_dim)
                 .reshape(batch_size, kv_len, nh, head_dim))

        q = query_states.to(torch.float32).transpose(1, 2)
        k = k_exp.to(torch.float32).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) * head_dim**-0.5
        big_neg = torch.finfo(scores.dtype).min
        scores = torch.where(attention_mask[:, None, :, :], scores, big_neg)
        probs = F.softmax(scores, dim=-1)

        attn_caps.append(probs.detach().cpu())

        out = torch.matmul(probs.to(v_exp.dtype),
                           v_exp.permute(0, 2, 1, 3))
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, -1, nh * head_dim)
        return out

    vlm.eager_attention_forward = hooked_attn

    # ── Call-index → (layer, type) mapping ──────────────────
    call_info = []
    for L in range(n_layers):
        if L % self_every == 0:
            call_info.append((L, "self"))
        else:
            call_info.append((L, "cross_vlm"))
            call_info.append((L, "cross_expert"))
    n_calls = len(call_info)
    print(f"  {n_calls} attention calls expected per forward")

    # ── Process frames ──────────────────────────────────────
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    per_layer_ent = defaultdict(list)
    per_layer_vfrac = defaultdict(list)

    for fi, frame in enumerate(frames):
        if fi % 50 == 0:
            print(f"  Processing {fi}/{len(frames)}...")

        attn_caps.clear()

        img1 = resize_with_pad(to_tensor(frame["img1"]).to(device), 512, 512) * 2 - 1
        img2 = resize_with_pad(to_tensor(frame["img2"]).to(device), 512, 512) * 2 - 1

        emb1 = vlm.embed_image(img1.unsqueeze(0))
        emb2 = vlm.embed_image(img2.unsqueeze(0))
        D = emb1.shape[-1]
        emb1 = emb1 * math.sqrt(D)
        emb2 = emb2 * math.sqrt(D)

        tid = frame["task_idx"]
        tname = task_names[tid] if tid < len(task_names) else "pick up the object"
        enc = tokenizer(tname, return_tensors="pt",
                        padding="max_length", max_length=48, truncation=True)
        lang_ids = enc["input_ids"].to(device)
        lang_mask = enc["attention_mask"].to(device).bool()

        lang_emb = vlm.embed_language_tokens(lang_ids) * math.sqrt(D)

        st = frame["state"].to(device)
        st_padded = F.pad(st, (0, 32 - st.shape[0]))
        state_emb = flow.state_proj(st_padded.unsqueeze(0)).unsqueeze(1)

        prefix = torch.cat([emb1, emb2, lang_emb, state_emb], dim=1)
        prefix_mask = torch.cat([
            torch.ones(1, 64, dtype=torch.bool, device=device),
            torch.ones(1, 64, dtype=torch.bool, device=device),
            lang_mask,
            torch.ones(1, 1, dtype=torch.bool, device=device),
        ], dim=1)
        prefix_att = torch.zeros(1, prefix.shape[1], dtype=torch.long, device=device)
        prefix_att[0, -1] = 1

        cs = policy.config.chunk_size
        x_t = torch.zeros(1, cs, policy.config.max_action_dim, device=device)
        t = torch.rand(1, device=device)
        suffix, suffix_mask, suffix_att = flow.embed_suffix(x_t, t)

        pad_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        att_mask = torch.cat([prefix_att, suffix_att.long()], dim=1)
        att_2d = make_att_2d_masks(pad_mask, att_mask)
        pos_ids = torch.cumsum(pad_mask.long(), dim=1) - 1

        vlm.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            fill_kv_cache=False,
        )

        if len(attn_caps) != n_calls:
            print(f"  WARNING: got {len(attn_caps)} calls, expected {n_calls}, skipping")
            continue

        prefix_len = prefix.shape[1]

        for ci, (layer_idx, atype) in enumerate(call_info):
            probs = attn_caps[ci]

            if atype == "self":
                q_s, q_e = n_vis, prefix_len
                vis_attn = probs[:, :, q_s:q_e, :n_vis]
                full_attn = probs[:, :, q_s:q_e, :]
            elif atype == "cross_vlm":
                q_s, q_e = n_vis, prefix_len
                vis_attn = probs[:, :, q_s:q_e, :n_vis]
                full_attn = probs[:, :, q_s:q_e, :]
            else:
                continue

            v_frac = (vis_attn.sum(dim=-1) / full_attn.sum(dim=-1).clamp(min=1e-10)).mean().item()

            vis_sum = vis_attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            vis_norm = vis_attn / vis_sum
            ent = -(vis_norm * torch.log(vis_norm + 1e-10)).sum(dim=-1)
            ent_ratio = (ent / math.log(n_vis)).mean().item()
            ent_std = (ent / math.log(n_vis)).std().item()

            per_layer_ent[layer_idx].append(ent_ratio)
            per_layer_vfrac[layer_idx].append(v_frac)

    # ── Aggregate ───────────────────────────────────────────
    results = {
        "model": "SmolVLA-0.5B",
        "dataset": "LIBERO-Object (task_index 20-29)",
        "n_images": len(frames),
        "n_tasks": n_tasks,
        "tasks_sampled": {str(tid): task_sample_counts[tid] for tid in sorted(task_sample_counts.keys())},
        "num_vlm_layers": n_layers,
        "vision_tokens": n_vis,
        "per_layer_entropy_ratio": {},
        "per_layer_entropy_std": {},
        "per_layer_vision_attention_fraction": {},
    }

    all_ent = []
    for L in range(n_layers):
        vals = per_layer_ent[L]
        results["per_layer_entropy_ratio"][str(L)] = round(float(np.mean(vals)), 4)
        results["per_layer_entropy_std"][str(L)] = round(float(np.std(vals)), 4)
        results["per_layer_vision_attention_fraction"][str(L)] = round(
            float(np.mean(per_layer_vfrac[L])), 4)
        all_ent.extend(vals)

    results["overall_entropy_ratio"] = round(float(np.mean(all_ent)), 4)
    results["overall_entropy_std"] = round(float(np.std(all_ent)), 4)

    out_path = "./eval_results/smolvla_attention_entropy_perlayer_fixed.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Overall entropy ratio: {results['overall_entropy_ratio']:.4f} +/- {results['overall_entropy_std']:.4f}")
    print("\nPer-layer breakdown:")
    for L in range(n_layers):
        e = results["per_layer_entropy_ratio"][str(L)]
        s = results["per_layer_entropy_std"][str(L)]
        v = results["per_layer_vision_attention_fraction"][str(L)]
        print(f"  L{L:2d}: entropy={e:.4f} +/- {s:.4f}  vis_frac={v:.4f}")


if __name__ == "__main__":
    main()
