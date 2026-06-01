#!/usr/bin/env python
"""Token importance visualization for paper.

Generates three figures showing L2norm-based vision token importance:
  1. fig_importance_heatmap — importance overlay + kept/pruned at k=0.85, k=0.95
  2. fig_importance_comparison — L2norm vs Random token retention
  3. fig_importance_histogram — importance distribution with cliff zone

Input:  pi0.5 model + LIBERO parquet dataset
Output: PDF + PNG (300 dpi) -> artifacts/figures/

Architecture: SigLIP 224x224, patch=14 -> 16x16 = 256 vision tokens/image
Importance: L2 norm of projected embeddings (vision_tower -> multi_modal_projector)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "./cache"
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

import sys
import argparse
sys.path.insert(0, "./lerobot/src")

import io
import types
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.disable = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom
import pyarrow.parquet as pq
from PIL import Image

# ── Paths ──
MODEL_PATH = "./pi05_libero_finetuned"
LIBERO_SNAP = Path(
    "./cache/lerobot/hub/"
    "datasets--HuggingFaceVLA--libero/snapshots/"
    "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
)
DATA_DIR = LIBERO_SNAP / "data" / "chunk-000"
OUT_DIR = Path("./artifacts/figures")

# ── Architecture ──
PATCH_SIZE = 14
IMG_SIZE = 224
GRID = IMG_SIZE // PATCH_SIZE  # 16
N_TOKENS = GRID * GRID          # 256

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "mathtext.fontset": "cm",
})


# ═══════════════════════════════════════
# Data
# ═══════════════════════════════════════

def load_task_names():
    df = pq.read_table(LIBERO_SNAP / "meta" / "tasks.parquet").to_pandas()
    return dict(zip(df["task_index"], df.index))


def load_frames(task_indices, n_per_task=3):
    """Load sample frames from LIBERO parquet for given task indices."""
    task_set = set(task_indices)
    per_task = defaultdict(list)

    parquet_files = sorted(os.listdir(DATA_DIR))
    for i, pf in enumerate(parquet_files):
        if i % 80 == 0:
            print(f"  Scanning: {i}/{len(parquet_files)}")
        tbl = pq.read_table(DATA_DIR / pf, columns=["task_index"])
        for idx, tid in enumerate(tbl.column("task_index").to_pylist()):
            if tid in task_set:
                per_task[tid].append((pf, idx))
        enough = all(len(per_task.get(t, [])) >= n_per_task * 3 for t in task_set)
        if enough and len(per_task) >= len(task_set):
            break

    task_names = load_task_names()
    frames = []
    for tid in sorted(per_task.keys()):
        rows = per_task[tid]
        step = max(1, len(rows) // n_per_task)
        for pf, idx in rows[::step][:n_per_task]:
            tbl = pq.read_table(DATA_DIR / pf)
            raw = tbl.column("observation.images.image")[idx]
            img = Image.open(io.BytesIO(raw["bytes"].as_py())).convert("RGB")
            frames.append({
                "image": img,
                "task_index": tid,
                "task_desc": task_names.get(tid, f"task_{tid}"),
            })

    print(f"  -> {len(frames)} frames, {len(set(f['task_index'] for f in frames))} tasks")
    return frames


# ═══════════════════════════════════════
# Model
# ═══════════════════════════════════════

def load_model(device):
    """Load pi0.5 for vision token importance extraction."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(MODEL_PATH)
    vlm = policy.model.paligemma_with_expert

    def patched_embed_image(self, image):
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        out = self.paligemma.model.vision_tower(image)
        return self.paligemma.model.multi_modal_projector(out.last_hidden_state)

    vlm.embed_image = types.MethodType(patched_embed_image, vlm)
    policy.to(device).eval()
    print(f"  Model: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M params")
    return policy


@torch.no_grad()
def get_importance(policy, img_pil, device):
    """Extract 16x16 L2 norm importance map for one image."""
    img = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # SigLIP normalization
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    features = policy.model.paligemma_with_expert.embed_image(t)
    norms = torch.norm(features[0].float(), dim=-1).cpu().numpy()
    return norms.reshape(GRID, GRID)


# ═══════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════

def _draw_kept_pruned(ax, img_arr, norms_2d, keep_ratio):
    """Show image with kept tokens bright, pruned tokens dimmed."""
    n_keep = max(1, int(N_TOKENS * keep_ratio))
    threshold = np.sort(norms_2d.flatten())[::-1][n_keep - 1]
    kept = norms_2d >= threshold
    kept_up = zoom(kept.astype(float), IMG_SIZE / GRID, order=0)

    display = img_arr.astype(float) / 255.0
    alpha = np.where(kept_up > 0.5, 1.0, 0.15)
    display = display * alpha[..., np.newaxis]
    ax.imshow(np.clip(display, 0, 1))

    for i in range(1, GRID):
        pos = i * PATCH_SIZE
        ax.axhline(pos, color="white", lw=0.3, alpha=0.25)
        ax.axvline(pos, color="white", lw=0.3, alpha=0.25)

    return N_TOKENS - n_keep


def _save(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {path}")


def _add_label(ax, label, color="white"):
    ax.text(0.03, 0.94, f"({label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=color, va="top",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.5))


# ═══════════════════════════════════════
# Figure 1: Heatmap
# ═══════════════════════════════════════

def make_fig_heatmap(frames, norms_list, path):
    """Original + continuous heatmap + k=0.95 + k=0.85, two rows."""
    n = min(2, len(frames))
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.2 * n + 0.3))
    if n == 1:
        axes = axes[np.newaxis, :]

    im = None
    for r in range(n):
        img = np.array(frames[r]["image"].resize((IMG_SIZE, IMG_SIZE)))
        nm = norms_list[r]
        nm_n = (nm - nm.min()) / (nm.max() - nm.min() + 1e-8)

        axes[r, 0].imshow(img)
        axes[r, 0].set_title("Original", fontsize=11)
        axes[r, 0].axis("off")

        nm_up = zoom(nm_n, IMG_SIZE / GRID, order=1)
        axes[r, 1].imshow(img)
        im = axes[r, 1].imshow(nm_up, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        axes[r, 1].set_title("Token Importance", fontsize=11)
        axes[r, 1].axis("off")

        for ci, k in enumerate([0.95, 0.85]):
            ax = axes[r, 2 + ci]
            np_ = _draw_kept_pruned(ax, img, nm, k)
            ax.set_title(f"$k$={k:.2f} ({np_} pruned)", fontsize=11)
            ax.axis("off")

        desc = frames[r]["task_desc"]
        if len(desc) > 50:
            desc = desc[:47] + "..."
        axes[r, 0].text(-0.05, 0.5, desc, transform=axes[r, 0].transAxes,
                        fontsize=8, va="center", ha="right", style="italic")

    labels = "abcdefgh"
    for r in range(n):
        for c in range(4):
            _add_label(axes[r, c], labels[r * 4 + c])

    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label="Normalized L2 Norm")

    plt.tight_layout(rect=[0.07, 0, 0.91, 1.0])
    _save(fig, path)


# ═══════════════════════════════════════
# Figure 2: L2norm vs Random
# ═══════════════════════════════════════

def make_fig_comparison(frame, norms_2d, path, keep_ratio=0.85):
    """L2norm vs Random retention side-by-side."""
    img = np.array(frame["image"].resize((IMG_SIZE, IMG_SIZE)))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    _draw_kept_pruned(axes[1], img, norms_2d, keep_ratio)
    axes[1].set_title(f"L2-Norm ($k$={keep_ratio})", fontsize=11)
    axes[1].axis("off")

    # Random baseline
    rng = np.random.RandomState(42)
    n_keep = max(1, int(N_TOKENS * keep_ratio))
    rand_mask = np.zeros(N_TOKENS, dtype=bool)
    rand_mask[rng.choice(N_TOKENS, n_keep, replace=False)] = True
    rand_up = zoom(rand_mask.reshape(GRID, GRID).astype(float),
                   IMG_SIZE / GRID, order=0)
    disp = img.astype(float) / 255.0
    disp = disp * np.where(rand_up > 0.5, 1.0, 0.15)[..., np.newaxis]
    axes[2].imshow(np.clip(disp, 0, 1))
    for i in range(1, GRID):
        p = i * PATCH_SIZE
        axes[2].axhline(p, color="white", lw=0.3, alpha=0.25)
        axes[2].axvline(p, color="white", lw=0.3, alpha=0.25)
    axes[2].set_title(f"Random ($k$={keep_ratio})", fontsize=11)
    axes[2].axis("off")

    for i, lb in enumerate("abc"):
        _add_label(axes[i], lb)

    # Legend
    bright = mpatches.Patch(facecolor="#FFFFFF", edgecolor="#333333", label="Retained")
    dim = mpatches.Patch(facecolor="#2A2A2A", edgecolor="#333333", label="Pruned")
    fig.legend(handles=[bright, dim], loc="lower center", ncol=2,
              fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, path)


# ═══════════════════════════════════════
# Figure 3: Histogram
# ═══════════════════════════════════════

def make_fig_histogram(all_norms_2d, path):
    """Importance distribution with color-coded cliff zone."""
    flat = all_norms_2d.flatten()
    sorted_desc = np.sort(flat)[::-1]
    n = len(sorted_desc)

    t85 = sorted_desc[int(n * 0.85) - 1]
    t90 = sorted_desc[int(n * 0.90) - 1]
    t95 = sorted_desc[int(n * 0.95) - 1]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))

    # Color-coded histogram
    bins = np.linspace(flat.min(), flat.max(), 80)
    counts, edges = np.histogram(flat, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    colors = []
    for bc in centers:
        if bc >= t85:
            colors.append("#27AE60")  # green = always kept
        elif bc >= t90:
            colors.append("#E74C3C")  # red = cliff zone
        else:
            colors.append("#95A5A6")  # gray = pruned at k>=0.90

    ax.bar(centers, counts, width=width, color=colors,
           edgecolor="white", lw=0.3, zorder=2)

    # Threshold lines
    for val, lab, col, ls in [
        (t95, "$k$=0.95", "#2980B9", ":"),
        (t90, "$k$=0.90", "#F39C12", "--"),
        (t85, "$k$=0.85", "#C0392B", "--"),
    ]:
        ax.axvline(val, color=col, ls=ls, lw=1.5, zorder=3, label=f"{lab}")

    # Cliff zone bracket annotation
    mid = (t85 + t90) / 2
    ymax = counts.max()
    ax.annotate("", xy=(t90, ymax * 0.95), xytext=(t85, ymax * 0.95),
                arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.5))
    ax.text(mid, ymax * 0.98, "Cliff Zone", ha="center", va="bottom",
            fontsize=9, color="#E74C3C", fontweight="bold")

    # Legend with custom entries
    green_p = mpatches.Patch(color="#27AE60", label="Kept at $k$=0.85")
    red_p = mpatches.Patch(color="#E74C3C", label="Cliff zone ($k$=0.85–0.90)")
    gray_p = mpatches.Patch(color="#95A5A6", label="Pruned at $k$=0.90")
    ax.legend(handles=[green_p, red_p, gray_p], loc="upper right",
              fontsize=8, framealpha=0.9)

    ax.set_xlabel("Token L2 Norm (after projection)")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_comparison", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading model ===")
    policy = load_model(device)

    print("\n=== Loading frames for visualization ===")
    # Task 22: "pick up the cream cheese..." (Object)
    # Task 3: "turn on the stove..." (Long-horizon)
    vis_frames = load_frames([22, 3], n_per_task=5)

    print("\n=== Extracting importance ===")
    vis_norms = [get_importance(policy, f["image"], device) for f in vis_frames]

    obj_idx = [i for i, f in enumerate(vis_frames) if f["task_index"] == 22]
    long_idx = [i for i, f in enumerate(vis_frames) if f["task_index"] == 3]

    best = lambda idxs: max(idxs, key=lambda i: vis_norms[i].std()) if idxs else 0
    bi_obj = best(obj_idx)
    bi_long = best(long_idx)

    for i in [bi_obj, bi_long]:
        f = vis_frames[i]
        nm = vis_norms[i]
        print(f"  Selected: task={f['task_index']} "
              f"norm_range=[{nm.min():.1f}, {nm.max():.1f}] std={nm.std():.2f}")

    if not args.only_comparison:
        # ── Figure 1 ──
        print("\n=== Figure 1: Heatmap ===")
        make_fig_heatmap(
            [vis_frames[bi_obj], vis_frames[bi_long]],
            [vis_norms[bi_obj], vis_norms[bi_long]],
            OUT_DIR / "fig_importance_heatmap.pdf",
        )

    # ── Figure 2 ──
    print("\n=== Figure 2: Comparison ===")
    make_fig_comparison(
        vis_frames[bi_obj], vis_norms[bi_obj],
        OUT_DIR / "fig_importance_comparison.pdf",
        keep_ratio=0.85,
    )

    if not args.only_comparison:
        # ── Figure 3 ──
        print("\n=== Figure 3: Histogram ===")
        print("Loading broader set for statistics...")
        stat_frames = load_frames(list(range(20, 30)), n_per_task=3)
        print("Extracting importance for histogram...")
        stat_norms = np.stack(
            [get_importance(policy, f["image"], device) for f in stat_frames]
        )
        print(f"  Aggregated: {stat_norms.shape[0]} frames x {N_TOKENS} tokens")
        make_fig_histogram(stat_norms, OUT_DIR / "fig_importance_histogram.pdf")

    del policy
    torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"Done. All figures in {OUT_DIR}")
    for name in ["fig_importance_heatmap", "fig_importance_comparison",
                  "fig_importance_histogram"]:
        for ext in [".pdf", ".png"]:
            p = OUT_DIR / (name + ext)
            if p.exists():
                print(f"  {p} ({p.stat().st_size // 1024} KB)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
