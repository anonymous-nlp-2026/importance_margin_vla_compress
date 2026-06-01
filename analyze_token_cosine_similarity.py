#!/usr/bin/env python3
"""Token Cosine Similarity Analysis for 6 VLA Models.

For each model:
1. Load one LIBERO-Object frame
2. Extract post-connector vision token embeddings via forward pass
3. Compute NxN pairwise cosine similarity matrix
4. Report statistics and generate 2x3 heatmap

Validates: low L2-norm CV → high inter-token cosine similarity (homogenization),
           high CV → low similarity (differentiation).
This explains WHY CV predicts pruning tolerance.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_token_cosine_similarity.py [--models pi05,openvla,smolvla]
"""

import argparse
import gc
import io
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "./cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(".")
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
sys.path.insert(0, str(PROJECT_DIR))

DEVICE = torch.device("cuda:0")

MODEL_CONFIGS = [
    {"name": "pi0",     "label": "π0",      "cv": 0.038, "n_tok": 256,
     "connector": "linear proj.",
     "path": "./pi0_libero_finetuned"},
    {"name": "pi05",    "label": "π0.5",    "cv": 0.055, "n_tok": 768,
     "connector": "linear proj.",
     "path": "./pi05_libero_finetuned"},
    {"name": "xvla",    "label": "X-VLA",   "cv": 0.151, "n_tok": 50,
     "connector": "linear+BART",
     "path": ["./models/xvla-libero",
              "./X-VLA-Libero"]},
    {"name": "openvla", "label": "OpenVLA", "cv": 0.215, "n_tok": 256,
     "connector": "3-layer MLP",
     "path": "./openvla-libero-object"},
    {"name": "oft",     "label": "OFT",     "cv": 0.219, "n_tok": 512,
     "connector": "3-layer MLP",
     "path": "./openvla-oft-libero-object"},
    {"name": "smolvla", "label": "SmolVLA", "cv": 0.314, "n_tok": 128,
     "connector": "pix-shuffle",
     "path": "./cache/lerobot/smolvla_base"},
]


# =====================================================================
# Utilities
# =====================================================================

def resolve_path(path_spec):
    if isinstance(path_spec, list):
        for p in path_spec:
            if Path(p).exists():
                return p
        return None
    return path_spec if Path(path_spec).exists() else None


def load_libero_frame():
    import pyarrow.parquet as pq
    snap = Path(
        "./cache/lerobot/hub/"
        "datasets--HuggingFaceVLA--libero/snapshots/"
        "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
    )
    data_dir = snap / "data" / "chunk-000"
    tasks_df = pq.read_table(snap / "meta" / "tasks.parquet").to_pandas()

    for pf in sorted(os.listdir(data_dir)):
        tbl = pq.read_table(data_dir / pf)
        for idx, tid in enumerate(tbl.column("task_index").to_pylist()):
            if 20 <= tid < 30:
                img1 = Image.open(io.BytesIO(
                    tbl.column("observation.images.image")[idx]["bytes"].as_py()
                )).convert("RGB")
                img2 = Image.open(io.BytesIO(
                    tbl.column("observation.images.image2")[idx]["bytes"].as_py()
                )).convert("RGB")
                task = str(tasks_df.index[tid]) if tid < len(tasks_df) else f"task_{tid}"
                print(f"Loaded LIBERO-Object frame: task_idx={tid}")
                return {"img1": img1, "img2": img2, "task": task}
    raise RuntimeError("No LIBERO-Object frame found")


def cosine_sim_matrix(tokens):
    normed = F.normalize(tokens.float(), dim=-1)
    return (normed @ normed.T).cpu().numpy()


def pairwise_stats(sim):
    n = sim.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p = sim[mask]
    return {
        "mean": round(float(np.mean(p)), 4),
        "std":  round(float(np.std(p)), 4),
        "min":  round(float(np.min(p)), 4),
        "max":  round(float(np.max(p)), 4),
        "median": round(float(np.median(p)), 4),
        "n_tokens": n,
    }


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# Per-model token extractors
# =====================================================================

def extract_pi05(frame, model_path):
    """π0.5: SigLIP → linear projection → ×√d scaling.
    Hook: paligemma.model.multi_modal_projector output × sqrt(hidden_size).
    Processes 2 camera images → 2×256 = 512 tokens."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from torchvision.transforms.functional import to_tensor, resize

    policy = PI05Policy.from_pretrained(model_path)
    policy = policy.to(DEVICE).eval()

    pg = policy.model.paligemma_with_expert.paligemma
    hidden_size = pg.config.text_config.hidden_size
    vt = pg.model.vision_tower
    proj = pg.model.multi_modal_projector

    all_tok = []
    for pil_img in [frame["img1"], frame["img2"]]:
        img = resize(to_tensor(pil_img), [224, 224], antialias=True)
        img = img.unsqueeze(0).to(DEVICE, dtype=torch.float32)
        with torch.no_grad():
            feats = vt(img).last_hidden_state
            projected = proj(feats) * (hidden_size ** 0.5)
            all_tok.append(projected[0])

    tokens = torch.cat(all_tok, dim=0)
    del policy
    return tokens


def extract_pi0(frame, model_path):
    """π0: Same arch as π0.5. SigLIP → linear → ×√d.
    Hook: paligemma.model.multi_modal_projector output × sqrt(hidden_size).
    Single camera → 256 tokens."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from torchvision.transforms.functional import to_tensor, resize

    policy = PI0Policy.from_pretrained(model_path)
    policy = policy.to(DEVICE).eval()

    pg = policy.model.paligemma_with_expert.paligemma
    hidden_size = pg.config.text_config.hidden_size
    vt = pg.model.vision_tower
    proj = pg.model.multi_modal_projector

    img = resize(to_tensor(frame["img1"]), [224, 224], antialias=True)
    img = img.unsqueeze(0).to(DEVICE, dtype=torch.float32)

    with torch.no_grad():
        feats = vt(img).last_hidden_state
        projected = proj(feats) * (hidden_size ** 0.5)

    del policy
    return projected[0]


def extract_xvla(frame, model_path):
    """X-VLA: Florence2 image encoder → linear projection → 12-layer BART encoder.
    Hook: model.forward_vlm()["vlm_features"] — output after full VLM processing.
    Yields ~50 tokens."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from torchvision.transforms.functional import to_tensor, resize
    from transformers import AutoTokenizer

    policy = XVLAPolicy.from_pretrained(model_path)
    policy = policy.to(DEVICE).eval()
    model = policy.model
    config = policy.config

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, padding_side=config.tokenizer_padding_side
    )

    def resize_with_pad(img_t, h, w):
        _, oh, ow = img_t.shape
        scale = min(h / oh, w / ow)
        nh, nw = int(oh * scale), int(ow * scale)
        img_r = resize(img_t, [nh, nw], antialias=True)
        ph, pw = h - nh, w - nw
        return F.pad(img_r, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2], value=0)

    img1 = resize_with_pad(to_tensor(frame["img1"]), 224, 224).unsqueeze(0).to(DEVICE)
    img2 = resize_with_pad(to_tensor(frame["img2"]), 224, 224).unsqueeze(0).to(DEVICE)

    task_text = frame["task"]
    if not task_text.endswith("\n"):
        task_text += "\n"
    enc = tokenizer([task_text], padding="longest", max_length=64,
                    truncation=True, return_tensors="pt")

    batch = {
        "observation.images.image": img1,
        "observation.images.image2": img2,
        "observation.state": torch.zeros(1, 8).to(DEVICE),
        "observation.language.tokens": enc["input_ids"].to(DEVICE),
        "action": torch.zeros(1, config.chunk_size, 7).to(DEVICE),
    }
    inputs = policy._build_model_inputs(batch)

    with torch.no_grad():
        out = model.forward_vlm(
            inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
        )

    tokens = out["vlm_features"][0]
    del policy
    return tokens


def extract_openvla(frame, model_path):
    """OpenVLA: DinoSigLIP fused dual backbone → 3-layer MLP projector.
    Hook: vla.projector output (called on vision_backbone features).
    Single image → 256 tokens."""
    openvla_deps = "./openvla_deps"
    if Path(openvla_deps).exists():
        sys.path.insert(0, openvla_deps)

    hf_dir = "./openvla-repo/prismatic/extern/hf"
    if not Path(hf_dir).exists():
        raise FileNotFoundError(f"OpenVLA HF code not found at {hf_dir}")
    init_path = os.path.join(hf_dir, "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()
    sys.path.insert(0, "./openvla-repo/prismatic/extern")

    from hf.configuration_prismatic import OpenVLAConfig
    from hf.modeling_prismatic import OpenVLAForActionPrediction
    from hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoTokenizer

    config = OpenVLAConfig.from_pretrained(model_path)
    vla = OpenVLAForActionPrediction.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()

    image_proc = PrismaticImageProcessor.from_pretrained(model_path)
    processor = PrismaticProcessor(
        image_processor=image_proc,
        tokenizer=AutoTokenizer.from_pretrained(model_path),
    )

    prompt = f"In: What action should the robot take to {frame['task'].lower()}?\nOut:"
    inputs = processor(prompt, frame["img1"]).to(DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        patches = vla.vision_backbone(inputs["pixel_values"])
        projected = vla.projector(patches)

    tokens = projected[0]
    del vla
    return tokens


def extract_oft(frame, model_path):
    """OFT: DinoSigLIP fused backbone → 3-layer MLP projector.
    Hook: vla.projector output (on concatenated dual-camera backbone features).
    Two cameras → 2×256 = 512 tokens."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    vla = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    lora_dir = os.path.join(model_path, "lora_adapter")
    if os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")):
        from peft import PeftModel
        vla = PeftModel.from_pretrained(vla, lora_dir)
        vla = vla.merge_and_unload()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    prompt = f"In: What action should the robot take to {frame['task'].lower()}?\nOut:"
    inp1 = processor(prompt, frame["img1"]).to(DEVICE, dtype=torch.bfloat16)
    inp2 = processor(prompt, frame["img2"]).to(DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        p1 = vla.vision_backbone(inp1["pixel_values"])
        p2 = vla.vision_backbone(inp2["pixel_values"])
        projected = vla.projector(torch.cat([p1, p2], dim=1))

    tokens = projected[0]
    del vla
    return tokens


def extract_smolvla(frame, model_path):
    """SmolVLA: SigLIP → pixel-shuffle 4×4 connector.
    Hook: connector module output (register_forward_hook).
    Two cameras × 64 tokens = 128 tokens."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_path)
    # Remap camera keys to match our batch format (image, image2 instead of camera1/2/3)
    from lerobot.configs.types import FeatureType, PolicyFeature
    new_inputs = {}
    for key, feat in policy.config.input_features.items():
        if feat.type == FeatureType.VISUAL:
            continue
        new_inputs[key] = feat
    new_inputs["observation.images.image"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=[3, 256, 256])
    new_inputs["observation.images.image2"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=[3, 256, 256])
    policy.config.input_features = new_inputs
    policy = policy.to(DEVICE).eval()

    # Get tokenizer from model internals
    _model = policy.model
    try:
        from peft import PeftModel
        if isinstance(_model, PeftModel):
            _model = _model.base_model.model
    except ImportError:
        pass
    tokenizer = _model.vlm_with_expert.processor.tokenizer

    # Find connector module
    connector = None
    for name, module in policy.named_modules():
        if name.endswith(".connector"):
            connector = module
            break
    if connector is None:
        raise RuntimeError("Cannot find .connector module in SmolVLA")

    # Hook to capture post-connector tokens
    captured = []
    def hook(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if t.dim() == 3:
            captured.append(t.detach())
    handle = connector.register_forward_hook(hook)

    # Build batch
    img1 = np.asarray(frame["img1"])
    img2 = np.asarray(frame["img2"])
    img1_t = torch.from_numpy(img1.copy()).permute(2, 0, 1).float() / 255.0
    img2_t = torch.from_numpy(img2.copy()).permute(2, 0, 1).float() / 255.0
    img1_t = torch.flip(img1_t, dims=[1, 2])
    img2_t = torch.flip(img2_t, dims=[1, 2])

    # Tokenize task text
    task_text = frame["task"]
    if not task_text.endswith("\n"):
        task_text += "\n"
    encoded = tokenizer(
        [task_text], padding="longest", max_length=48,
        truncation=True, return_tensors="pt",
    )

    # Use state dim from model config
    state_shape = policy.config.input_features.get(
        "observation.state", None)
    state_dim = state_shape.shape[0] if state_shape else 8
    batch = {
        "observation.images.image": img1_t.unsqueeze(0).to(DEVICE),
        "observation.images.image2": img2_t.unsqueeze(0).to(DEVICE),
        "observation.state": torch.zeros(1, state_dim).to(DEVICE),
        "observation.language.tokens": encoded["input_ids"].to(DEVICE),
        "observation.language.attention_mask": encoded["attention_mask"].bool().to(DEVICE),
    }

    # Full forward pass to trigger connector hook
    with torch.no_grad():
        try:
            policy.predict_action_chunk(batch)
        except Exception:
            pass  # OK if action prediction fails, we only need the hook output

    handle.remove()

    if not captured:
        raise RuntimeError("Connector hook did not capture any tokens")
    tokens = torch.cat(captured, dim=1)[0]

    del policy
    return tokens


# =====================================================================
# Plotting
# =====================================================================

def plot_heatmaps(results, output_path):
    n = len(results)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    items = list(results.items())
    for i, (name, data) in enumerate(items):
        ax = axes_flat[i]
        sim = data["sim_matrix"]
        stats = data["stats"]

        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="equal",
                       interpolation="nearest")

        ax.set_title(
            f'{data["label"]}  (CV={data["cv"]:.3f}, {data["connector"]})\n'
            f'mean cos={stats["mean"]:.3f}, std={stats["std"]:.3f}',
            fontsize=9, pad=6,
        )
        ax.set_xlabel(f'{stats["n_tokens"]} tokens', fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    cbar = fig.colorbar(im, ax=axes_flat[:n].tolist(), shrink=0.7,
                        label="Cosine Similarity", pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Post-Connector Vision Token Pairwise Cosine Similarity\n"
        "Low CV → high similarity (homogenization)  |  High CV → low similarity (differentiation)",
        fontsize=11, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.93])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    png_path = str(output_path).replace(".pdf", ".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")
    print(f"Figure saved: {png_path}")


# =====================================================================
# Main
# =====================================================================

EXTRACTORS = {
    "pi0":     extract_pi0,
    "pi05":    extract_pi05,
    "xvla":    extract_xvla,
    "openvla": extract_openvla,
    "oft":     extract_oft,
    "smolvla": extract_smolvla,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all available)")
    args = parser.parse_args()

    selected = set(args.models.split(",")) if args.models else None
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LIBERO-Object frame...")
    frame = load_libero_frame()

    results = {}

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        if selected and name not in selected:
            continue

        model_path = resolve_path(cfg["path"])
        if model_path is None:
            p = cfg["path"] if isinstance(cfg["path"], str) else cfg["path"][0]
            print(f"\n[SKIP] {cfg['label']}: not found at {p}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {cfg['label']} (CV={cfg['cv']}, {cfg['connector']})")
        print(f"  Path: {model_path}")

        try:
            tokens = EXTRACTORS[name](frame, model_path)
            sim = cosine_sim_matrix(tokens)
            stats = pairwise_stats(sim)

            results[name] = {
                "label":     cfg["label"],
                "cv":        cfg["cv"],
                "connector": cfg["connector"],
                "expected_tokens": cfg["n_tok"],
                "sim_matrix": sim,
                "stats":     stats,
            }

            print(f"  Tokens: {tokens.shape[0]} × {tokens.shape[1]}D")
            print(f"  Mean cos sim: {stats['mean']:.4f}  std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}  Max: {stats['max']:.4f}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()

        free_gpu()

    if not results:
        print("\nNo models processed successfully.")
        return

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<10} {'CV':>6} {'#Tok':>5} {'MeanCosSim':>11} "
          f"{'Std':>7} {'Min':>7} {'Max':>7}")
    print("-" * 80)
    for name, d in results.items():
        s = d["stats"]
        print(f"{d['label']:<10} {d['cv']:>6.3f} {s['n_tokens']:>5d} "
              f"{s['mean']:>11.4f} {s['std']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f}")

    # Correlation check
    if len(results) >= 3:
        cvs = [d["cv"] for d in results.values()]
        means = [d["stats"]["mean"] for d in results.values()]
        corr = np.corrcoef(cvs, means)[0, 1]
        print(f"\nCorrelation(CV, MeanCosSim) = {corr:.4f}")
        expected = "✓ confirmed" if corr < -0.5 else "✗ unexpected"
        print(f"  Expected negative correlation: {expected}")

    # Plot
    output_pdf = ARTIFACTS_DIR / "token_cosine_similarity.pdf"
    plot_heatmaps(results, output_pdf)

    # Save stats JSON (without numpy arrays)
    stats_out = {}
    for name, d in results.items():
        stats_out[name] = {
            "label": d["label"],
            "cv": d["cv"],
            "connector": d["connector"],
            "expected_tokens": d["expected_tokens"],
            **d["stats"],
        }
    json_path = ARTIFACTS_DIR / "token_cosine_similarity_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Stats JSON: {json_path}")

    # Save raw similarity matrices as npz
    npz_path = ARTIFACTS_DIR / "token_cosine_similarity_matrices.npz"
    np.savez_compressed(str(npz_path),
                        **{name: d["sim_matrix"] for name, d in results.items()})
    print(f"Raw matrices: {npz_path}")


if __name__ == "__main__":
    main()
