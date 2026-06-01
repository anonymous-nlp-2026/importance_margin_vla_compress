"""OpenVLA-OFT Vision Token Pruning Evaluation on LIBERO.

Evaluates OpenVLA-OFT checkpoints with optional vision token pruning.
Prunes projected vision tokens (after 3-layer MLP projector) by L2 norm or randomly.

Architecture:
  - Vision: DinoSigLIP fused backbone → 3-layer MLP projector → LLM token space
  - Action: L1 regression MLP head (continuous), K=8 action chunking (open-loop)
  - 7D actions: [delta_pos(3) + delta_rot(3) + grip(1)]
  - 8D proprio: [eef_pos(3) + eef_axisangle(3) + grip_qpos(2)]
  - Dual cameras: agentview + wrist (both 256x256 → resized to 224x224)

Input:  OFT checkpoint path, LIBERO suite, pruning params
Output: Per-task SR + overall SR (JSON + console log)

Dependencies: transformers, torch, numpy, Pillow, timm, libero

Usage:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python eval_oft_libero.py \
        --checkpoint ./openvla-oft-libero-object \
        --suite libero_object \
        --prune_mode l2norm \
        --keep_ratio 0.5 \
        --num_episodes 50
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import time as time_mod
from collections import deque
from functools import wraps
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
PROPRIO_DIM = 8
OPENVLA_IMAGE_SIZE = 224
CROP_SCALE = 0.9

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

STANDARD_TASKS = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_cream_cheese_in_the_bowl",
        "put_the_wine_bottle_on_the_rack",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "turn_on_the_stove",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA-OFT pruning eval on LIBERO")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to OFT checkpoint dir (local)")
    p.add_argument("--suite", type=str, default="libero_object",
                   choices=list(SUITE_MAX_STEPS.keys()))
    p.add_argument("--prune_mode", type=str, default="none",
                   choices=["none", "l2norm", "random"],
                   help="Vision token pruning strategy")
    p.add_argument("--keep_ratio", type=float, default=1.0,
                   help="Fraction of vision tokens to keep (1.0 = no pruning)")
    p.add_argument("--num_episodes", type=int, default=50,
                   help="Episodes per task")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: auto)")
    p.add_argument("--task_ids", type=str, default=None,
                   help="Comma-separated task IDs (default: all)")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--num_open_loop_steps", type=int, default=NUM_ACTIONS_CHUNK,
                   help="Actions to execute open-loop per policy query")
    p.add_argument("--no_proprio", action="store_true",
                   help="Disable proprio input (debug only)")
    p.add_argument("--no_wrist", action="store_true",
                   help="Disable wrist camera (single image input)")
    p.add_argument("--standard_only", action="store_true",
                   help="Use only standard 10 tasks per suite (bypass LIBERO-plus variants)")
    return p.parse_args()


# Inline Action Head / Projector (matching OFT repo class defs)

class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for blk in self.mlp_resnet_blocks:
            x = blk(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        B = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(B, NUM_ACTIONS_CHUNK, -1)
        return self.model(x)


class ProprioProjector(nn.Module):
    def __init__(self, llm_dim, proprio_dim=PROPRIO_DIM):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)

    def forward(self, proprio):
        return self.fc2(self.act_fn1(self.fc1(proprio)))


# Vision Token Pruning

def install_vision_pruning(model, keep_ratio, prune_mode):
    if prune_mode == "none" or keep_ratio >= 1.0:
        log.info("No vision pruning (mode=%s, keep_ratio=%.2f)", prune_mode, keep_ratio)
        return

    original_forward = model.projector.forward

    @wraps(original_forward)
    def pruned_forward(img_patches):
        projected = original_forward(img_patches)
        B, N, D = projected.shape
        k = max(1, int(keep_ratio * N))
        if k >= N:
            return projected
        if prune_mode == "l2norm":
            norms = projected.norm(dim=-1)
            _, idx = norms.topk(k, dim=-1)
            idx = idx.sort(dim=-1).values
            return projected.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        elif prune_mode == "random":
            perm = torch.randperm(N)[:k].sort().values.to(projected.device)
            return projected[:, perm, :]
        return projected

    model.projector.forward = pruned_forward
    log.info("Vision pruning installed: mode=%s, keep_ratio=%.2f", prune_mode, keep_ratio)


# Image / Observation Utilities

def get_libero_image(obs):
    img = obs["agentview_image"]
    return img[::-1, ::-1].copy()


def get_libero_wrist_image(obs):
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1].copy()


def resize_image_for_policy(img_np, size=OPENVLA_IMAGE_SIZE):
    """Resize with JPEG roundtrip to match official TF-based training pipeline."""
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    pil = Image.open(buf)
    pil.load()
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)


def center_crop_image(image, crop_scale=CROP_SCALE):
    """Center crop matching official OFT implementation (TF crop_and_resize)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    w, h = image.size
    scale = math.sqrt(crop_scale)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    left = (w - new_w) / 2.0
    top = (h - new_h) / 2.0
    image = image.crop((
        int(math.floor(left)), int(math.floor(top)),
        int(math.floor(left)) + new_w, int(math.floor(top)) + new_h,
    ))
    image = image.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.BILINEAR)
    return image


def prepare_image(img_np):
    """Full image preprocessing: rotate already done -> resize -> center crop -> PIL."""
    img_resized = resize_image_for_policy(img_np, OPENVLA_IMAGE_SIZE)
    pil = center_crop_image(img_resized, CROP_SCALE)
    return pil


def quat2axisangle(quat):
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if den < 1e-7:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(abs(w))
    return (quat[:3] * angle / den).astype(np.float32)


def extract_proprio(obs):
    eef_pos = obs["robot0_eef_pos"]
    eef_aa = quat2axisangle(obs["robot0_eef_quat"])
    grip = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, eef_aa, grip]).astype(np.float32)


# Action Post-Processing

def normalize_gripper_action(action, binarize=True):
    a = action.copy()
    a[..., -1] = 2.0 * a[..., -1] - 1.0
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def invert_gripper_action(action):
    a = action.copy()
    a[..., -1] *= -1.0
    return a


def normalize_proprio(proprio, norm_stats):
    """Normalize proprio with mask-based bounds_q99 (matching official OFT repo)."""
    mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)))
    q01 = np.array(norm_stats["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["q99"], dtype=np.float32)
    return np.clip(
        np.where(mask, 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0, proprio),
        -1.0, 1.0,
    )


# Wilson CI

def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denom
    return center, max(0.0, center - spread), min(1.0, center + spread)


# Model Loading

def _load_state_dict_strip_ddp(path):
    """Load checkpoint state dict and strip DDP module. prefix if present."""
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def load_model_and_components(checkpoint_path, device, load_in_8bit=False, load_in_4bit=False):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    ckpt = Path(checkpoint_path)
    log.info("Loading OFT model from %s", ckpt)

    model = AutoModelForVision2Seq.from_pretrained(
        str(ckpt),
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    if not load_in_8bit and not load_in_4bit:
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)

    llm_dim = model.config.text_config.hidden_size
    log.info("LLM hidden dim: %d", llm_dim)

    # Load action head
    ah_files = sorted(ckpt.glob("action_head--*_checkpoint.pt"))
    assert ah_files, f"No action_head checkpoint in {ckpt}"
    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim)
    action_head.load_state_dict(_load_state_dict_strip_ddp(ah_files[-1]))
    action_head = action_head.to(torch.bfloat16).to(device).eval()
    log.info("Action head: %s", ah_files[-1].name)

    # Load proprio projector
    pp_files = sorted(ckpt.glob("proprio_projector--*_checkpoint.pt"))
    assert pp_files, f"No proprio_projector checkpoint in {ckpt}"
    proprio_proj = ProprioProjector(llm_dim=llm_dim)
    proprio_proj.load_state_dict(_load_state_dict_strip_ddp(pp_files[-1]))
    proprio_proj = proprio_proj.to(torch.bfloat16).to(device).eval()
    log.info("Proprio projector: %s", pp_files[-1].name)

    # Load norm stats
    stats_path = ckpt / "dataset_statistics.json"
    assert stats_path.exists(), f"dataset_statistics.json not found in {ckpt}"
    with open(stats_path) as f:
        model.norm_stats = json.load(f)
    log.info("Norm stats loaded (%d keys)", len(model.norm_stats))

    return model, processor, action_head, proprio_proj


# OFT-Style Action Prediction
# The HF checkpoint predict_action uses discrete tokens (base OpenVLA).
# OFT requires L1 regression via an external action head.

@torch.inference_mode()
def predict_actions_oft(
    model, processor, img_pil, wrist_pil, task_desc,
    proprio_normed, action_head, proprio_proj, unnorm_key, device,
    use_proprio=True, use_wrist=True,
):
    """OFT action prediction: single forward pass + L1 regression action head.

    Returns: np.ndarray (NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)
    """
    prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"

    # Tokenize prompt
    inputs = processor(prompt, img_pil).to(device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    primary_pixels = inputs["pixel_values"]

    # Ensure trailing empty token (29871) to match training
    if input_ids[0, -1].item() != 29871:
        input_ids = torch.cat(
            [input_ids, torch.tensor([[29871]], device=device, dtype=input_ids.dtype)],
            dim=1,
        )
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
            dim=1,
        )

    n_prompt = input_ids.shape[-1] - 1
    n_act_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK  # 56

    # Extend with action placeholder tokens + stop token (matching official OFT pipeline)
    STOP_TOKEN_ID = 2  # '</s>'
    input_ids_ext = torch.cat(
        [input_ids,
         torch.ones(1, n_act_tokens, device=device, dtype=input_ids.dtype),
         torch.full((1, 1), STOP_TOKEN_ID, device=device, dtype=input_ids.dtype)],
        dim=1,
    )
    attn_ext = torch.cat(
        [attention_mask, torch.ones(1, n_act_tokens + 1, device=device, dtype=attention_mask.dtype)],
        dim=1,
    )

    # Embed all tokens, then zero-out action positions (keep stop token embedding)
    input_embeds = model.get_input_embeddings()(input_ids_ext)
    input_embeds[:, -(n_act_tokens + 1):-1, :] = 0

    # Vision features
    patches_primary = model.vision_backbone(primary_pixels)

    if use_wrist and wrist_pil is not None:
        wrist_inputs = processor(prompt, wrist_pil).to(device, dtype=torch.bfloat16)
        patches_wrist = model.vision_backbone(wrist_inputs["pixel_values"])
        all_patches = torch.cat([patches_primary, patches_wrist], dim=1)
    else:
        all_patches = patches_primary

    projected = model.projector(all_patches)

    # Proprio token
    if use_proprio and proprio_proj is not None and proprio_normed is not None:
        proprio_t = torch.tensor(
            proprio_normed, dtype=projected.dtype, device=device
        )
        proprio_emb = proprio_proj(proprio_t.unsqueeze(0)).unsqueeze(1)
        projected = torch.cat([projected, proprio_emb], dim=1)

    n_patches = projected.shape[1]

    # Build multimodal sequence: [first_token | patches | rest_tokens]
    multimodal = torch.cat(
        [input_embeds[:, :1, :], projected, input_embeds[:, 1:, :]],
        dim=1,
    )
    patch_attn = torch.ones(1, n_patches, device=device, dtype=attn_ext.dtype)
    multimodal_attn = torch.cat(
        [attn_ext[:, :1], patch_attn, attn_ext[:, 1:]],
        dim=1,
    )

    # LLM forward
    outputs = model.language_model(
        inputs_embeds=multimodal,
        attention_mask=multimodal_attn,
        output_hidden_states=True,
        return_dict=True,
    )

    # Extract action hidden states
    last_h = outputs.hidden_states[-1]
    a_start = n_patches + n_prompt
    a_end = a_start + n_act_tokens
    action_h = last_h[:, a_start:a_end, :]

    # Action head -> continuous actions
    pred = action_head.predict_action(action_h)
    pred = pred.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().numpy()

    # Unnormalize (bounds_q99)
    stats = model.norm_stats[unnorm_key]["action"]
    mask = np.array(stats.get("mask", np.ones(ACTION_DIM, dtype=bool)))
    q99 = np.array(stats["q99"], dtype=np.float32)
    q01 = np.array(stats["q01"], dtype=np.float32)
    actions = np.where(
        mask,
        0.5 * (pred + 1) * (q99 - q01 + 1e-8) + q01,
        pred,
    )

    return actions


# Main Evaluation

def run_evaluation(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    max_steps = args.max_steps or SUITE_MAX_STEPS.get(args.suite, 300)
    output_path = Path(
        args.output
        or f"eval_results/oft_prune_{args.prune_mode}_k{args.keep_ratio}_{args.suite}.json"
    )

    log.info(
        "Suite=%s | Device=%s | prune=%s keep=%.2f | Ep/task=%d | max_steps=%d",
        args.suite, device, args.prune_mode, args.keep_ratio, args.num_episodes, max_steps,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, processor, action_head, proprio_proj = load_model_and_components(
        args.checkpoint, device, args.load_in_8bit, args.load_in_4bit,
    )

    # Resolve unnorm key
    unnorm_key = args.suite
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, (
        f"Unnorm key '{unnorm_key}' not found. Available: {list(model.norm_stats.keys())}"
    )
    log.info("unnorm_key=%s", unnorm_key)
    proprio_norm_stats = model.norm_stats[unnorm_key]["proprio"]

    # Measure vision tokens
    with torch.no_grad():
        dummy_pix = torch.randn(1, 6, 224, 224, device=device, dtype=torch.bfloat16)
        dummy_patches = model.vision_backbone(dummy_pix)
        n_patches_per_img = dummy_patches.shape[1]
        del dummy_pix, dummy_patches
    n_images = 2 if not args.no_wrist else 1
    total_vis = n_patches_per_img * n_images
    log.info("Vision tokens: %d/img x %d imgs = %d total", n_patches_per_img, n_images, total_vis)

    install_vision_pruning(model, args.keep_ratio, args.prune_mode)

    # LIBERO setup
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils

    if args.standard_only and args.suite in STANDARD_TASKS:
        task_names = STANDARD_TASKS[args.suite]
        bddl_dir = get_libero_path("bddl_files")
        init_dir = get_libero_path("init_states")
        task_id_list = list(range(len(task_names)))
        log.info("Standard-only mode: %d tasks", len(task_names))

        _std_tasks = []
        _std_init_states = {}
        for i, tname in enumerate(task_names):
            bddl_path = os.path.join(bddl_dir, args.suite, f"{tname}.bddl")
            info = BDDLUtils.get_problem_info(bddl_path)
            _std_tasks.append({
                "language": info["language_instruction"],
                "bddl_path": bddl_path,
                "name": tname,
            })
            init_path = os.path.join(init_dir, args.suite, f"{tname}.pruned_init")
            _std_init_states[i] = torch.load(init_path, weights_only=False)
    else:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.suite]()
        n_tasks = task_suite.n_tasks
        task_id_list = (
            [int(x) for x in args.task_ids.split(",")]
            if args.task_ids
            else list(range(n_tasks))
        )
        _std_tasks = None
        _std_init_states = None

    log.info("Tasks: %s (%d total)", task_id_list, len(task_id_list))

    eval_start = time_mod.time()
    all_results = {}
    total_succ, total_ep = 0, 0

    for task_id in task_id_list:
        if _std_tasks is not None:
            task_desc = _std_tasks[task_id]["language"]
            task_bddl = _std_tasks[task_id]["bddl_path"]
            init_states = _std_init_states[task_id]
        else:
            task = task_suite.get_task(task_id)
            task_desc = task.language
            task_bddl = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file,
            )
            init_states = task_suite.get_task_init_states(task_id)
        log.info("--- Task %d: %s ---", task_id, task_desc)

        task_successes = []

        for ep in range(args.num_episodes):
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl, camera_heights=256, camera_widths=256,
            )
            env.seed(0)
            obs = env.reset()
            obs = env.set_init_state(init_states[ep % init_states.shape[0]])

            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

            action_queue = deque(maxlen=args.num_open_loop_steps)
            success = False

            for step_idx in range(max_steps):
                if len(action_queue) == 0:
                    img_pil = prepare_image(get_libero_image(obs))
                    wrist_pil = (
                        prepare_image(get_libero_wrist_image(obs))
                        if not args.no_wrist
                        else None
                    )
                    proprio = extract_proprio(obs)
                    proprio_normed = normalize_proprio(proprio, proprio_norm_stats)

                    actions = predict_actions_oft(
                        model, processor, img_pil, wrist_pil, task_desc,
                        proprio_normed if not args.no_proprio else None,
                        action_head, proprio_proj, unnorm_key, device,
                        use_proprio=not args.no_proprio,
                        use_wrist=not args.no_wrist,
                    )

                    for i in range(min(args.num_open_loop_steps, actions.shape[0])):
                        a = normalize_gripper_action(actions[i], binarize=True)
                        a = invert_gripper_action(a)
                        action_queue.append(a)

                action = action_queue.popleft()
                obs, reward, done, info = env.step(action.tolist())

                if done:
                    success = True
                    break

            task_successes.append(success)
            log.info(
                "  Ep %d/%d | %s | steps=%d",
                ep + 1, args.num_episodes,
                "OK" if success else "FAIL", step_idx + 1,
            )
            env.close()

        n_s = sum(task_successes)
        n_t = len(task_successes)
        sr = n_s / n_t if n_t > 0 else 0.0
        all_results[task_id] = {
            "task_id": task_id,
            "task_description": task_desc,
            "success_count": n_s,
            "total_episodes": n_t,
            "success_rate": round(sr, 4),
        }
        total_succ += n_s
        total_ep += n_t
        log.info("  Task %d SR: %d/%d (%.1f%%)", task_id, n_s, n_t, sr * 100)

    elapsed = time_mod.time() - eval_start
    overall_sr = total_succ / total_ep if total_ep > 0 else 0.0
    _, ci_lo, ci_hi = wilson_ci(total_succ, total_ep)

    log.info("=" * 60)
    log.info("SUITE: %s | prune: %s k=%.2f", args.suite, args.prune_mode, args.keep_ratio)
    log.info("Overall SR: %d/%d (%.1f%%)", total_succ, total_ep, overall_sr * 100)
    log.info("95%% Wilson CI: [%.1f%%, %.1f%%]", ci_lo * 100, ci_hi * 100)
    for tid, tr in sorted(all_results.items()):
        log.info(
            "  Task %d: %.0f%% (%d/%d) -- %s",
            tid, tr["success_rate"] * 100, tr["success_count"],
            tr["total_episodes"], tr["task_description"],
        )
    log.info("Elapsed: %.1f min", elapsed / 60)
    log.info("=" * 60)

    summary = {
        "model": "openvla-oft",
        "checkpoint": str(args.checkpoint),
        "prune_mode": args.prune_mode,
        "suite": args.suite,
        "keep_ratio": args.keep_ratio,
        "n_vision_tokens_original": total_vis,
        "n_vision_tokens_kept": max(1, int(args.keep_ratio * total_vis)),
        "total_episodes": total_ep,
        "total_successes": total_succ,
        "overall_success_rate": round(overall_sr, 4),
        "wilson_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "num_episodes_per_task": args.num_episodes,
        "max_steps": max_steps,
        "num_open_loop_steps": args.num_open_loop_steps,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "per_task": all_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Results -> %s", output_path)

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
