"""SmolVLA MetaWorld eval with vision token pruning (baseline / l2norm / random / attention).

Adapted from eval_smolvla_pruning_official.py for MetaWorld benchmark.

Usage:
    python eval_smolvla_pruning_metaworld.py --pretrained_dir /path/to/ckpt --task easy --mode baseline
    python eval_smolvla_pruning_metaworld.py --pretrained_dir /path/to/ckpt --task pick-place-v3 --mode l2norm_prune --k_ratio 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as time_mod
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "./cache")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

METAWORLD_MAX_STEPS = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_dir", type=str, required=True,
                   help="Path to SmolVLA checkpoint fine-tuned on MetaWorld")
    p.add_argument("--task", type=str, default="medium",
                   help="Difficulty group (easy/medium/hard/very_hard) or comma-separated task names")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_horizon", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--mode", choices=["baseline", "l2norm_prune", "random_prune", "attention_prune"],
                   default="baseline")
    p.add_argument("--pruning_method", type=str, default="true_removal",
                   choices=["zero_mask", "true_removal"])
    p.add_argument("--k_ratio", type=float, default=0.95)
    return p.parse_args()


# --- Normalization helpers ---

def load_norm_stats(pretrained_dir, device):
    from safetensors.torch import load_file
    pre_path = Path(pretrained_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_path = Path(pretrained_dir) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre = load_file(str(pre_path))
    post = load_file(str(post_path))
    return {
        "state_mean": pre["observation.state.mean"].to(device),
        "state_std": pre["observation.state.std"].to(device),
        "action_mean": post["action.mean"].to(device),
        "action_std": post["action.std"].to(device),
        "eps": 1e-8,
    }


def normalize_state(state_tensor, norm_stats):
    return (state_tensor - norm_stats["state_mean"]) / (norm_stats["state_std"] + norm_stats["eps"])


def unnormalize_actions(actions, norm_stats):
    return actions * norm_stats["action_std"] + norm_stats["action_mean"]


# --- Obs to batch ---

def obs_to_batch(obs, task_text, tokenizer, camera_keys, device, norm_stats=None):
    """Convert MetaWorld observation to model input batch.
    
    MetaWorld obs dict:
      - "pixels": (H, W, 3) uint8 from corner2 camera
      - "agent_pos": (4,) float64 [eef_x, eef_y, eef_z, gripper]
    """
    batch = {}
    img = obs["pixels"]
    img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0)
    for cam_key in camera_keys:
        batch[cam_key] = img_t.to(device)

    state = np.asarray(obs["agent_pos"], dtype=np.float32)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
    if norm_stats is not None:
        state_t = normalize_state(state_t, norm_stats)
    batch["observation.state"] = state_t

    text = task_text if task_text.endswith("\n") else task_text + "\n"
    encoded = tokenizer(
        [text], padding="longest", max_length=48,
        truncation=True, return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"].to(device)
    batch["observation.language.attention_mask"] = encoded["attention_mask"].bool().to(device)
    return batch


# --- Pruning hooks (same as LIBERO version) ---

def find_connector(model):
    for name, module in model.named_modules():
        if name.endswith(".connector") or "multi_modal_projector" in name:
            log.info("Found connector: %s (%s)", name, type(module).__name__)
            return module
    return None


def register_l2norm_prune_hook(connector, k_ratio, method="true_removal"):
    call_count = [0]
    def hook_fn(module, input, output):
        tokens = output[0] if isinstance(output, tuple) else output
        if tokens.dim() != 3:
            return output
        B, N, D = tokens.shape
        k = max(1, int(N * k_ratio))
        if k >= N:
            return output
        norms = torch.norm(tokens.float(), dim=-1)
        _, topk_idx = torch.topk(norms, k, dim=-1)
        topk_idx_sorted = topk_idx.sort(dim=-1).values
        if call_count[0] < 2:
            log.info("L2norm hook: N=%d, k=%d (keep %.1f%%), method=%s",
                     N, k, k_ratio * 100, method)
        call_count[0] += 1
        if method == "true_removal":
            pruned = torch.gather(tokens, 1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D))
        else:
            mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
            mask.scatter_(1, topk_idx_sorted, True)
            pruned = tokens.clone()
            pruned[~mask] = 0
        if isinstance(output, tuple):
            return (pruned,) + output[1:]
        return pruned
    return connector.register_forward_hook(hook_fn)


def register_random_prune_hook(connector, k_ratio, method="true_removal"):
    def hook_fn(module, input, output):
        tokens = output[0] if isinstance(output, tuple) else output
        if tokens.dim() != 3:
            return output
        B, N, D = tokens.shape
        k = max(1, int(N * k_ratio))
        if k >= N:
            return output
        scores = torch.rand(1, N, device=tokens.device).expand(B, -1)
        _, topk_idx = torch.topk(scores, k, dim=-1)
        topk_idx_sorted = topk_idx.sort(dim=-1).values
        if method == "true_removal":
            pruned = torch.gather(tokens, 1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D))
        else:
            mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
            mask.scatter_(1, topk_idx_sorted, True)
            pruned = tokens.clone()
            pruned[~mask] = 0
        if isinstance(output, tuple):
            return (pruned,) + output[1:]
        return pruned
    return connector.register_forward_hook(hook_fn)


def register_attention_prune_hook(connector, policy, k_ratio, method="true_removal"):
    import math
    call_count = [0]

    def hook_fn(module, input, output):
        import torch.nn.functional as F
        tokens = output[0] if isinstance(output, tuple) else output
        if tokens.dim() != 3:
            return output
        B, N, D = tokens.shape
        k = max(1, int(N * k_ratio))
        if k >= N:
            return output

        vlm = policy.model.vlm_with_expert.get_vlm_model()
        lm = vlm.text_model
        position_ids = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        hidden = tokens
        all_attn = []

        with torch.no_grad():
            for layer_idx in range(min(2, len(lm.layers))):
                layer = lm.layers[layer_idx]
                normed = layer.input_layernorm(hidden)
                head_dim = layer.self_attn.head_dim
                q = layer.self_attn.q_proj(normed)
                k_proj = layer.self_attn.k_proj(normed)
                v = layer.self_attn.v_proj(normed)
                n_q_heads = q.shape[-1] // head_dim
                n_kv_heads = k_proj.shape[-1] // head_dim
                q = q.view(B, N, n_q_heads, head_dim).transpose(1, 2)
                k_proj = k_proj.view(B, N, n_kv_heads, head_dim).transpose(1, 2)
                v = v.view(B, N, n_kv_heads, head_dim).transpose(1, 2)
                cos, sin = lm.rotary_emb(v, position_ids)
                from transformers.models.llama.modeling_llama import rotate_half
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k_proj * cos) + (rotate_half(k_proj) * sin)
                if n_kv_heads != n_q_heads:
                    n_rep = n_q_heads // n_kv_heads
                    k_embed = k_embed.repeat_interleave(n_rep, dim=1)
                    v = v.repeat_interleave(n_rep, dim=1)
                attn_weights = torch.matmul(q_embed, k_embed.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                all_attn.append(attn_weights)
                attn_out = torch.matmul(attn_weights.to(v.dtype), v)
                attn_out = attn_out.transpose(1, 2).reshape(B, N, n_q_heads * head_dim)
                attn_out = layer.self_attn.o_proj(attn_out)
                hidden = hidden + attn_out
                normed2 = layer.post_attention_layernorm(hidden)
                mlp_out = layer.mlp(normed2)
                hidden = hidden + mlp_out

        avg_attn = torch.stack(all_attn).mean(dim=0).mean(dim=1)
        importance = avg_attn.sum(dim=1)
        _, topk_idx = torch.topk(importance, k, dim=-1)
        topk_idx_sorted = topk_idx.sort(dim=-1).values
        if call_count[0] < 2:
            log.info("Attention hook: N=%d, k=%d (keep %.1f%%), method=%s",
                     N, k, k_ratio * 100, method)
        call_count[0] += 1
        if method == "true_removal":
            pruned = torch.gather(tokens, 1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D))
        else:
            mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
            mask.scatter_(1, topk_idx_sorted, True)
            pruned = tokens.clone()
            pruned[~mask] = 0
        if isinstance(output, tuple):
            return (pruned,) + output[1:]
        return pruned
    return connector.register_forward_hook(hook_fn)


@torch.no_grad()
def predict_actions(policy, batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    actions = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state
    )
    original_action_dim = policy.config.action_feature.shape[0]
    return actions[0, :, :original_action_dim]


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denom
    return center, max(0.0, center - spread), min(1.0, center + spread)


def get_tokenizer(policy):
    model = policy.model
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model
    except ImportError:
        pass
    return model.vlm_with_expert.processor.tokenizer


def resolve_tasks(task_spec):
    """Resolve task specification to a list of (task_name, task_description) tuples."""
    from lerobot.envs.metaworld import DIFFICULTY_TO_TASKS, TASK_DESCRIPTIONS
    tasks = []
    for part in task_spec.split(","):
        part = part.strip()
        if part in DIFFICULTY_TO_TASKS:
            for t in DIFFICULTY_TO_TASKS[part]:
                tasks.append((t, TASK_DESCRIPTIONS[t]))
        elif part in TASK_DESCRIPTIONS:
            tasks.append((part, TASK_DESCRIPTIONS[part]))
        else:
            raise ValueError(f"Unknown task or difficulty group: {part}")
    return tasks


def make_metaworld_env(task_name):
    """Create a MetaWorld env with rendering enabled."""
    import metaworld
    mt1 = metaworld.MT1(task_name, seed=42)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env.reset()
    env._freeze_rand_vec = False
    return env


def get_obs(env, raw_obs):
    """Format raw MetaWorld observation into our expected dict."""
    img = env.render()
    img = np.flip(img, (0, 1)).copy()
    state = raw_obs[:4].astype(np.float32)
    return {"pixels": img, "agent_pos": state}


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    max_steps = args.max_steps or METAWORLD_MAX_STEPS

    if args.mode != "baseline":
        suffix = f"_{args.mode}_k{args.k_ratio}_{args.pruning_method}"
    else:
        suffix = "_baseline"
    output_path = Path(args.output or f"eval_results/metaworld_{args.task}{suffix}.json")

    log.info("Task: %s | Device: %s | Mode: %s | Eps/task: %d | Max steps: %d",
             args.task, device, args.mode, args.num_episodes, max_steps)
    if args.mode != "baseline":
        log.info("k_ratio: %.2f | pruning_method: %s", args.k_ratio, args.pruning_method)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    log.info("Loading checkpoint from %s ...", args.pretrained_dir)
    t0 = time_mod.time()
    policy = SmolVLAPolicy.from_pretrained(args.pretrained_dir)
    policy.to(device)
    policy.eval()
    log.info("Model loaded in %.1fs", time_mod.time() - t0)

    tokenizer = get_tokenizer(policy)
    norm_stats = load_norm_stats(args.pretrained_dir, device)

    camera_keys = [
        k for k, v in policy.config.input_features.items()
        if v.type.name == "VISUAL"
    ]
    log.info("Camera keys: %s", camera_keys)

    connector = None
    if args.mode != "baseline":
        connector = find_connector(policy.model)
        if connector is None:
            raise RuntimeError("Cannot find connector module")

    tasks = resolve_tasks(args.task)
    log.info("Running %d tasks", len(tasks))

    all_task_results = {}
    total_successes = 0
    total_episodes = 0

    t_start = time_mod.time()
    for task_idx, (task_name, task_desc) in enumerate(tasks):
        log.info("--- Task %d/%d: %s (%s) ---", task_idx + 1, len(tasks), task_name, task_desc)

        env = make_metaworld_env(task_name)
        task_successes = []
        task_lengths = []

        for ep in range(args.num_episodes):
            raw_obs, _ = env.reset(seed=args.seed + task_idx * 1000 + ep)
            obs = get_obs(env, raw_obs)
            action_chunk = None
            action_idx = 0
            success = False

            for step in range(max_steps):
                if action_chunk is None or action_idx >= args.action_horizon:
                    batch = obs_to_batch(obs, task_desc, tokenizer, camera_keys, device, norm_stats)

                    if args.mode == "l2norm_prune":
                        hook = register_l2norm_prune_hook(connector, args.k_ratio, args.pruning_method)
                        action_chunk = predict_actions(policy, batch)
                        hook.remove()
                    elif args.mode == "random_prune":
                        hook = register_random_prune_hook(connector, args.k_ratio, args.pruning_method)
                        action_chunk = predict_actions(policy, batch)
                        hook.remove()
                    elif args.mode == "attention_prune":
                        hook = register_attention_prune_hook(connector, policy, args.k_ratio, args.pruning_method)
                        action_chunk = predict_actions(policy, batch)
                        hook.remove()
                    else:
                        action_chunk = predict_actions(policy, batch)

                    action_chunk = unnormalize_actions(action_chunk, norm_stats)
                    action_idx = 0

                action = action_chunk[action_idx].cpu().numpy().astype(np.float32)
                action = np.clip(action, -1.0, 1.0)
                action_idx += 1

                raw_obs, reward, terminated, truncated, info = env.step(action)
                obs = get_obs(env, raw_obs)

                if info.get("success", 0.0) > 0:
                    success = True
                    break
                if terminated or truncated:
                    break

            ep_steps = step + 1
            task_successes.append(success)
            task_lengths.append(ep_steps)
            log.info("  Ep %d/%d | %s | steps=%d",
                     ep + 1, args.num_episodes,
                     "SUCCESS" if success else "FAIL", ep_steps)

        env.close()

        n_succ = sum(task_successes)
        sr = n_succ / args.num_episodes if args.num_episodes > 0 else 0.0
        all_task_results[task_name] = {
            "task_name": task_name,
            "task_description": task_desc,
            "success_count": n_succ,
            "total_episodes": args.num_episodes,
            "success_rate": round(sr, 4),
            "mean_steps": round(float(np.mean(task_lengths)), 1),
        }
        total_successes += n_succ
        total_episodes += args.num_episodes
        elapsed = time_mod.time() - t_start
        log.info("  %s SR: %d/%d (%.1f%%) | Total: %d/%d (%.1f%%) | Time: %.1fmin",
                  task_name, n_succ, args.num_episodes, sr * 100,
                 total_successes, total_episodes,
                 total_successes / total_episodes * 100, elapsed / 60)

    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    _, ci_lo, ci_hi = wilson_ci(total_successes, total_episodes)

    log.info("=" * 60)
    log.info("METAWORLD | TASK: %s | MODE: %s", args.task, args.mode)
    if args.mode != "baseline":
        log.info("PRUNING: %s | k=%.2f", args.pruning_method, args.k_ratio)
    log.info("Overall SR: %d/%d (%.1f%%)", total_successes, total_episodes, overall_sr * 100)
    log.info("95%% Wilson CI: [%.1f%%, %.1f%%]", ci_lo * 100, ci_hi * 100)
    for tname, tr in all_task_results.items():
        log.info("  %s: %.0f%% (%d/%d) -- %s",
                 tname, tr["success_rate"] * 100, tr["success_count"],
                 tr["total_episodes"], tr["task_description"])
    log.info("=" * 60)

    summary = {
        "benchmark": "metaworld",
        "task_spec": args.task,
        "mode": args.mode,
        "pruning_method": args.pruning_method if args.mode != "baseline" else "none",
        "k_ratio": args.k_ratio if args.mode != "baseline" else None,
        "pretrained_dir": args.pretrained_dir,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": round(overall_sr, 4),
        "wilson_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "num_episodes_per_task": args.num_episodes,
        "max_steps": max_steps,
        "action_horizon": args.action_horizon,
        "seed": args.seed,
        "per_task": all_task_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Results saved to %s", output_path)
    return summary


if __name__ == "__main__":
    main()
