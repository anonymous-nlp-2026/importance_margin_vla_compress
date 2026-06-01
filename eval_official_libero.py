"""Evaluate official SmolVLA-LIBERO checkpoint: unpruned baseline + random pruning."""
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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# --- Normalization helpers ---

def load_norm_stats(pretrained_dir, device):
    """Load mean/std normalization stats from checkpoint."""
    from safetensors.torch import load_file
    pre_path = Path(pretrained_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_path = Path(pretrained_dir) / "policy_postprocessor_step_1_unnormalizer_processor.safetensors"
    pre = load_file(str(pre_path))
    post = load_file(str(post_path))
    eps = 1e-8
    return {
        "state_mean": pre["observation.state.mean"].to(device),
        "state_std": pre["observation.state.std"].to(device),
        "action_mean": post["action.mean"].to(device),
        "action_std": post["action.std"].to(device),
        "eps": eps,
    }


def normalize_state(state_tensor, norm_stats):
    """Normalize observation state: (x - mean) / (std + eps)."""
    return (state_tensor - norm_stats["state_mean"]) / (norm_stats["state_std"] + norm_stats["eps"])


def unnormalize_actions(actions, norm_stats):
    """Unnormalize predicted actions: x * std + mean."""
    return actions * norm_stats["action_std"] + norm_stats["action_mean"]


def parse_args():
    p = argparse.ArgumentParser(description="Eval official SmolVLA on LIBERO")
    p.add_argument("--pretrained_dir", type=str, required=True,
                   help="Path to official lerobot-format checkpoint dir")
    p.add_argument("--suite", type=str, default="libero_object",
                   choices=list(SUITE_MAX_STEPS.keys()))
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_horizon", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--mode", choices=["baseline", "random_prune", "l2_norm"], default="baseline")
    p.add_argument("--pruning_method", type=str, default="true_removal",
                   choices=["zero_mask", "true_removal"])
    p.add_argument("--k_ratio", type=float, default=0.9,
                   help="Fraction of visual tokens to KEEP (0.9 = keep 90%%)")
    p.add_argument("--task_ids", type=str, default=None)
    return p.parse_args()


# --- State preprocessing ---

def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    q = quat.astype(np.float64)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den > 1e-10:
        angle = 2.0 * np.arccos(w)
        axis = q[:3] / den
        return (axis * angle).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def extract_state(obs: dict) -> np.ndarray:
    rs = obs["robot_state"]
    eef_pos = np.asarray(rs["eef"]["pos"], dtype=np.float32)
    eef_quat = np.asarray(rs["eef"]["quat"], dtype=np.float64)
    gripper_qpos = np.asarray(rs["gripper"]["qpos"], dtype=np.float32)
    eef_aa = quat_to_axis_angle(eef_quat)
    return np.concatenate([eef_pos, eef_aa, gripper_qpos])


def obs_to_batch(obs, task_text, tokenizer, camera_keys, device, norm_stats=None):
    batch = {}
    for cam_key in camera_keys:
        short_name = cam_key.split(".")[-1]
        img = obs["pixels"][short_name]
        img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0)
        img_t = torch.flip(img_t, dims=[2, 3])
        batch[cam_key] = img_t.to(device)

    state = extract_state(obs)
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


# --- Model helpers ---

def find_connector(model):
    for name, module in model.named_modules():
        if name.endswith(".connector") or "multi_modal_projector" in name:
            log.info("Found connector: %s (%s)", name, type(module).__name__)
            return module
    return None


def register_random_prune_hook(connector, k_ratio, method="true_removal"):
    def hook_fn(module, input, output):
        tokens = output[0] if isinstance(output, tuple) else output
        if tokens.dim() != 3:
            return output
        B, N, D = tokens.shape
        k = max(1, int(N * k_ratio))
        # Same random selection across batch for consistency
        scores = torch.rand(1, N, device=tokens.device).expand(B, -1)
        _, topk_idx = torch.topk(scores, k, dim=-1)

        if method == "true_removal":
            kept = torch.zeros(N, dtype=torch.bool, device=tokens.device)
            kept[topk_idx[0]] = True
            masked = tokens[:, kept, :]
        else:
            mask = torch.zeros(B, N, device=tokens.device)
            mask.scatter_(1, topk_idx, 1.0)
            masked = tokens * mask.unsqueeze(-1)

        if isinstance(output, tuple):
            return (masked,) + output[1:]
        return masked
    return connector.register_forward_hook(hook_fn)



def register_l2_norm_prune_hook(connector, k_ratio, method="true_removal"):
    def hook_fn(module, input, output):
        tokens = output[0] if isinstance(output, tuple) else output
        if tokens.dim() != 3:
            return output
        B, N, D = tokens.shape
        k = max(1, int(N * k_ratio))
        norms = tokens.norm(dim=-1)
        _, topk_idx = torch.topk(norms, k, dim=-1)
        if method == "true_removal":
            topk_idx_sorted, _ = torch.sort(topk_idx, dim=-1)
            masked = torch.gather(tokens, 1, topk_idx_sorted.unsqueeze(-1).expand(-1, -1, D))
        else:
            mask = torch.zeros(B, N, device=tokens.device)
            mask.scatter_(1, topk_idx, 1.0)
            masked = tokens * mask.unsqueeze(-1)
        if isinstance(output, tuple):
            return (masked,) + output[1:]
        return masked
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


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    max_steps = args.max_steps or SUITE_MAX_STEPS.get(args.suite, 400)
    suffix = f"_{args.mode}_k{args.k_ratio}_{args.pruning_method}" if args.mode != "baseline" else "_baseline"
    output_path = Path(args.output or f"eval_results/official_{args.suite}{suffix}.json")

    log.info("Suite: %s | Device: %s | Mode: %s | Episodes/task: %d | Max steps: %d",
             args.suite, device, args.mode, args.num_episodes, max_steps)
    if args.mode in ("random_prune", "l2_norm"):
        log.info("k_ratio: %.2f | pruning_method: %s", args.k_ratio, args.pruning_method)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load official checkpoint directly
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    log.info("Loading official checkpoint from %s ...", args.pretrained_dir)
    t0 = time_mod.time()
    policy = SmolVLAPolicy.from_pretrained(args.pretrained_dir)
    policy.to(device)
    policy.eval()
    log.info("Model loaded in %.1fs", time_mod.time() - t0)

    tokenizer = get_tokenizer(policy)

    # Load normalization stats
    norm_stats = load_norm_stats(args.pretrained_dir, device)
    log.info("Loaded normalization stats: state_mean=%s, action_mean=%s",
             norm_stats["state_mean"].tolist()[:3], norm_stats["action_mean"].tolist()[:3])

    # Detect camera keys from model config
    camera_keys = [
        k for k, v in policy.config.input_features.items()
        if v.type.name == "VISUAL"
    ]
    log.info("Camera keys from config: %s", camera_keys)

    connector = None
    if args.mode in ("random_prune", "l2_norm"):
        connector = find_connector(policy.model)
        if connector is None:
            raise RuntimeError("Cannot find connector module in model")

    # Load LIBERO
    from libero.libero import benchmark
    from lerobot.envs.libero import get_task_init_states, LiberoEnv

    bench_dict = benchmark.get_benchmark_dict()
    if args.suite not in bench_dict:
        raise ValueError(f"Unknown suite '{args.suite}'. Available: {sorted(bench_dict.keys())}")
    task_suite = bench_dict[args.suite]()
    n_tasks = len(task_suite.tasks)
    task_ids = list(range(n_tasks))
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]

    log.info("Suite '%s': %d tasks, evaluating %s", args.suite, n_tasks, task_ids)

    all_task_results = {}
    total_successes = 0
    total_episodes = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_desc = task.language
        log.info("--- Task %d: %s ---", task_id, task_desc)

        env = LiberoEnv(
            task_suite=task_suite,
            task_id=task_id,
            task_suite_name=args.suite,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            init_states=True,
            n_envs=args.num_episodes,
            episode_index=0,
            control_mode="relative",
        )

        task_successes = []
        task_lengths = []

        for ep in range(args.num_episodes):
            obs, info = env.reset(seed=args.seed + task_id * 1000 + ep)
            action_chunk = None
            action_idx = 0
            success = False

            for step in range(max_steps):
                if action_chunk is None or action_idx >= args.action_horizon:
                    batch = obs_to_batch(obs, task_desc, tokenizer, camera_keys, device, norm_stats)

                    if args.mode == "random_prune":
                        hook = register_random_prune_hook(
                            connector, args.k_ratio, args.pruning_method)
                        action_chunk = predict_actions(policy, batch)
                        hook.remove()
                        action_chunk = unnormalize_actions(action_chunk, norm_stats)
                    elif args.mode == "l2_norm":
                        hook = register_l2_norm_prune_hook(
                            connector, args.k_ratio, args.pruning_method)
                        action_chunk = predict_actions(policy, batch)
                        hook.remove()
                        action_chunk = unnormalize_actions(action_chunk, norm_stats)
                    else:
                        action_chunk = predict_actions(policy, batch)
                        action_chunk = unnormalize_actions(action_chunk, norm_stats)

                    action_idx = 0

                action = action_chunk[action_idx].cpu().numpy().astype(np.float32)
                action = np.clip(action, -1.0, 1.0)
                action_idx += 1

                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("is_success", False):
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
        all_task_results[task_id] = {
            "task_id": task_id,
            "task_description": task_desc,
            "success_count": n_succ,
            "total_episodes": args.num_episodes,
            "success_rate": round(sr, 4),
            "mean_steps": round(float(np.mean(task_lengths)), 1),
        }
        total_successes += n_succ
        total_episodes += args.num_episodes
        log.info("  Task %d SR: %d/%d (%.1f%%)", task_id, n_succ, args.num_episodes, sr * 100)

    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    _, ci_lo, ci_hi = wilson_ci(total_successes, total_episodes)

    log.info("=" * 60)
    log.info("SUITE: %s | MODE: %s", args.suite, args.mode)
    if args.mode in ("random_prune", "l2_norm"):
        log.info("PRUNING: %s | k_ratio: %.2f", args.pruning_method, args.k_ratio)
    log.info("Overall SR: %d/%d (%.1f%%)", total_successes, total_episodes, overall_sr * 100)
    log.info("95%% Wilson CI: [%.1f%%, %.1f%%]", ci_lo * 100, ci_hi * 100)
    for tid, tr in sorted(all_task_results.items()):
        log.info("  Task %d: %.0f%% (%d/%d) -- %s",
                 tid, tr["success_rate"] * 100, tr["success_count"],
                 tr["total_episodes"], tr["task_description"])
    log.info("=" * 60)

    summary = {
        "suite": args.suite,
        "mode": args.mode,
        "pruning_method": args.pruning_method if args.mode in ("random_prune", "l2_norm") else "none",
        "k_ratio": args.k_ratio if args.mode in ("random_prune", "l2_norm") else None,
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
