"""LIBERO Success Rate Evaluation for SmartCompress (VisionTokenCompressor) models.

Usage:
    python eval_smart_compress_libero.py \
        --n_queries 32 \
        --checkpoint_path checkpoints/smart_compress_m32/checkpoint.pt \
        --suite libero_object \
        --num_episodes 50 \
        --gpu 0
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

import numpy as np
import torch

ARTIFACTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACTS_DIR))

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


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO eval for SmartCompress models")
    p.add_argument("--n_queries", type=int, required=True,
                   help="M: number of compressed vision tokens")
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to checkpoint.pt")
    p.add_argument("--suite", type=str, default="libero_object",
                   choices=list(SUITE_MAX_STEPS.keys()))
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_horizon", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--task_ids", type=str, default=None)
    return p.parse_args()


def load_model(args, device):
    from train_smart_compress import load_smolvla_policy, load_config
    from smart_compress_module import SmartCompressWrapper

    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt:
        config = ckpt["config"]
        log.info("Using config from checkpoint")
    else:
        config = load_config("configs/smart_compress.yaml")
        log.info("No config in checkpoint, falling back to yaml")

    policy = load_smolvla_policy(config, device)

    compress_config = {"num_queries": args.n_queries}
    wrapper = SmartCompressWrapper(policy, compress_config=compress_config)
    wrapper.to(device)

    missing, unexpected = wrapper.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        log.warning("Missing keys (%d): %s", len(missing), missing[:10])
    if unexpected:
        log.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:10])
    if not missing and not unexpected:
        log.info("Checkpoint loaded with strict match (no missing/unexpected keys)")
    wrapper.eval()

    log.info("SmartCompressWrapper loaded: M=%d, compressor params=%d",
             args.n_queries,
             sum(p.numel() for p in wrapper.compressor.parameters()))

    return wrapper, config


def quat_to_axis_angle(quat):
    q = quat.astype(np.float64)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den > 1e-10:
        angle = 2.0 * np.arccos(w)
        axis = q[:3] / den
        return (axis * angle).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def extract_state(obs):
    rs = obs["robot_state"]
    eef_pos = np.asarray(rs["eef"]["pos"], dtype=np.float32)
    eef_quat = np.asarray(rs["eef"]["quat"], dtype=np.float64)
    gripper_qpos = np.asarray(rs["gripper"]["qpos"], dtype=np.float32)
    eef_aa = quat_to_axis_angle(eef_quat)
    return np.concatenate([eef_pos, eef_aa, gripper_qpos])


def obs_to_batch(obs, task_text, tokenizer, camera_keys, device):
    batch = {}
    for cam_key in camera_keys:
        short_name = cam_key.split(".")[-1]
        img = obs["pixels"][short_name]
        img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0)
        img_t = torch.flip(img_t, dims=[2, 3])
        batch[cam_key] = img_t.to(device)

    state = extract_state(obs)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    text = task_text if task_text.endswith("\n") else task_text + "\n"
    encoded = tokenizer(
        [text], padding="longest", max_length=48,
        truncation=True, return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"].to(device)
    batch["observation.language.attention_mask"] = encoded["attention_mask"].bool().to(device)

    return batch


@torch.no_grad()
def predict_actions_compressed(wrapper, batch):
    actions = wrapper.forward_inference(batch)
    action_dim = wrapper.policy.config.action_feature.shape[0]
    return actions[0, :, :action_dim]


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denom
    return center, max(0.0, center - spread), min(1.0, center + spread)


def run_evaluation(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    max_steps = args.max_steps or SUITE_MAX_STEPS.get(args.suite, 400)
    output_path = Path(args.output or f"eval_results/smart_compress_m{args.n_queries}_{args.suite}.json")

    log.info("Suite: %s | Device: %s | M=%d | Episodes/task: %d | Max steps: %d",
             args.suite, device, args.n_queries, args.num_episodes, max_steps)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info("Loading model from %s ...", args.checkpoint_path)
    t0 = time_mod.time()
    wrapper, config = load_model(args, device)
    log.info("Model loaded in %.1fs", time_mod.time() - t0)

    from train import get_tokenizer_from_policy
    tokenizer = get_tokenizer_from_policy(wrapper.policy)

    ds_cfg = config.get("dataset", {})
    camera_keys = ds_cfg.get("features", {}).get("cameras", [
        "observation.images.image", "observation.images.image2"
    ])

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

    log.info("Suite '%s' has %d tasks, evaluating %d: %s",
             args.suite, n_tasks, len(task_ids), task_ids)

    all_task_results = {}
    total_successes = 0
    total_episodes = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_desc = task.language
        log.info("--- Task %d: %s ---", task_id, task_desc)

        init_states = get_task_init_states(task_suite, task_id)

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
                    batch = obs_to_batch(obs, task_desc, tokenizer, camera_keys, device)
                    action_chunk = predict_actions_compressed(wrapper, batch)
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
        n_total = len(task_successes)
        sr = n_succ / n_total if n_total > 0 else 0.0

        all_task_results[task_id] = {
            "task_id": task_id,
            "task_description": task_desc,
            "success_count": n_succ,
            "total_episodes": n_total,
            "success_rate": round(sr, 4),
            "mean_steps": round(float(np.mean(task_lengths)), 1),
        }
        total_successes += n_succ
        total_episodes += n_total

        log.info("  Task %d SR: %d/%d (%.1f%%)", task_id, n_succ, n_total, sr * 100)

    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    _, ci_lo, ci_hi = wilson_ci(total_successes, total_episodes)

    log.info("=" * 60)
    log.info("SUITE: %s | M=%d", args.suite, args.n_queries)
    log.info("Overall SR: %d/%d (%.1f%%)", total_successes, total_episodes, overall_sr * 100)
    log.info("95%% Wilson CI: [%.1f%%, %.1f%%]", ci_lo * 100, ci_hi * 100)

    for tid, tr in sorted(all_task_results.items()):
        log.info("  Task %d: %.0f%% (%d/%d) -- %s",
                 tid, tr["success_rate"] * 100, tr["success_count"],
                 tr["total_episodes"], tr["task_description"])
    log.info("=" * 60)

    summary = {
        "suite": args.suite,
        "n_queries": args.n_queries,
        "checkpoint": args.checkpoint_path,
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
    args = parse_args()
    run_evaluation(args)
