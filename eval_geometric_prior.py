"""Geometric Prior Pruning eval on LIBERO-Object.

Uses Sobel edge detection to select vision tokens at inference time.
No learned pruning -- training-free baseline for compression analysis.
"""
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import norm

from test_mode_collapse import load_model, load_config
from train import get_tokenizer_from_policy, preprocess_batch
from geometric_prior_pruning import GeometricPriorPruner


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (round(max(0, center - spread), 4), round(min(1, center + spread), 4))


def quat_to_axis_angle(quat):
    r = Rotation.from_quat(quat)
    return r.as_rotvec().astype(np.float32)


def obs_to_batch(obs, task_text, tokenizer, device):
    img1 = obs["pixels"]["image"]
    img2 = obs["pixels"]["image2"]

    img1_t = torch.from_numpy(np.asarray(img1)).permute(2, 0, 1).float() / 255.0
    img2_t = torch.from_numpy(np.asarray(img2)).permute(2, 0, 1).float() / 255.0

    eef_pos = obs["robot_state"]["eef"]["pos"]
    eef_quat = obs["robot_state"]["eef"]["quat"]
    gripper_qpos = obs["robot_state"]["gripper"]["qpos"]

    axis_angle = quat_to_axis_angle(eef_quat)
    state = np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)
    state_t = torch.from_numpy(state).unsqueeze(0)

    batch = {
        "observation.images.image": img1_t.unsqueeze(0),
        "observation.images.image2": img2_t.unsqueeze(0),
        "observation.state": state_t,
        "task": [task_text],
    }
    return preprocess_batch(batch, tokenizer, device=device)


def get_raw_images(obs):
    img1 = np.asarray(obs["pixels"]["image"])
    img2 = np.asarray(obs["pixels"]["image2"])
    return [img1, img2]


@torch.no_grad()
def predict_actions(policy, batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    actions = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state
    )
    action_dim = policy.config.action_feature.shape[0]
    return actions[0, :, :action_dim]


def run_eval(args):
    device = torch.device("cuda:0")

    print(f"Loading model from {args.checkpoint}...")
    policy, config = load_model(args.checkpoint, args.config, device)
    tokenizer = get_tokenizer_from_policy(policy)

    pruner = GeometricPriorPruner(
        policy.model, k_ratio=args.k_ratio, method="true_removal"
    )
    print(f"GeometricPriorPruner initialized: k_ratio={args.k_ratio}, "
          f"expected tokens kept={int(128 * args.k_ratio)}/128")

    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = len(suite.tasks)

    results_per_task = {}
    total_success = 0
    total_eps = 0

    for task_id in range(min(n_tasks, 10)):
        task_text = suite.tasks[task_id].language + "\n"
        task_successes = []

        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name="libero_object",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            control_mode="relative",
            init_states=True,
            episode_index=0,
            n_envs=1,
        )

        for ep in range(args.episodes_per_task):
            obs, info = env.reset(seed=42 + ep)
            action_chunk = None
            action_idx = 0
            success = False

            for step in range(args.max_steps):
                if action_chunk is None or action_idx >= args.action_horizon:
                    raw_imgs = get_raw_images(obs)
                    pruner.set_current_images(raw_imgs)
                    pruner.attach(device)

                    batch = obs_to_batch(obs, task_text, tokenizer, device)
                    action_chunk = predict_actions(policy, batch)

                    pruner.detach()

                    if ep == 0 and step == 0 and task_id == 0:
                        stats = pruner.get_importance_stats()
                        print(f"  Pruning stats (first step): {stats}")

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

            task_successes.append(success)
            status = "OK" if success else "FAIL"
            print(f"  T{task_id} Ep{ep+1}/{args.episodes_per_task}: {status} (steps={step+1})")

        env.close()
        sr = sum(task_successes) / len(task_successes)
        results_per_task[str(task_id)] = {
            "task_id": task_id,
            "task_description": task_text.strip(),
            "success_count": sum(task_successes),
            "total_episodes": len(task_successes),
            "success_rate": sr,
        }
        total_success += sum(task_successes)
        total_eps += len(task_successes)
        print(f"  Task {task_id} SR: {sr:.2%} ({sum(task_successes)}/{len(task_successes)})")

    del policy
    torch.cuda.empty_cache()

    overall_sr = total_success / total_eps if total_eps > 0 else 0.0
    ci = wilson_ci(total_success, total_eps)

    output = {
        "suite": "libero_object",
        "mode": "geometric_prior_pruning",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "k_ratio": args.k_ratio,
        "expected_tokens_kept": int(128 * args.k_ratio),
        "total_tokens": 128,
        "total_episodes": total_eps,
        "total_successes": total_success,
        "overall_success_rate": overall_sr,
        "wilson_ci_95": list(ci),
        "num_episodes_per_task": args.episodes_per_task,
        "max_steps": args.max_steps,
        "action_horizon": args.action_horizon,
        "seed": 42,
        "pruning_method": "true_removal",
        "per_task": results_per_task,
    }

    out_path = PROJECT_DIR / "eval_results" / args.output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\n{'='*50}")
    print(f"OVERALL SR: {overall_sr:.2%} ({total_success}/{total_eps})")
    print(f"Wilson 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"{'='*50}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--k_ratio", type=float, default=0.9)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=280)
    p.add_argument("--action_horizon", type=int, default=50)
    p.add_argument("--output_name", type=str, default="eval_geometric_prior_k09.json")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    run_eval(args)
