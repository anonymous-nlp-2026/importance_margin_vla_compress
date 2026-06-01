"""Random Pruning eval on LIBERO-Object for SmolVLA.

Randomly selects vision tokens to remove at inference time.
Training-free baseline for compression cliff-edge analysis.
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

from test_mode_collapse import load_model, load_config
from train import get_tokenizer_from_policy, preprocess_batch
from geometric_prior_pruning import find_connector, register_geometric_hook


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


class RandomPruner:
    def __init__(self, model, k_ratio=0.5, method="true_removal", seed=42):
        self.k_ratio = k_ratio
        self.method = method
        self.connector = find_connector(model)
        if self.connector is None:
            raise RuntimeError("Cannot find connector module in model")
        self.rng = np.random.RandomState(seed)
        self._hook_handle = None
        self._last_mask = None

    def attach(self, device, n_tokens=128):
        k = max(1, int(n_tokens * self.k_ratio))
        indices = self.rng.permutation(n_tokens)[:k]
        mask = torch.zeros(1, n_tokens, device=device)
        mask[0, indices] = 1.0
        self._last_mask = mask
        self._hook_handle = register_geometric_hook(
            self.connector, mask, method=self.method
        )
        return mask

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_stats(self):
        if self._last_mask is None:
            return {}
        return {
            "total_kept": int(self._last_mask.sum().item()),
            "total_tokens": self._last_mask.shape[1],
        }


def run_eval(args):
    device = torch.device("cuda:0")

    ckpt_path = PROJECT_DIR / "checkpoints/libero_full_ft/step_099999/checkpoint.pt"
    config_path = str(PROJECT_DIR / "configs/libero_full_ft.yaml")

    print(f"Loading model from {ckpt_path}...")
    policy, config = load_model(str(ckpt_path), config_path, device)
    tokenizer = get_tokenizer_from_policy(policy)

    pruner = RandomPruner(
        policy.model, k_ratio=args.k_ratio, method="true_removal", seed=args.seed
    )
    k_tokens = max(1, int(128 * args.k_ratio))
    removed = 128 - k_tokens
    print(f"RandomPruner: k_ratio={args.k_ratio}, keep {k_tokens}/128 tokens, remove {removed}")

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
                    pruner.attach(device, n_tokens=128)
                    batch = obs_to_batch(obs, task_text, tokenizer, device)
                    action_chunk = predict_actions(policy, batch)
                    pruner.detach()

                    if ep == 0 and step == 0 and task_id == 0:
                        stats = pruner.get_stats()
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
        "mode": "random_pruning",
        "checkpoint": str(ckpt_path),
        "config": config_path,
        "k_ratio": args.k_ratio,
        "tokens_kept": k_tokens,
        "tokens_removed": removed,
        "total_tokens": 128,
        "total_episodes": total_eps,
        "total_successes": total_success,
        "overall_success_rate": overall_sr,
        "wilson_ci_95": list(ci),
        "num_episodes_per_task": args.episodes_per_task,
        "max_steps": args.max_steps,
        "action_horizon": args.action_horizon,
        "seed": args.seed,
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
    p.add_argument("--k_ratio", type=float, default=0.95)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=280)
    p.add_argument("--action_horizon", type=int, default=50)
    p.add_argument("--output_name", type=str, default="smolvla_random_prune_k095.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_eval(args)
