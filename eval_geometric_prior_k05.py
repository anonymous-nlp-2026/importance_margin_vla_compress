"""Geometric Prior Pruning eval: k=0.5, libero_object, 50 eps/task."""
import os
import sys
import json
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_DIR = Path(".")
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from test_mode_collapse import load_model, load_config
from train import get_tokenizer_from_policy, preprocess_batch
from geometric_prior_pruning import GeometricPriorPruner

K_RATIO = 0.5
SUITE_NAME = "libero_object"
MAX_STEPS = 280
ACTION_HORIZON = 50
EPS_PER_TASK = 50
SEED = 42


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


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


def run_eval():
    device = torch.device("cuda:0")

    ckpt_path = PROJECT_DIR / "checkpoints/libero_full_ft/step_099999/checkpoint.pt"
    config_path = str(PROJECT_DIR / "configs/libero_full_ft.yaml")

    print(f"Loading model from {ckpt_path}...")
    policy, config = load_model(str(ckpt_path), config_path, device)
    tokenizer = get_tokenizer_from_policy(policy)
    print("Model loaded.")

    pruner = GeometricPriorPruner(policy.model, k_ratio=K_RATIO, method="true_removal")
    print(f"GeometricPriorPruner initialized: k_ratio={K_RATIO}")

    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    n_tasks = len(suite.tasks)

    results_per_task = {}
    total_success = 0
    total_eps = 0
    t0 = time.time()

    for task_id in range(min(n_tasks, 10)):
        task_text = suite.tasks[task_id].language + "\n"
        task_desc = suite.tasks[task_id].language
        task_successes = []

        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=SUITE_NAME,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            control_mode="relative",
            init_states=True,
            episode_index=0,
            n_envs=1,
        )

        for ep in range(EPS_PER_TASK):
            obs, info = env.reset(seed=SEED + ep)
            action_chunk = None
            action_idx = 0
            success = False

            for step in range(MAX_STEPS):
                if action_chunk is None or action_idx >= ACTION_HORIZON:
                    raw_images = get_raw_images(obs)
                    pruner.set_current_images(raw_images)
                    pruner.attach(device)

                    batch = obs_to_batch(obs, task_text, tokenizer, device)
                    action_chunk = predict_actions(policy, batch)

                    pruner.detach()

                    if ep == 0 and step == 0 and task_id == 0:
                        stats = pruner.get_importance_stats()
                        print(f"  Pruning stats (first step): {stats}")
                        print(f"  Action shape: {action_chunk.shape}")

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
            print(f"  Task {task_id} Ep {ep+1}/{EPS_PER_TASK}: {status} (steps={step+1})")

        env.close()
        sr = sum(task_successes) / len(task_successes)
        ci = wilson_ci(sum(task_successes), len(task_successes))
        results_per_task[str(task_id)] = {
            "task_id": task_id,
            "task_description": task_desc,
            "success_count": sum(task_successes),
            "total_episodes": len(task_successes),
            "success_rate": sr,
            "wilson_ci_95": [round(ci[0], 4), round(ci[1], 4)],
        }
        total_success += sum(task_successes)
        total_eps += len(task_successes)
        print(f"  Task {task_id} SR: {sr:.0%} ({sum(task_successes)}/{len(task_successes)}) CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

    elapsed = time.time() - t0
    overall_sr = total_success / total_eps if total_eps > 0 else 0.0
    overall_ci = wilson_ci(total_success, total_eps)

    del policy
    torch.cuda.empty_cache()

    output = {
        "suite": SUITE_NAME,
        "mode": "geometric_prior",
        "pruning_method": "true_removal",
        "k_ratio": K_RATIO,
        "checkpoint": str(ckpt_path),
        "config": config_path,
        "total_episodes": total_eps,
        "total_successes": total_success,
        "overall_success_rate": overall_sr,
        "wilson_ci_95": [round(overall_ci[0], 4), round(overall_ci[1], 4)],
        "num_episodes_per_task": EPS_PER_TASK,
        "max_steps": MAX_STEPS,
        "action_horizon": ACTION_HORIZON,
        "seed": SEED,
        "elapsed_seconds": round(elapsed, 1),
        "per_task": results_per_task,
    }

    out_path = PROJECT_DIR / "eval_results" / "eval_geometric_prior_k05.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GEOMETRIC PRIOR PRUNING k={K_RATIO}")
    print(f"OVERALL SR: {overall_sr:.1%} ({total_success}/{total_eps})")
    print(f"Wilson 95% CI: [{overall_ci[0]:.4f}, {overall_ci[1]:.4f}]")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"Results saved to {out_path}")
    return output


if __name__ == "__main__":
    run_eval()
