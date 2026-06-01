"""100K LIBERO-Object sim rollout SR eval on cuda:1."""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_DIR = Path("/root/autodl-tmp/importance_margin_vla_compress")
sys.path.insert(0, str(PROJECT_DIR))

import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from test_mode_collapse import load_model, load_config
from train import get_tokenizer_from_policy, preprocess_batch


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


def run_eval(ckpt_path, config_path, n_episodes=100, max_steps=400, action_horizon=50):
    device = torch.device("cuda:0")
    
    print(f"Loading model from {ckpt_path}...")
    policy, config = load_model(str(ckpt_path), config_path, device)
    tokenizer = get_tokenizer_from_policy(policy)
    
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_task_init_states
    
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    n_tasks = len(suite.tasks)
    
    results_per_task = {}
    total_success = 0
    total_eps = 0
    
    eps_per_task = max(n_episodes // n_tasks, 1)
    
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
        
        for ep in range(eps_per_task):
            obs, info = env.reset(seed=42 + ep)
            action_chunk = None
            action_idx = 0
            success = False
            
            for step in range(max_steps):
                if action_chunk is None or action_idx >= action_horizon:
                    batch = obs_to_batch(obs, task_text, tokenizer, device)
                    action_chunk = predict_actions(policy, batch)
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
            status = "SUCCESS" if success else "FAIL"
            print(f"  Task {task_id} Ep {ep+1}/{eps_per_task}: {status} (steps={step+1})")
        
        env.close()
        sr = sum(task_successes) / len(task_successes)
        results_per_task[f"task_{task_id}"] = {
            "task_text": task_text.strip(),
            "success_count": sum(task_successes),
            "n_episodes": len(task_successes),
            "success_rate": sr,
        }
        total_success += sum(task_successes)
        total_eps += len(task_successes)
        print(f"  Task {task_id} SR: {sr:.2%} ({sum(task_successes)}/{len(task_successes)})")
    
    del policy
    torch.cuda.empty_cache()
    
    overall_sr = total_success / total_eps if total_eps > 0 else 0.0
    
    output = {
        "checkpoint": str(ckpt_path),
        "config": config_path,
        "total_success": total_success,
        "total_episodes": total_eps,
        "overall_success_rate": overall_sr,
        "max_steps": max_steps,
        "action_horizon": action_horizon,
        "per_task": results_per_task,
    }
    
    out_path = PROJECT_DIR / "eval_results" / "eval_lora_100k_sr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\n{'='*50}")
    print(f"OVERALL SR: {overall_sr:.2%} ({total_success}/{total_eps})")
    print(f"{'='*50}")
    return output


if __name__ == "__main__":
    ckpt = PROJECT_DIR / "checkpoints/libero_baseline/step_099999/checkpoint.pt"
    cfg = str(PROJECT_DIR / "configs/libero_baseline.yaml")
    run_eval(ckpt, cfg)
