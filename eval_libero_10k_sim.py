"""5-episode LIBERO-Object sim rollout for 10K checkpoints on cuda:2."""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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


def quat_to_axis_angle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3D)."""
    r = Rotation.from_quat(quat)  # scipy uses (x,y,z,w)
    return r.as_rotvec().astype(np.float32)


def obs_to_batch(obs, task_text, tokenizer, device):
    """Convert LIBERO env observation to model input batch."""
    # Images: (H, W, 3) uint8 -> (1, 3, H, W) float [0,1]
    img1 = obs["pixels"]["image"]  # agentview
    img2 = obs["pixels"]["image2"]  # wrist
    
    img1_t = torch.from_numpy(np.asarray(img1)).permute(2, 0, 1).float() / 255.0
    img2_t = torch.from_numpy(np.asarray(img2)).permute(2, 0, 1).float() / 255.0
    
    # State: eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8D
    eef_pos = obs["robot_state"]["eef"]["pos"]  # (3,)
    eef_quat = obs["robot_state"]["eef"]["quat"]  # (4,) xyzw
    gripper_qpos = obs["robot_state"]["gripper"]["qpos"]  # (2,)
    
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
    """Predict action chunk from preprocessed batch."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    
    actions = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state
    )
    action_dim = policy.config.action_feature.shape[0]
    return actions[0, :, :action_dim]  # (chunk_size, action_dim)


def run_eval(ckpt_path, config_path, n_episodes=5, max_steps=280, action_horizon=10):
    device = torch.device("cuda:0")
    
    print(f"Loading model from {ckpt_path}...")
    policy, config = load_model(str(ckpt_path), config_path, device)
    tokenizer = get_tokenizer_from_policy(policy)
    
    # Create LIBERO env
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_task_init_states
    
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_id = 0
    task_text = suite.tasks[task_id].language + "\n"
    
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
    
    successes = []
    for ep in range(n_episodes):
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
        
        successes.append(success)
        print(f"  Ep {ep+1}/{n_episodes}: {'SUCCESS' if success else 'FAIL'} (steps={step+1})")
    
    env.close()
    del policy
    torch.cuda.empty_cache()
    
    n_success = sum(successes)
    return n_success, n_episodes


if __name__ == "__main__":
    configs = {
        "Full FT 10K": (
            PROJECT_DIR / "checkpoints/libero_full_ft/step_010000/checkpoint.pt",
            str(PROJECT_DIR / "configs/libero_full_ft.yaml"),
        ),
        "LoRA 10K": (
            PROJECT_DIR / "checkpoints/libero_baseline/step_010000/checkpoint.pt",
            str(PROJECT_DIR / "configs/libero_baseline.yaml"),
        ),
    }
    
    results = {}
    for name, (ckpt, cfg) in configs.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")
        n_success, n_total = run_eval(ckpt, cfg)
        results[name] = f"{n_success}/{n_total}"
        print(f"  => success={n_success}/{n_total}")
    
    print(f"\n{'='*50}")
    print("SIM ROLLOUT SUMMARY")
    print(f"{'='*50}")
    for name, r in results.items():
        print(f"  {name}: success={r}")
