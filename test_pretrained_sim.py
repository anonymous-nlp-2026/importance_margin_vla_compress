"""Pretrained SmolVLA (no LoRA) sanity check in aloha sim."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time as time_mod
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from train import get_tokenizer_from_policy
from eval_success_rate import obs_to_batch, predict_actions

def main():
    device = torch.device("cuda:0")

    pretrained_path = "/root/autodl-tmp/.hf_cache/lerobot/smolvla_base"
    print(f"Loading pretrained from {pretrained_path}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_path)

    # Adapt features to aloha sim (14-dim state/action, top camera only)
    policy.config.input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(14,)
    )
    policy.config.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION, shape=(14,)
    )
    old_cams = [k for k, v in policy.config.input_features.items() if v.type == FeatureType.VISUAL]
    for k in old_cams:
        del policy.config.input_features[k]
    policy.config.input_features["observation.images.top"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 256, 256)
    )

    policy.to(device)
    policy.eval()
    print("Model loaded (pretrained, NO LoRA).")

    tokenizer = get_tokenizer_from_policy(policy)

    import gymnasium as gym
    import gym_aloha

    env = gym.make(
        "gym_aloha/AlohaInsertion-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=400,
    )

    # === Action Diversity Check ===
    print("\n=== Action Diversity Check (5 samples, same obs) ===")
    obs, info = env.reset(seed=42)
    batch = obs_to_batch(obs, tokenizer, device)
    actions_list = []
    for _ in range(5):
        a = predict_actions(policy, batch)
        actions_list.append(a)
    stacked = torch.stack(actions_list)
    std_across = stacked.std(dim=0).mean().item()
    print(f"Diversity std (across 5 samples): {std_across:.6f}")
    print(f"Action range: [{stacked.min().item():.3f}, {stacked.max().item():.3f}]")
    for i, a in enumerate(actions_list):
        print(f"  Sample {i}: mean={a.mean().item():.4f} std={a.std().item():.4f} "
              f"first3={a[0,:3].tolist()}")

    # === 20 Episode Rollout ===
    print("\n=== 20 Episode Rollout ===")
    successes = []
    all_actions = []
    for ep in range(20):
        t0 = time_mod.time()
        obs, info = env.reset(seed=42 + ep)
        action_chunk = None
        action_idx = 0
        total_reward = 0.0
        success = False
        ep_actions = []

        for step in range(400):
            if action_chunk is None or action_idx >= 50:
                batch = obs_to_batch(obs, tokenizer, device)
                action_chunk = predict_actions(policy, batch)
                action_idx = 0

            action = action_chunk[action_idx].cpu().numpy().astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            ep_actions.append(action.copy())
            action_idx += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info.get("is_success", False):
                success = True
                break
            if terminated or truncated:
                break

        dt = time_mod.time() - t0
        successes.append(success)
        ep_arr = np.array(ep_actions)
        ep_std = ep_arr.std(axis=0).mean()
        all_actions.extend(ep_actions)
        print(f"Ep {ep+1:2d}/20 | {'SUCCESS' if success else 'FAIL   '} | steps={step+1:3d} | "
              f"reward={total_reward:.4f} | action_std={ep_std:.4f} | {dt:.1f}s")

    env.close()

    rate = sum(successes) / len(successes)
    all_arr = np.array(all_actions)
    overall_std = all_arr.std(axis=0).mean()

    print(f"\n=== Summary ===")
    print(f"Success: {sum(successes)}/20 ({rate*100:.1f}%)")
    print(f"Overall action std: {overall_std:.6f}")
    print(f"Diversity std (same obs, diff noise): {std_across:.6f}")

    if rate == 0 and std_across < 0.01 and overall_std < 0.01:
        print("DIAGNOSIS: Pretrained 0% + low diversity -> PIPELINE BUG")
    elif rate == 0:
        print("DIAGNOSIS: Pretrained 0% but diverse actions -> Expected (not trained on aloha), pipeline OK")
    else:
        print(f"DIAGNOSIS: Pretrained {rate*100:.1f}% success -> Pipeline works")

if __name__ == "__main__":
    main()
