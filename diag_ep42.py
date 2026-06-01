import torch, numpy as np, sys
sys.path.insert(0, '.')
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"
os.environ["WANDB_MODE"] = "offline"

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from train import load_smolvla_policy, load_config, get_tokenizer_from_policy
from eval_success_rate import predict_actions, TASK_TEXT
import gymnasium as gym
import gym_aloha

device = torch.device('cuda:0')
config = load_config('configs/baseline_v6_alpha16.yaml')
policy = load_smolvla_policy(config, device)
ckpt = torch.load('checkpoints/baseline_v6_alpha16/latest/checkpoint.pt', map_location=device, weights_only=False)
policy.load_state_dict(ckpt['model_state_dict'], strict=False)
policy.eval()
tokenizer = get_tokenizer_from_policy(policy)

repo_id = 'lerobot/aloha_sim_insertion_scripted_image'
meta = LeRobotDatasetMetadata(repo_id)
ds = LeRobotDataset(repo_id=repo_id, delta_timestamps={'action': [i / meta.fps for i in range(50)]})

# Ep42 from dataset
idx = 42 * 400
sample = ds[idx]
train_img = sample['observation.images.top']
print(f'Dataset ep_idx at frame {idx}: {sample.get("episode_index", "N/A")}')

# Env seed 42
env = gym.make('gym_aloha/AlohaInsertion-v0', obs_type='pixels_agent_pos', max_episode_steps=400)
obs, _ = env.reset(seed=42)
env_img = torch.from_numpy(obs['pixels']['top']).permute(2,0,1).float()/255.0

print(f'Image diff train_ep42 vs env_seed42: {(train_img - env_img).abs().mean():.6f}')
print(f'Train img mean: {train_img.mean():.4f}')
print(f'Env img mean: {env_img.mean():.4f}')

# Model on training ep42 image
batch = {
    'observation.images.top': train_img.unsqueeze(0).to(device),
    'observation.state': sample['observation.state'].unsqueeze(0).to(device),
}
encoded = tokenizer([TASK_TEXT], padding='longest', max_length=48, truncation=True, return_tensors='pt')
batch['observation.language.tokens'] = encoded['input_ids'].to(device)
batch['observation.language.attention_mask'] = encoded['attention_mask'].bool().to(device)

pred_train = predict_actions(policy, batch, temperature=1.0)
print(f'\n--- Prediction on TRAINING ep42 image ---')
print(f'Pred range: [{pred_train.min():.3f}, {pred_train.max():.3f}]')
print(f'Pred[:7]: {pred_train[0, :7].cpu().numpy()}')
print(f'GT[:7]:   {sample["action"][0, :7].numpy()}')

# Model on ENV seed42 image
from eval_success_rate import obs_to_batch
batch2 = obs_to_batch(obs, tokenizer, device)
pred_env = predict_actions(policy, batch2, temperature=1.0)
print(f'\n--- Prediction on ENV seed42 image ---')
print(f'Pred range: [{pred_env.min():.3f}, {pred_env.max():.3f}]')
print(f'Pred[:7]: {pred_env[0, :7].cpu().numpy()}')

# Also test env seed 0 vs env seed 42 directly
obs0, _ = env.reset(seed=0)
batch0 = obs_to_batch(obs0, tokenizer, device)
pred_env0 = predict_actions(policy, batch0, temperature=1.0)
print(f'\n--- Prediction on ENV seed0 image ---')
print(f'Pred range: [{pred_env0.min():.3f}, {pred_env0.max():.3f}]')
print(f'Pred[:7]: {pred_env0[0, :7].cpu().numpy()}')

env.close()
print('\nDone!')
