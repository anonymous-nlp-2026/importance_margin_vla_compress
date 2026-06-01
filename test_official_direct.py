"""Run the EXACT official eval code with patched transformers."""
import os, sys, json
os.environ["MUJOCO_GL"] = "egl"

# Add official repo to path
sys.path.insert(0, "/root/autodl-tmp/openvla-oft-official")

import numpy as np
import torch
from collections import deque
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    get_libero_wrist_image, quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_vla_action, resize_image_for_policy, normalize_proprio,
    prepare_images_for_vla, center_crop_image, OPENVLA_IMAGE_SIZE,
)
from experiments.robot.robot_utils import (
    invert_gripper_action, normalize_gripper_action,
)
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from transformers import AutoProcessor
from pathlib import Path
from dataclasses import dataclass

CKPT = "/root/autodl-tmp/openvla-oft-libero-object"
DEVICE = "cuda:0"

# Minimal config object that mimics GenerateConfig
@dataclass
class MinimalCfg:
    model_family: str = "openvla"
    pretrained_checkpoint: str = CKPT
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    unnorm_key: str = "libero_object_no_noops"

cfg = MinimalCfg()

# Load model (same as get_vla but without HF Hub checks)
print("Loading model...")
vla = OpenVLAForActionPrediction.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
).eval().to(DEVICE)
vla.vision_backbone.set_num_images_in_input(2)

with open(Path(CKPT) / "dataset_statistics.json") as f:
    vla.norm_stats = json.load(f)

processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

# Load action head
action_head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
ah_ckpt = list(Path(CKPT).glob("action_head*checkpoint*"))[0]
ah_state = torch.load(str(ah_ckpt), weights_only=True)
ah_state = {k.replace("module.", ""): v for k, v in ah_state.items()}
action_head.load_state_dict(ah_state)
action_head = action_head.to(DEVICE, dtype=torch.bfloat16).eval()

# Load proprio projector
pp = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
pp_ckpt = list(Path(CKPT).glob("proprio_projector*checkpoint*"))[0]
pp_state = torch.load(str(pp_ckpt), weights_only=True)
pp_state = {k.replace("module.", ""): v for k, v in pp_state.items()}
pp.load_state_dict(pp_state)
pp = pp.to(DEVICE, dtype=torch.bfloat16).eval()

# Check attention type
attn_type = type(vla.language_model.model.layers[0].self_attn).__name__
print(f"Attention type: {attn_type}")

print("Model loaded!")

# Get task from benchmark (exactly like the official eval)
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_object"]()
num_tasks = task_suite.n_tasks
print(f"Number of tasks: {num_tasks}")

# Run eval on task 0 (pick up alphabet soup)
task_id = 0
task = task_suite.get_task(task_id)
initial_states = task_suite.get_task_init_states(task_id)
env, task_description = get_libero_env(task, "openvla")

print(f"Task: '{task_description}'")
print(f"Init states shape: {initial_states.shape}")

resize_size = OPENVLA_IMAGE_SIZE
max_steps = 280
num_episodes = 10
num_steps_wait = 10

successes = 0
for ep in range(num_episodes):
    env.reset()
    obs = env.set_init_state(initial_states[ep])

    action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
    t = 0
    success = False

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        # Prepare observation (EXACTLY like official code)
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)
        img_resized = resize_image_for_policy(img, resize_size)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        if len(action_queue) == 0:
            actions = get_vla_action(
                cfg, vla, processor, observation, task_description,
                action_head=action_head, proprio_projector=pp,
            )
            action_queue.extend(actions)

        action = action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    successes += int(success)
    print(f"Ep {ep+1}/{num_episodes}: {'OK' if success else 'FAIL'} (steps={t})")
    env.close()
    # Re-create env for each episode (like official code)
    if ep < num_episodes - 1:
        env, _ = get_libero_env(task, "openvla")

print(f"\nResult: {successes}/{num_episodes} ({successes/num_episodes*100:.0f}%)")
