"""Debug: examine the actual action values produced by the OFT model."""
import os, sys, json, math, io
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, "./openvla-oft-official")

CKPT = "./openvla-oft-libero-object"
DEVICE = "cuda:0"
ACTION_DIM = 7

print("Loading model with SDPA + patched transformers...")
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

# Try loading with AutoModelForVision2Seq to use checkpoint's own modeling code
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
).eval().to(DEVICE)
model.vision_backbone.set_num_images_in_input(2)

with open(Path(CKPT) / "dataset_statistics.json") as f:
    model.norm_stats = json.load(f)

processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector

ckpt_path = Path(CKPT)
action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=ACTION_DIM)
ah_state = torch.load(str(list(ckpt_path.glob("action_head*.pt"))[0]), map_location="cpu", weights_only=True)
ah_state = {k.replace("module.", ""): v for k, v in ah_state.items()}
action_head.load_state_dict(ah_state)
action_head = action_head.to(DEVICE, dtype=torch.bfloat16).eval()

pp = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
pp_state = torch.load(str(list(ckpt_path.glob("proprio_projector*.pt"))[0]), map_location="cpu", weights_only=True)
pp_state = {k.replace("module.", ""): v for k, v in pp_state.items()}
pp.load_state_dict(pp_state)
pp = pp.to(DEVICE, dtype=torch.bfloat16).eval()

# Check attention type
print(f"Attention type: {type(model.language_model.model.layers[0].self_attn).__name__}")

# Check model class
print(f"Model class: {type(model).__name__}")
print(f"Model loaded from: {type(model).__module__}")

unnorm_key = "libero_object_no_noops"
norm_stats = model.norm_stats[unnorm_key]
print(f"\nNormalization stats for {unnorm_key}:")
print(f"  action q01: {norm_stats['action']['q01']}")
print(f"  action q99: {norm_stats['action']['q99']}")

# Setup env
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.robot.openvla_utils import resize_image_for_policy, normalize_proprio, center_crop_image, get_vla_action, OPENVLA_IMAGE_SIZE
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
from libero.libero import benchmark
from dataclasses import dataclass
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

@dataclass
class MinCfg:
    model_family: str = "openvla"
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    unnorm_key: str = "libero_object_no_noops"

cfg = MinCfg()

bd = benchmark.get_benchmark_dict()
ts = bd["libero_object"]()
task = ts.get_task(0)
init_states = ts.get_task_init_states(0)

from experiments.robot.libero.libero_utils import get_libero_env
env, task_desc = get_libero_env(task, "openvla")
print(f"\nTask: '{task_desc}'")

env.reset()
obs = env.set_init_state(init_states[0])

# Warmup
for _ in range(10):
    obs, _, _, _ = env.step([0,0,0,0,0,0,-1])

# Get observation
img = get_libero_image(obs)
wrist_img = get_libero_wrist_image(obs)
img_resized = resize_image_for_policy(img, OPENVLA_IMAGE_SIZE)
wrist_img_resized = resize_image_for_policy(wrist_img, OPENVLA_IMAGE_SIZE)

observation = {
    "full_image": img_resized,
    "wrist_image": wrist_img_resized,
    "state": np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ),
}

print(f"\nProprio (raw): {observation['state']}")

actions = get_vla_action(
    cfg, model, processor, observation, task_desc,
    action_head=action_head, proprio_projector=pp,
)

print(f"\nPredicted actions (unnormalized):")
for i, a in enumerate(actions):
    print(f"  Chunk {i}: {a}")

print(f"\nAfter post-processing:")
for i, a in enumerate(actions):
    a_proc = normalize_gripper_action(a, binarize=True)
    a_proc = invert_gripper_action(a_proc)
    print(f"  Chunk {i}: {a_proc}")

env.close()
