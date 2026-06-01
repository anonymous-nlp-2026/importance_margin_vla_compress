"""Test OFT eval with corrected task descriptions ("pick up the..." vs "pick the...")."""
import os, sys, json, math, io
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from collections import deque

sys.path.insert(0, "./openvla-oft-official")

CKPT = "./openvla-oft-libero-object"
DEVICE = "cuda:0"
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8

# Load model via official code
print("Loading model via official code...")
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from transformers import AutoProcessor

model = OpenVLAForActionPrediction.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation='eager',
).eval().to(DEVICE)
model.vision_backbone.set_num_images_in_input(2)

with open(Path(CKPT) / "dataset_statistics.json") as f:
    model.norm_stats = json.load(f)

processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

# Load action head and proprio projector
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

unnorm_key = "libero_object_no_noops"
proprio_norm_stats = model.norm_stats[unnorm_key]["proprio"]
print("Model loaded!")

# Image preprocessing (official style)
def resize_image_for_policy(img_np, size=224):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    pil = Image.open(buf)
    pil.load()
    return np.array(pil.resize((size, size), Image.LANCZOS))

def center_crop_image(image, crop_scale=0.9):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    w, h = image.size
    s = math.sqrt(crop_scale)
    nw, nh = int(round(w*s)), int(round(h*s))
    l, t = (w-nw)/2.0, (h-nh)/2.0
    image = image.crop((int(math.floor(l)), int(math.floor(t)),
                        int(math.floor(l))+nw, int(math.floor(t))+nh))
    return image.resize((224, 224), Image.BILINEAR)

def quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def normalize_proprio(proprio, norm_stats):
    mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)))
    q01, q99 = np.array(norm_stats["q01"], dtype=np.float32), np.array(norm_stats["q99"], dtype=np.float32)
    return np.clip(np.where(mask, 2.0*(proprio - q01)/(q99 - q01 + 1e-8) - 1.0, proprio), -1.0, 1.0)

def normalize_gripper_action(action, binarize=True):
    a = action.copy()
    a[..., -1] = 2.0*a[..., -1] - 1.0
    if binarize: a[..., -1] = np.sign(a[..., -1])
    return a

def invert_gripper_action(action):
    a = action.copy()
    a[..., -1] *= -1.0
    return a

# Setup LIBERO
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils

bddl_dir = get_libero_path("bddl_files")
init_dir = get_libero_path("init_states")
task_name = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
bddl_path = os.path.join(bddl_dir, "libero_object", f"{task_name}.bddl")
init_states = torch.load(os.path.join(init_dir, "libero_object", f"{task_name}.pruned_init"), weights_only=False)

# KEY FIX: Use benchmark-style task description instead of BDDL-parsed one
# BDDL gives: "pick the alphabet soup and place it in the basket"  
# Benchmark gives: "pick up the alphabet soup and place it in the basket"
# Training data used benchmark descriptions!
bddl_info = BDDLUtils.get_problem_info(bddl_path)
task_desc_bddl = bddl_info["language_instruction"]
task_desc_fixed = "pick up the alphabet soup and place it in the basket"

print(f"BDDL task desc: '{task_desc_bddl}'")
print(f"Fixed task desc: '{task_desc_fixed}'")
print(f"Init states: {init_states.shape}")

# Use the fixed description
task_desc = task_desc_fixed

max_steps = 280
num_open_loop = 8
num_episodes = 10

successes = 0
for ep in range(num_episodes):
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[ep % init_states.shape[0]])
    
    # 10-step warm-up
    for _ in range(10):
        obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
    
    action_queue = deque(maxlen=num_open_loop)
    success = False
    
    for step in range(max_steps):
        if len(action_queue) == 0:
            img = obs["agentview_image"][::-1, ::-1].copy()
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
            
            img_resized = resize_image_for_policy(img)
            wrist_resized = resize_image_for_policy(wrist_img)
            img_pil = center_crop_image(img_resized)
            wrist_pil = center_crop_image(wrist_resized)
            
            proprio = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"]
            ]).astype(np.float32)
            proprio_normed = normalize_proprio(proprio, proprio_norm_stats)
            
            prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
            inp1 = processor(prompt, img_pil).to(DEVICE, dtype=torch.bfloat16)
            inp2 = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
            inp1["pixel_values"] = torch.cat([inp1["pixel_values"], inp2["pixel_values"]], dim=1)
            
            with torch.inference_mode():
                actions_pred, _ = model.predict_action(
                    **inp1,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                    proprio=proprio_normed,
                    proprio_projector=pp,
                    action_head=action_head,
                )
            
            for i in range(min(num_open_loop, len(actions_pred))):
                a = normalize_gripper_action(actions_pred[i], binarize=True)
                a = invert_gripper_action(a)
                action_queue.append(a)
        
        action = action_queue.popleft()
        obs, reward, done, info_env = env.step(action.tolist())
        if done:
            success = True
            break
    
    successes += int(success)
    print(f"Ep {ep+1}/{num_episodes}: {'OK' if success else 'FAIL'} (steps={step+1})")
    env.close()

print(f"\nResult: {successes}/{num_episodes} ({successes/num_episodes*100:.0f}%)")
