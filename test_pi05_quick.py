"""Quick smoke test: 1 task, 2 episodes."""
import os, sys, types
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"

import numpy as np
import torch, math, logging
import torch._dynamo; torch._dynamo.config.disable = True
from pathlib import Path
from collections import deque
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_DIR = "./pi05-libero-finetuned"
TOKENIZER_PATH = "./cache/huggingface/hub/models--unsloth--gemma-2b/snapshots/7ac9d201a57c8cdc6f939069fc0f044c60197a4a"
DEVICE = "cuda:0"

log.info("Loading Pi0.5...")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
policy = PI05Policy.from_pretrained(CKPT_DIR)

# Patch embed_image for transformers 4.x compatibility
paligemma_expert = policy.model.paligemma_with_expert

def patched_embed_image(self, image):
    out_dtype = image.dtype
    if image.dtype != torch.float32:
        image = image.to(torch.float32)
    image_outputs = self.paligemma.model.get_image_features(image)
    if isinstance(image_outputs, torch.Tensor):
        features = image_outputs
    else:
        features = image_outputs.pooler_output
    features = features * self.paligemma.config.text_config.hidden_size**0.5
    if features.dtype != out_dtype:
        features = features.to(out_dtype)
    return features

paligemma_expert.embed_image = types.MethodType(patched_embed_image, paligemma_expert)
log.info("Patched embed_image")

policy = policy.to(DEVICE)
policy.eval()
log.info(f"Model loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params")
log.info(f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.padding_side = "right"

norm_stats = load_file(os.path.join(CKPT_DIR, "policy_preprocessor_step_2_normalizer_processor.safetensors"))
state_mean = norm_stats["observation.state.mean"]
state_std = norm_stats["observation.state.std"]
action_mean = norm_stats["action.mean"].to(DEVICE)
action_std = norm_stats["action.std"].to(DEVICE)

def quat_to_axisangle(quat):
    q = quat.copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = np.sqrt(1.0 - q[3]**2)
    if abs(den) < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * np.arccos(q[3]) / den).astype(np.float32)

def preprocess_obs(obs, task_text):
    base_img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    base_t = torch.from_numpy(base_img.copy()).permute(2, 0, 1).float() / 255.0
    wrist_t = torch.from_numpy(wrist_img.copy()).permute(2, 0, 1).float() / 255.0

    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    gripper_qpos = obs["robot0_gripper_qpos"]
    state = np.concatenate([eef_pos, quat_to_axisangle(eef_quat), gripper_qpos]).astype(np.float32)
    state_t = torch.from_numpy(state)
    norm_state = (state_t - state_mean) / (state_std + 1e-8)
    state_np = np.clip(norm_state.numpy(), -1.0, 1.0)
    discretized = np.clip(np.digitize(state_np, np.linspace(-1, 1, 257)[:-1]) - 1, 0, 255)

    cleaned_text = task_text.strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized))
    prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
    enc = tokenizer(prompt, max_length=200, padding="max_length", truncation=True, return_tensors="pt")

    return {
        "observation.images.image": base_t.unsqueeze(0).to(DEVICE),
        "observation.images.image2": wrist_t.unsqueeze(0).to(DEVICE),
        "observation.language.tokens": enc.input_ids.to(DEVICE),
        "observation.language.attention_mask": enc.attention_mask.to(DEVICE),
    }

# LIBERO
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

np.random.seed(7)
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(0)
init_states = suite.get_task_init_states(0)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
env.seed(7)
task_desc = task.language
log.info(f"Task: {task_desc}")

for ep in range(2):
    env.reset()
    obs = env.set_init_state(init_states[ep])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0]*6 + [-1.0])

    action_queue = deque()
    done = False
    for t in range(280):
        if not action_queue:
            batch = preprocess_obs(obs, task_desc)
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch)
            actions = actions * action_std + action_mean
            action_np = actions[0].cpu().numpy()
            for a in action_np[:5]:
                action_queue.append(a[:7])
            if t == 0 and ep == 0:
                log.info(f"  First action chunk shape: {actions.shape}")
                log.info(f"  First action: {action_np[0, :7]}")
                log.info(f"  GPU mem after 1st inference: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

        action = action_queue.popleft()
        obs, reward, done, info = env.step(action.tolist())
        if done:
            log.info(f"  ep={ep} SUCCESS at step {t}")
            break

    if not done:
        log.info(f"  ep={ep} FAIL (timeout at step 280)")

env.close()
log.info("Smoke test complete!")
