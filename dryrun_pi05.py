"""Quick 5-episode dry-run to verify Pi0.5 eval fix."""
import os, sys, json, math, time as time_mod, logging, types
from pathlib import Path
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"

import numpy as np
import torch
import torch._dynamo; torch._dynamo.config.disable = True
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_DIR = "/root/autodl-tmp/pi05-libero-finetuned"
TOKENIZER_PATH = "/root/.cache/huggingface/hub/models--unsloth--gemma-2b/snapshots/7ac9d201a57c8cdc6f939069fc0f044c60197a4a"
TASK_SUITE = "libero_object"
N_EPISODES = 5
DEVICE = "cuda:0"
SEED = 7
REPLAN_STEPS = 5
MAX_TOKEN_LEN = 200

log.info(f"Loading Pi0.5 from {CKPT_DIR} ...")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
policy = PI05Policy.from_pretrained(CKPT_DIR)

paligemma_expert = policy.model.paligemma_with_expert

def patched_embed_image(self, image):
    out_dtype = image.dtype
    if image.dtype != torch.float32:
        image = image.to(torch.float32)
    image_outputs = self.paligemma.model.vision_tower(image)
    selected_image_feature = image_outputs.last_hidden_state
    features = self.paligemma.model.multi_modal_projector(selected_image_feature)
    features = features * self.paligemma.config.text_config.hidden_size**0.5
    if features.dtype != out_dtype:
        features = features.to(out_dtype)
    return features

paligemma_expert.embed_image = types.MethodType(patched_embed_image, paligemma_expert)
log.info("Patched embed_image: bypass get_image_features, use vision_tower+projector directly")

policy = policy.to(DEVICE)
policy.eval()
model = policy.model
log.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.padding_side = "right"

norm_stats = load_file(os.path.join(CKPT_DIR, "policy_preprocessor_step_2_normalizer_processor.safetensors"))
state_mean = norm_stats["observation.state.mean"]
state_std = norm_stats["observation.state.std"]
action_mean = norm_stats["action.mean"]
action_std = norm_stats["action.std"]
NORM_EPS = 1e-8

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

np.random.seed(SEED)
task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()

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
    norm_state = (state_t - state_mean) / (state_std + NORM_EPS)
    state_np = np.clip(norm_state.numpy(), -1.0, 1.0)
    discretized = np.clip(np.digitize(state_np, np.linspace(-1, 1, 257)[:-1]) - 1, 0, 255)

    cleaned_text = task_text.strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized))
    prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
    enc = tokenizer(prompt, max_length=MAX_TOKEN_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")

    return {
        "observation.images.image": base_t.unsqueeze(0).to(DEVICE),
        "observation.images.image2": wrist_t.unsqueeze(0).to(DEVICE),
        "observation.language.tokens": enc.input_ids.to(DEVICE),
        "observation.language.attention_mask": enc.attention_mask.to(DEVICE),
    }

def unnormalize_actions(actions):
    a_mean = action_mean.to(actions.device)
    a_std = action_std.to(actions.device)
    return actions * a_std + a_mean

# Run 5 episodes on task 0
task = task_suite.get_task(0)
init_states = task_suite.get_task_init_states(0)
desc = task.language
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
env.seed(SEED)

log.info(f"Running {N_EPISODES} episodes on task 0: {desc}")

successes = 0
for ep in range(N_EPISODES):
    env.reset()
    obs = env.set_init_state(init_states[ep])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0]*6 + [-1.0])

    action_queue = deque()
    done = False
    first_action_logged = False

    for t in range(280):
        if not action_queue:
            batch = preprocess_obs(obs, desc)
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch)
            actions = unnormalize_actions(actions)
            action_np = actions[0].cpu().numpy()

            if not first_action_logged:
                log.info(f"  ep{ep} first action chunk stats: mean={action_np.mean():.4f} std={action_np.std():.4f} "
                         f"min={action_np.min():.4f} max={action_np.max():.4f}")
                log.info(f"  ep{ep} first 3 actions: {action_np[:3, :4].tolist()}")
                first_action_logged = True

            for a in action_np[:REPLAN_STEPS]:
                action_queue.append(a[:7])

        action = action_queue.popleft()
        obs, reward, done, info = env.step(action.tolist())
        if done:
            successes += 1
            log.info(f"  ep{ep}: SUCCESS at step {t}")
            break

    if not done:
        log.info(f"  ep{ep}: FAIL (timeout)")

env.close()
log.info(f"\nDry-run result: {successes}/{N_EPISODES} = {successes/N_EPISODES*100:.0f}%")
