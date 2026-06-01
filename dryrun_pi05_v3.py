"""Dry-run using standard LeRobot make_pre_post_processors."""
import os, sys, types, math, logging
from pathlib import Path
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch._dynamo; torch._dynamo.config.disable = True

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_DIR = "./pi05-libero-finetuned"
DEVICE = "cuda:0"
SEED = 7
REPLAN_STEPS = 5

log.info("Loading Pi0.5...")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
policy = PI05Policy.from_pretrained(CKPT_DIR)

# Fix embed_image
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
log.info("Patched embed_image")

policy = policy.to(DEVICE)
policy.eval()
log.info(f"Model loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params")

# Load standard pre/post processors
log.info("Loading preprocessor/postprocessor via make_pre_post_processors...")
from lerobot.policies.factory import make_pre_post_processors
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    CKPT_DIR,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)
log.info("Preprocessor/postprocessor loaded successfully")

# LIBERO setup
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

np.random.seed(SEED)
task_suite = benchmark.get_benchmark_dict()["libero_object"]()

def quat_to_axisangle(quat):
    q = quat.copy()
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3]**2)
    if abs(den) < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * np.arccos(q[3]) / den).astype(np.float32)

# Test on task 0 and task 1
for task_id in [0, 1]:
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(SEED)
    task_desc = task.language
    log.info(f"Task {task_id}: {task_desc}")

    successes = 0
    for ep in range(3):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0]*6 + [-1.0])

        action_queue = deque()
        done = False

        for t in range(280):
            if not action_queue:
                base_img = obs["agentview_image"]
                wrist_img = obs["robot0_eye_in_hand_image"]
                base_t = torch.from_numpy(base_img.copy()).permute(2, 0, 1).float() / 255.0
                wrist_t = torch.from_numpy(wrist_img.copy()).permute(2, 0, 1).float() / 255.0

                eef_pos = obs["robot0_eef_pos"]
                eef_quat = obs["robot0_eef_quat"]
                gripper_qpos = obs["robot0_gripper_qpos"]
                state = np.concatenate([eef_pos, quat_to_axisangle(eef_quat), gripper_qpos]).astype(np.float32)
                state_t = torch.from_numpy(state)

                raw_obs = {
                    "observation.images.image": base_t,
                    "observation.images.image2": wrist_t,
                    "observation.state": state_t,
                    "task": task_desc,
                }

                batch = preprocess(raw_obs)

                with torch.no_grad():
                    actions = policy.predict_action_chunk(batch)

                post_out = postprocess({"action": actions})
                action_np = post_out["action"][0].cpu().numpy()

                if t == 0 and ep == 0:
                    log.info(f"  Batch keys: {list(batch.keys())}")
                    log.info(f"  Action shape: {action_np.shape}")
                    log.info(f"  First 3 actions (full 7d):")
                    for i in range(min(3, len(action_np))):
                        log.info(f"    step{i}: {action_np[i, :7].tolist()}")

                for a in action_np[:REPLAN_STEPS]:
                    action_queue.append(a[:7])

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action.tolist())
            if done:
                successes += 1
                log.info(f"  T{task_id} ep{ep}: SUCCESS at step {t}")
                break

        if not done:
            log.info(f"  T{task_id} ep{ep}: FAIL (timeout)")

    env.close()
    log.info(f"Task {task_id}: {successes}/3")

log.info("Done")
