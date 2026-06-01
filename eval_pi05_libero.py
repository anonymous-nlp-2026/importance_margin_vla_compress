"""Pi0.5 LIBERO-Object eval with optional random vision token pruning.

Usage:
    python eval_pi05_libero.py           # baseline (no pruning)
    python eval_pi05_libero.py 0.5       # 50% random pruning
"""
import os, sys, json, math, time as time_mod, logging, types
from pathlib import Path
from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"

import numpy as np
import torch
import torch._dynamo; torch._dynamo.config.disable = True
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───
CKPT_DIR = "./pi05-libero-finetuned"
TOKENIZER_PATH = "./cache/huggingface/hub/models--unsloth--gemma-2b/snapshots/7ac9d201a57c8cdc6f939069fc0f044c60197a4a"
TASK_SUITE = "libero_object"
N_EPISODES_PER_TASK = 50
DEVICE = "cuda:0"
PRUNE_RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
SEED = 7
REPLAN_STEPS = 5
MAX_TOKEN_LEN = 200
RESULTS_DIR = Path("./eval_results")

# ─── Load model ───
log.info(f"Loading Pi0.5 from {CKPT_DIR} ...")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
policy = PI05Policy.from_pretrained(CKPT_DIR)

# Patch embed_image for transformers >=4.50 compatibility
# New transformers get_image_features() divides output by hidden_size**0.5,
# but the original openpi code expected raw projected features (no division).
# Fix: bypass get_image_features and call vision_tower + projector directly.
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
total_params = sum(p.numel() for p in model.parameters())
log.info(f"Model loaded: {total_params/1e6:.1f}M params on {DEVICE}")

# ─── Load tokenizer ───
log.info("Loading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.padding_side = "right"

# ─── Load normalization stats ───
norm_stats = load_file(os.path.join(CKPT_DIR, "policy_preprocessor_step_2_normalizer_processor.safetensors"))
state_mean = norm_stats["observation.state.mean"]
state_std = norm_stats["observation.state.std"]
action_mean = norm_stats["action.mean"]
action_std = norm_stats["action.std"]
NORM_EPS = 1e-8

# ─── Vision token info ───
N_PATCHES = (224 // 14) ** 2  # 256
log.info(f"Vision tokens: {N_PATCHES} per image, 3 image slots -> {N_PATCHES*3} total")

# ─── Install pruning hook ───
if PRUNE_RATIO > 0:
    def pruned_embed_prefix(self_model, images, img_masks, tokens, masks):
        embs = []
        pad_masks_list = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self_model.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs, dim = img_emb.shape

            n_keep = max(1, int(num_img_embs * (1.0 - PRUNE_RATIO)))
            indices = torch.randperm(num_img_embs)[:n_keep].sort().values.to(img_emb.device)
            img_emb = img_emb[:, indices, :]

            embs.append(img_emb)
            pad_masks_list.append(img_mask[:, None].expand(bsize, n_keep))
            att_masks += [0] * n_keep

        lang_emb = self_model.paligemma_with_expert.embed_language_tokens(tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks_list.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks_out = torch.cat(pad_masks_list, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks_out.device)
        att_masks = att_masks[None, :].expand(pad_masks_out.shape[0], -1)
        return embs, pad_masks_out, att_masks

    model.embed_prefix = types.MethodType(
        lambda self, images, img_masks, tokens, masks: pruned_embed_prefix(self, images, img_masks, tokens, masks),
        model
    )
    n_keep = max(1, int(N_PATCHES * (1 - PRUNE_RATIO)))
    log.info(f"Pruning: random {PRUNE_RATIO*100:.0f}% removal -> {n_keep}/{N_PATCHES} tokens/img")

# ─── LIBERO setup ───
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

np.random.seed(SEED)
task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
n_tasks = task_suite.n_tasks
max_steps_map = {"libero_object": 280, "libero_spatial": 220, "libero_goal": 300, "libero_10": 520}
max_steps = max_steps_map.get(TASK_SUITE, 400)

log.info(f"Suite: {TASK_SUITE}, {n_tasks} tasks x {N_EPISODES_PER_TASK} eps")

# ─── Helpers ───
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

def get_libero_env(task, resolution, seed):
    desc = task.language
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, desc

# ─── Eval loop ───
total_success = 0
total_episodes = 0
results = {
    "config": {
        "model": "pi05", "ckpt": CKPT_DIR, "suite": TASK_SUITE,
        "prune_ratio": PRUNE_RATIO, "n_eps_per_task": N_EPISODES_PER_TASK,
        "seed": SEED, "vision_tokens_per_image": N_PATCHES,
        "total_vision_tokens": N_PATCHES * 3,
        "n_keep_per_image": max(1, int(N_PATCHES * (1 - PRUNE_RATIO))) if PRUNE_RATIO > 0 else N_PATCHES,
    },
    "per_task": {},
}

t_start = time_mod.time()
for task_id in range(n_tasks):
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    env, task_desc = get_libero_env(task, 256, SEED)

    task_successes = 0

    for ep in range(N_EPISODES_PER_TASK):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0]*6 + [-1.0])

        done = False
        action_queue = deque()

        for t in range(max_steps):
            if not action_queue:
                batch = preprocess_obs(obs, task_desc)
                with torch.no_grad():
                    actions = policy.predict_action_chunk(batch)
                actions = unnormalize_actions(actions)
                action_np = actions[0].cpu().numpy()
                for a in action_np[:REPLAN_STEPS]:
                    action_queue.append(a[:7])

            action = action_queue.popleft()
            obs, reward, done, info = env.step(action.tolist())
            if done:
                task_successes += 1
                total_success += 1
                break

        total_episodes += 1

        if (ep + 1) % 10 == 0:
            log.info(f"  T{task_id} [{task_desc[:40]}] ep={ep+1}: "
                     f"{task_successes}/{ep+1}={task_successes/(ep+1)*100:.0f}%")

    task_sr = task_successes / N_EPISODES_PER_TASK
    results["per_task"][task_desc] = {"id": task_id, "sr": task_sr, "n": task_successes}
    elapsed = time_mod.time() - t_start
    log.info(f"Task {task_id}: SR={task_sr*100:.1f}% | "
             f"Total: {total_success}/{total_episodes}={total_success/total_episodes*100:.1f}% | "
             f"Time: {elapsed/60:.1f}min")
    env.close()

results["overall_sr"] = total_success / total_episodes
results["total_successes"] = total_success
results["total_episodes"] = total_episodes
results["elapsed_min"] = (time_mod.time() - t_start) / 60

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if PRUNE_RATIO > 0:
    fname = f"pi05_random_prune_k{1-PRUNE_RATIO:.2f}_{TASK_SUITE}.json"
else:
    fname = f"pi05_baseline_{TASK_SUITE}.json"
out_path = RESULTS_DIR / fname
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log.info(f"\n{'='*60}")
log.info(f"RESULT: {TASK_SUITE} prune={PRUNE_RATIO} SR={results['overall_sr']*100:.1f}%")
log.info(f"Saved: {out_path}")
log.info(f"{'='*60}")
