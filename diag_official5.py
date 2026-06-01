"""Use OFT predict_action by properly stubbing imports."""
import os, io, math, json, sys, shutil, types, importlib.util
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

CKPT = "/root/autodl-tmp/openvla-oft-libero-object"
OFT_REPO = "/root/autodl-tmp/openvla-oft-official"
DEVICE = "cuda:0"
IMG_SIZE = 224
CROP_SCALE = 0.9

# Copy OFT modeling files
for fname in ["modeling_prismatic.py", "configuration_prismatic.py", "processing_prismatic.py"]:
    shutil.copy2(os.path.join(OFT_REPO, "prismatic/extern/hf", fname), os.path.join(CKPT, fname))

config_path = os.path.join(CKPT, "config.json")
with open(config_path) as f: config = json.load(f)
config["auto_map"] = {"AutoConfig": "configuration_prismatic.OpenVLAConfig", "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"}
with open(config_path, "w") as f: json.dump(config, f, indent=2)

# Build minimal prismatic package stubs
def make_stub(name, path=None):
    m = types.ModuleType(name)
    if path: m.__path__ = [path]
    sys.modules[name] = m
    return m

OFT_P = os.path.join(OFT_REPO, "prismatic")
make_stub("prismatic", OFT_P)
make_stub("prismatic.models", os.path.join(OFT_P, "models"))
make_stub("prismatic.extern", os.path.join(OFT_P, "extern"))
make_stub("prismatic.extern.hf", os.path.join(OFT_P, "extern/hf"))
make_stub("prismatic.training", os.path.join(OFT_P, "training"))
make_stub("prismatic.vla", os.path.join(OFT_P, "vla"))
make_stub("prismatic.vla.datasets", os.path.join(OFT_P, "vla/datasets"))
make_stub("prismatic.vla.datasets.rlds", os.path.join(OFT_P, "vla/datasets/rlds"))
make_stub("prismatic.vla.datasets.rlds.utils", os.path.join(OFT_P, "vla/datasets/rlds/utils"))

# Load constants (no problematic imports)
spec = importlib.util.spec_from_file_location("prismatic.vla.constants", os.path.join(OFT_P, "vla/constants.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["prismatic.vla.constants"] = mod; spec.loader.exec_module(mod)

# Create NormalizationType stub in data_utils (avoid dlimp import)
data_utils_stub = types.ModuleType("prismatic.vla.datasets.rlds.utils.data_utils")
data_utils_stub.NormalizationType = mod.NormalizationType  # reuse from constants
sys.modules["prismatic.vla.datasets.rlds.utils.data_utils"] = data_utils_stub

# Load train_utils (only needs constants)
spec_tu = importlib.util.spec_from_file_location("prismatic.training.train_utils", os.path.join(OFT_P, "training/train_utils.py"))
tu = importlib.util.module_from_spec(spec_tu); sys.modules["prismatic.training.train_utils"] = tu; spec_tu.loader.exec_module(tu)

# Load action_heads (needs constants)
spec_ah = importlib.util.spec_from_file_location("prismatic.models.action_heads", os.path.join(OFT_P, "models/action_heads.py"))
ah_mod = importlib.util.module_from_spec(spec_ah); sys.modules["prismatic.models.action_heads"] = ah_mod; spec_ah.loader.exec_module(ah_mod)

# Load projectors
spec_pj = importlib.util.spec_from_file_location("prismatic.models.projectors", os.path.join(OFT_P, "models/projectors.py"))
pj_mod = importlib.util.module_from_spec(spec_pj); sys.modules["prismatic.models.projectors"] = pj_mod; spec_pj.loader.exec_module(pj_mod)

# Load configuration
spec_cfg = importlib.util.spec_from_file_location("prismatic.extern.hf.configuration_prismatic", os.path.join(OFT_P, "extern/hf/configuration_prismatic.py"))
cfg_mod = importlib.util.module_from_spec(spec_cfg); sys.modules["prismatic.extern.hf.configuration_prismatic"] = cfg_mod; spec_cfg.loader.exec_module(cfg_mod)

print(f"Constants OK: ACTION_DIM={mod.ACTION_DIM}")

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
AutoConfig.register("openvla", cfg_mod.OpenVLAConfig)

print("Loading model with OFT modeling code...")
model = AutoModelForVision2Seq.from_pretrained(CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model.vision_backbone.set_num_images_in_input(2)

ckpt_path = Path(CKPT)
llm_dim = model.language_model.config.hidden_size
print(f"Model: {type(model).__name__}, LLM dim: {llm_dim}")

def load_sd(path):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

ah_files = sorted(ckpt_path.glob("action_head--*_checkpoint.pt"))
action_head = ah_mod.L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim)
action_head.load_state_dict(load_sd(ah_files[-1]))
action_head = action_head.to(DEVICE, dtype=torch.bfloat16).eval()

pp_files = sorted(ckpt_path.glob("proprio_projector--*_checkpoint.pt"))
pp = pj_mod.ProprioProjector(llm_dim=llm_dim, proprio_dim=8)
pp.load_state_dict(load_sd(pp_files[-1]))
pp = pp.to(DEVICE, dtype=torch.bfloat16).eval()

ns = json.load(open(ckpt_path / "dataset_statistics.json"))
unnorm_key = [k for k in ns if "libero" in k][0]
model.norm_stats = ns

import inspect
sig = inspect.signature(model.predict_action)
print(f"predict_action params: {list(sig.parameters.keys())}")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()["libero_object"]()
task = bm.get_task(0)
task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_states = bm.get_task_init_states(0)

env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
obs = env.set_init_state(init_states[0])
for _ in range(10):
    obs, _, _, _ = env.step([0,0,0,0,0,0,-1])

def quat2axisangle(q):
    q = np.array(q); sin_half = np.linalg.norm(q[:3]); cos_half = q[3]
    angle = 2.0 * np.arctan2(sin_half, cos_half)
    if sin_half < 1e-6: return np.zeros(3)
    axis = q[:3] / sin_half
    if angle > np.pi: angle -= 2*np.pi
    return axis * angle

def get_proprio(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)

def prepare_img(img_np):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO(); pil.save(buf, format="JPEG"); buf.seek(0)
    pil = Image.open(buf); pil.load()
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    w, h = pil.size; s = math.sqrt(CROP_SCALE)
    nw, nh = int(round(w*s)), int(round(h*s))
    l, t = (w-nw)/2, (h-nh)/2
    pil = pil.crop((int(math.floor(l)), int(math.floor(t)), int(math.floor(l))+nw, int(math.floor(t))+nh))
    return pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

agent_img = obs["agentview_image"][::-1, ::-1]
wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]
agent_pil = prepare_img(agent_img)
wrist_pil = prepare_img(wrist_img)

proprio = get_proprio(obs)
ps = ns[unnorm_key]["proprio"]
mask_p = np.array(ps.get("mask", np.ones(8, dtype=bool)))
q01_p = np.array(ps["q01"], dtype=np.float32)
q99_p = np.array(ps["q99"], dtype=np.float32)
proprio_n = np.clip(np.where(mask_p, 2*(proprio-q01_p)/(q99_p-q01_p+1e-8)-1, proprio), -1, 1)

prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"

with torch.inference_mode():
    inputs = processor(prompt, agent_pil).to(DEVICE, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
    
    action, _ = model.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False,
        proprio=proprio_n, proprio_projector=pp, action_head=action_head,
    )
    
    print(f"\n=== OFT predict_action (unnormalized, shape={action.shape}) ===")
    for i in range(action.shape[0]):
        print(f"  [{i}]: [{', '.join(f'{v:.4f}' for v in action[i])}]")

env.close()
print("Done.")
