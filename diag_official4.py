"""Compare: use OFT predict_action. Fully bypass prismatic init chains."""
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
    src = os.path.join(OFT_REPO, "prismatic/extern/hf", fname)
    dst = os.path.join(CKPT, fname)
    bak = dst + ".orig"
    if not os.path.exists(bak):
        shutil.copy2(dst, bak)
    shutil.copy2(src, dst)

config_path = os.path.join(CKPT, "config.json")
with open(config_path) as f:
    config = json.load(f)
config["auto_map"] = {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

# Create stub modules to avoid import chain
def make_stub(name, path=None):
    m = types.ModuleType(name)
    if path:
        m.__path__ = [path]
    sys.modules[name] = m
    return m

OFT = OFT_REPO
make_stub("prismatic", os.path.join(OFT, "prismatic"))
make_stub("prismatic.models", os.path.join(OFT, "prismatic/models"))
make_stub("prismatic.vla", os.path.join(OFT, "prismatic/vla"))
make_stub("prismatic.extern", os.path.join(OFT, "prismatic/extern"))
make_stub("prismatic.extern.hf", os.path.join(OFT, "prismatic/extern/hf"))

# Import constants directly from file
spec = importlib.util.spec_from_file_location("prismatic.vla.constants", os.path.join(OFT, "prismatic/vla/constants.py"))
constants = importlib.util.module_from_spec(spec)
sys.modules["prismatic.vla.constants"] = constants
spec.loader.exec_module(constants)
print(f"ACTION_DIM={constants.ACTION_DIM}, NUM_ACTIONS_CHUNK={constants.NUM_ACTIONS_CHUNK}")

# Import data utils for NormalizationType
spec2 = importlib.util.spec_from_file_location("prismatic.vla.datasets.rlds.utils.data_utils", os.path.join(OFT, "prismatic/vla/datasets/rlds/utils/data_utils.py"))
data_utils = importlib.util.module_from_spec(spec2)
sys.modules["prismatic.vla.datasets.rlds.utils.data_utils"] = data_utils
spec2.loader.exec_module(data_utils)

# Import action_heads
spec3 = importlib.util.spec_from_file_location("prismatic.models.action_heads", os.path.join(OFT, "prismatic/models/action_heads.py"))
action_heads_mod = importlib.util.module_from_spec(spec3)
sys.modules["prismatic.models.action_heads"] = action_heads_mod
spec3.loader.exec_module(action_heads_mod)

# Import projectors
spec4 = importlib.util.spec_from_file_location("prismatic.models.projectors", os.path.join(OFT, "prismatic/models/projectors.py"))
projectors_mod = importlib.util.module_from_spec(spec4)
sys.modules["prismatic.models.projectors"] = projectors_mod
spec4.loader.exec_module(projectors_mod)

# Import configuration
spec5 = importlib.util.spec_from_file_location("prismatic.extern.hf.configuration_prismatic", os.path.join(OFT, "prismatic/extern/hf/configuration_prismatic.py"))
config_mod = importlib.util.module_from_spec(spec5)
sys.modules["prismatic.extern.hf.configuration_prismatic"] = config_mod
spec5.loader.exec_module(config_mod)

L1RegressionActionHead = action_heads_mod.L1RegressionActionHead
ProprioProjector = projectors_mod.ProprioProjector
OpenVLAConfig = config_mod.OpenVLAConfig

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
AutoConfig.register("openvla", OpenVLAConfig)

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model.vision_backbone.set_num_images_in_input(2)

ckpt_path = Path(CKPT)
llm_dim = model.language_model.config.hidden_size
print(f"LLM dim: {llm_dim}, model class: {type(model).__name__}")

def load_sd(path):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

ah_files = sorted(ckpt_path.glob("action_head--*_checkpoint.pt"))
action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim)
action_head.load_state_dict(load_sd(ah_files[-1]))
action_head = action_head.to(DEVICE, dtype=torch.bfloat16).eval()

pp_files = sorted(ckpt_path.glob("proprio_projector--*_checkpoint.pt"))
pp = ProprioProjector(llm_dim=llm_dim, proprio_dim=8)
pp.load_state_dict(load_sd(pp_files[-1]))
pp = pp.to(DEVICE, dtype=torch.bfloat16).eval()

ns = json.load(open(ckpt_path / "dataset_statistics.json"))
unnorm_key = [k for k in ns if "libero" in k][0]
model.norm_stats = ns
print(f"unnorm_key: {unnorm_key}")

import inspect
sig = inspect.signature(model.predict_action)
params = list(sig.parameters.keys())
print(f"predict_action params: {params}")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()["libero_object"]()
task = bm.get_task(0)
task_desc = task.language
task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_states = bm.get_task_init_states(0)

env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
obs = env.set_init_state(init_states[0])
for _ in range(10):
    obs, _, _, _ = env.step([0,0,0,0,0,0,-1])

def quat2axisangle(q):
    q = np.array(q)
    sin_half = np.linalg.norm(q[:3])
    cos_half = q[3]
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
    w, h = pil.size
    s = math.sqrt(CROP_SCALE)
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

prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
print(f"Prompt: '{prompt}'")

with torch.inference_mode():
    inputs = processor(prompt, agent_pil).to(DEVICE, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
    
    print(f"pixel_values: {inputs['pixel_values'].shape}")
    
    if "action_head" in params:
        action, _ = model.predict_action(
            **inputs, unnorm_key=unnorm_key, do_sample=False,
            proprio=proprio_n, proprio_projector=pp, action_head=action_head,
        )
        label = "OFT"
    else:
        action, _ = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        label = "OLD_DISCRETE"
    
    print(f"\n=== {label} predict_action output ===")
    for i in range(min(action.shape[0], 8)):
        print(f"  [{i}]: [{', '.join(f'{v:.4f}' for v in action[i])}]")
    
    print(f"\n=== AFTER GRIPPER POST-PROCESS ===")
    for i in range(min(action.shape[0], 8)):
        a = action[i].copy()
        a[-1] = 2*a[-1] - 1
        a[-1] = np.sign(a[-1])
        a[-1] *= -1
        print(f"  [{i}]: [{', '.join(f'{v:.4f}' for v in a)}]")

env.close()
print("\nDone.")
