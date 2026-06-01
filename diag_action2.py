"""Diagnostic: print raw action outputs from OFT model for first step."""
import os, io, math, json
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

CKPT = "./openvla-oft-libero-object"
DEVICE = "cuda:0"
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
PROPRIO_DIM = 8
CROP_SCALE = 0.9
IMG_SIZE = 224

class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
    def forward(self, x): return x + self.ffn(x)

class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for blk in self.mlp_resnet_blocks: x = blk(x)
        return self.fc2(self.layer_norm2(x))

class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim)
    def predict_action(self, h):
        return self.model(h.reshape(h.shape[0], NUM_ACTIONS_CHUNK, -1))

class ProprioProjector(nn.Module):
    def __init__(self, llm_dim, proprio_dim=PROPRIO_DIM):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
    def forward(self, proprio):
        return self.fc2(self.act_fn1(self.fc1(proprio)))

def load_sd(path):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

from transformers import AutoModelForImageTextToText, AutoProcessor
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

ckpt_path = Path(CKPT)
llm_dim = model.language_model.config.hidden_size

ah_files = sorted(ckpt_path.glob("action_head--*_checkpoint.pt"))
action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim)
action_head.load_state_dict(load_sd(ah_files[-1]))
action_head = action_head.to(DEVICE, dtype=torch.bfloat16).eval()

pp_files = sorted(ckpt_path.glob("proprio_projector--*_checkpoint.pt"))
pp = ProprioProjector(llm_dim=llm_dim)
pp.load_state_dict(load_sd(pp_files[-1]))
pp = pp.to(DEVICE, dtype=torch.bfloat16).eval()

ns = json.load(open(ckpt_path / "dataset_statistics.json"))
unnorm_key = [k for k in ns if "libero" in k][0]
norm_stats = ns[unnorm_key]
print(f"unnorm_key: {unnorm_key}")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()["libero_object"]()
task = bm.get_task(0)
task_desc = task.language
task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_states = bm.get_task_init_states(0)
print(f"Task: '{task_desc}'")
print(f"BDDL: {task_bddl}")

env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
obs = env.set_init_state(init_states[0])

# Wait 10 steps
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
ps = norm_stats["proprio"]
mask_p = np.array(ps.get("mask", np.ones(8, dtype=bool)))
q01_p = np.array(ps["q01"], dtype=np.float32)
q99_p = np.array(ps["q99"], dtype=np.float32)
proprio_n = np.clip(np.where(mask_p, 2*(proprio-q01_p)/(q99_p-q01_p+1e-8)-1, proprio), -1, 1)
print(f"Proprio raw: {proprio}")
print(f"Proprio normed: {proprio_n}")

prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
print(f"Prompt: '{prompt}'")

with torch.inference_mode():
    inputs = processor(prompt, agent_pil).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]
    px_primary = inputs["pixel_values"]
    
    wrist_in = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
    
    # Process images separately through vision backbone (both have num_images_in_input=1)
    patches_primary = model.vision_backbone(px_primary)
    patches_wrist = model.vision_backbone(wrist_in["pixel_values"])
    all_patches = torch.cat([patches_primary, patches_wrist], dim=1)
    projected = model.projector(all_patches)
    
    print(f"pixel_values: primary={px_primary.shape}")
    print(f"Patches: primary={patches_primary.shape}, wrist={patches_wrist.shape}, total={all_patches.shape}")
    print(f"Projected: {projected.shape}")
    print(f"input_ids: {input_ids.shape}, last 5 tokens: {input_ids[0, -5:].tolist()}")
    
    if input_ids[0, -1].item() != 29871:
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], device=DEVICE, dtype=input_ids.dtype)], dim=1)
        attn = torch.cat([attn, torch.ones(1,1,device=DEVICE,dtype=attn.dtype)], dim=1)
        print("Added 29871")
    
    n_prompt = input_ids.shape[-1] - 1
    n_act = ACTION_DIM * NUM_ACTIONS_CHUNK
    
    ids_ext = torch.cat([input_ids, torch.ones(1,n_act,device=DEVICE,dtype=input_ids.dtype), torch.full((1,1),2,device=DEVICE,dtype=input_ids.dtype)], dim=1)
    attn_ext = torch.cat([attn, torch.ones(1,n_act+1,device=DEVICE,dtype=attn.dtype)], dim=1)
    
    embeds = model.get_input_embeddings()(ids_ext)
    embeds[:, -(n_act+1):-1, :] = 0
    
    # Proprio
    pt = torch.tensor(proprio_n, dtype=projected.dtype, device=DEVICE)
    pe = pp(pt.unsqueeze(0)).unsqueeze(1)
    projected = torch.cat([projected, pe], dim=1)
    n_patches = projected.shape[1]
    
    mm = torch.cat([embeds[:,:1,:], projected, embeds[:,1:,:]], dim=1)
    mm_attn = torch.cat([attn_ext[:,:1], torch.ones(1,n_patches,device=DEVICE,dtype=attn_ext.dtype), attn_ext[:,1:]], dim=1)
    
    print(f"n_prompt={n_prompt}, n_patches={n_patches}")
    print(f"Multimodal: {mm.shape}")
    
    out = model.language_model(inputs_embeds=mm, attention_mask=mm_attn, output_hidden_states=True, return_dict=True)
    lh = out.hidden_states[-1]
    
    a_s = n_patches + n_prompt
    a_e = a_s + n_act
    action_h = lh[:, a_s:a_e, :]
    print(f"Action extraction: [{a_s}:{a_e}], shape={action_h.shape}")
    print(f"Hidden norm: {action_h.float().norm(dim=-1).mean():.4f}")
    
    pred = action_head.predict_action(action_h)
    pred_np = pred.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().numpy()
    
    print(f"\n=== NORMALIZED (raw) ===")
    for i in range(NUM_ACTIONS_CHUNK):
        print(f"  [{i}]: [{', '.join(f'{v:.4f}' for v in pred_np[i])}]")
    
    astats = norm_stats["action"]
    amask = np.array(astats.get("mask", np.ones(ACTION_DIM, dtype=bool)))
    aq99 = np.array(astats["q99"], dtype=np.float32)
    aq01 = np.array(astats["q01"], dtype=np.float32)
    actions = np.where(amask, 0.5*(pred_np+1)*(aq99-aq01+1e-8)+aq01, pred_np)
    
    print(f"\n=== UNNORMALIZED ===")
    for i in range(NUM_ACTIONS_CHUNK):
        print(f"  [{i}]: [{', '.join(f'{v:.4f}' for v in actions[i])}]")
    
    print(f"\nq01={aq01}")
    print(f"q99={aq99}")
    print(f"mask={amask}")
    
    print(f"\n=== AFTER GRIPPER POST-PROCESS ===")
    for i in range(NUM_ACTIONS_CHUNK):
        a = actions[i].copy()
        a[-1] = 2*a[-1] - 1
        a[-1] = np.sign(a[-1])
        a[-1] *= -1
        print(f"  [{i}]: [{', '.join(f'{v:.4f}' for v in a)}]")

env.close()
print("\nDone.")
