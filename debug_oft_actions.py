"""Diagnostic: compare our OFT inference with official predict_action."""
import sys, os, json, math, io
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

CKPT = "/root/autodl-tmp/openvla-oft-libero-object"
DEVICE = "cuda:0"
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
PROPRIO_DIM = 8
CROP_SCALE = 0.9
OPENVLA_IMAGE_SIZE = 224

def quat2axisangle(quat):
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if den < 1e-7: return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(abs(w))
    return (quat[:3] * angle / den).astype(np.float32)

def extract_proprio(obs):
    return np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"]
    ]).astype(np.float32)

def normalize_proprio(proprio, norm_stats):
    mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)))
    q01, q99 = np.array(norm_stats["q01"], dtype=np.float32), np.array(norm_stats["q99"], dtype=np.float32)
    return np.clip(np.where(mask, 2.0*(proprio - q01)/(q99 - q01 + 1e-8) - 1.0, proprio), -1.0, 1.0)

def resize_image_for_policy(img_np, size=OPENVLA_IMAGE_SIZE):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO(); pil.save(buf, format="JPEG"); buf.seek(0)
    pil = Image.open(buf); pil.load()
    return np.array(pil.resize((size, size), Image.LANCZOS))

def center_crop_image(image, crop_scale=CROP_SCALE):
    if isinstance(image, np.ndarray): image = Image.fromarray(image).convert("RGB")
    w, h = image.size
    s = math.sqrt(crop_scale)
    nw, nh = int(round(w*s)), int(round(h*s))
    l, t = (w-nw)/2.0, (h-nh)/2.0
    image = image.crop((int(math.floor(l)), int(math.floor(t)),
                        int(math.floor(l))+nw, int(math.floor(t))+nh))
    return image.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.BILINEAR)

def prepare_image(img_np):
    return center_crop_image(resize_image_for_policy(img_np), CROP_SCALE)

def get_libero_image(obs): return obs["agentview_image"][::-1, ::-1].copy()
def get_libero_wrist_image(obs): return obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()

# Correct MLPResNet matching checkpoint
class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
    def forward(self, x):
        return x + self.ffn(x)

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
        for blk in self.mlp_resnet_blocks:
            x = blk(x)
        return self.fc2(self.layer_norm2(x))

class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(num_blocks=2, input_dim=input_dim*action_dim,
                               hidden_dim=hidden_dim, output_dim=action_dim)
    def predict_action(self, ah):
        B = ah.shape[0]
        x = ah.reshape(B, NUM_ACTIONS_CHUNK, -1)
        return self.model(x)

class ProprioProjector(nn.Module):
    def __init__(self, llm_dim, proprio_dim=PROPRIO_DIM):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
    def forward(self, p):
        return self.fc2(self.act_fn1(self.fc1(p)))

def _load_sd(path):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

# ── Load model ──
print("Loading model...")
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

llm_dim = model.config.text_config.hidden_size
ckpt = Path(CKPT)

ah = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim)
ah.load_state_dict(_load_sd(sorted(ckpt.glob("action_head--*_checkpoint.pt"))[-1]))
ah = ah.to(torch.bfloat16).to(DEVICE).eval()
print("Action head loaded OK")

pp = ProprioProjector(llm_dim=llm_dim)
pp.load_state_dict(_load_sd(sorted(ckpt.glob("proprio_projector--*_checkpoint.pt"))[-1]))
pp = pp.to(torch.bfloat16).to(DEVICE).eval()
print("Proprio projector loaded OK")

with open(ckpt / "dataset_statistics.json") as f:
    norm_stats = json.load(f)
unnorm_key = "libero_object_no_noops"
action_stats = norm_stats[unnorm_key]["action"]
proprio_stats = norm_stats[unnorm_key]["proprio"]

# ── Setup env ──
print("Setting up env...")
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils

task_name = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
bddl_dir, init_dir = get_libero_path("bddl_files"), get_libero_path("init_states")
bddl_path = os.path.join(bddl_dir, "libero_object", f"{task_name}.bddl")
info = BDDLUtils.get_problem_info(bddl_path)
task_desc = info["language_instruction"]
print(f"Task: '{task_desc}'")

init_states = torch.load(os.path.join(init_dir, "libero_object", f"{task_name}.pruned_init"), weights_only=False)
env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
obs = env.set_init_state(init_states[0])

raw_img = get_libero_image(obs)
raw_wrist = get_libero_wrist_image(obs)
img_pil = prepare_image(raw_img)
wrist_pil = prepare_image(raw_wrist)
proprio = extract_proprio(obs)
proprio_normed = normalize_proprio(proprio, proprio_stats)

print(f"Raw img shape: {raw_img.shape}, dtype: {raw_img.dtype}")
print(f"Proprio: {proprio}")
print(f"Proprio normed: {proprio_normed}")

# ── OUR pipeline ──
print("\n=== OUR PIPELINE ===")
with torch.inference_mode():
    prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
    print(f"Prompt: '{prompt}'")
    
    inputs = processor(prompt, img_pil).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    primary_pixels = inputs["pixel_values"]
    
    print(f"Input IDs: shape={input_ids.shape}, last 5 tokens={input_ids[0,-5:].tolist()}")
    print(f"Pixel values: shape={primary_pixels.shape}, range=[{primary_pixels.min():.3f}, {primary_pixels.max():.3f}]")
    
    if input_ids[0, -1].item() != 29871:
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], device=DEVICE, dtype=input_ids.dtype)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1,1,device=DEVICE,dtype=attention_mask.dtype)], dim=1)
        print(f"Added trailing 29871")
    
    n_prompt = input_ids.shape[-1] - 1
    n_act = ACTION_DIM * NUM_ACTIONS_CHUNK
    
    STOP = 2
    ids_ext = torch.cat([input_ids, torch.ones(1,n_act,device=DEVICE,dtype=input_ids.dtype),
                          torch.full((1,1),STOP,device=DEVICE,dtype=input_ids.dtype)], dim=1)
    attn_ext = torch.cat([attention_mask, torch.ones(1,n_act+1,device=DEVICE,dtype=attention_mask.dtype)], dim=1)
    
    embs = model.get_input_embeddings()(ids_ext)
    embs[:, -(n_act+1):-1, :] = 0
    
    p1 = model.vision_backbone(primary_pixels)
    w_in = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
    p2 = model.vision_backbone(w_in["pixel_values"])
    all_p = torch.cat([p1, p2], dim=1)
    proj = model.projector(all_p)
    
    pt = torch.tensor(proprio_normed, dtype=proj.dtype, device=DEVICE)
    pe = pp(pt.unsqueeze(0)).unsqueeze(1)
    proj = torch.cat([proj, pe], dim=1)
    n_patches = proj.shape[1]
    
    mm = torch.cat([embs[:,:1,:], proj, embs[:,1:,:]], dim=1)
    pa = torch.ones(1, n_patches, device=DEVICE, dtype=attn_ext.dtype)
    mm_attn = torch.cat([attn_ext[:,:1], pa, attn_ext[:,1:]], dim=1)
    
    print(f"n_prompt={n_prompt}, n_patches={n_patches}, multimodal={mm.shape}")
    
    out = model.language_model(inputs_embeds=mm, attention_mask=mm_attn,
                                output_hidden_states=True, return_dict=True)
    lh = out.hidden_states[-1]
    
    a_start = n_patches + n_prompt
    a_end = a_start + n_act
    ah_h = lh[:, a_start:a_end, :]
    print(f"Action hidden: [{a_start}:{a_end}], shape={ah_h.shape}")
    print(f"  stats: mean={ah_h.float().mean():.4f}, std={ah_h.float().std():.4f}")
    
    pred = ah.predict_action(ah_h)
    pred_np = pred.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().numpy()
    
    mask = np.array(action_stats.get("mask", np.ones(ACTION_DIM, dtype=bool)))
    q99, q01 = np.array(action_stats["q99"], dtype=np.float32), np.array(action_stats["q01"], dtype=np.float32)
    actions = np.where(mask, 0.5*(pred_np+1)*(q99-q01+1e-8)+q01, pred_np)
    
    print(f"\nRaw pred (before unnorm):")
    for i in range(NUM_ACTIONS_CHUNK):
        print(f"  [{i}] {pred_np[i]}")
    print(f"\nUnnormalized actions:")
    for i in range(NUM_ACTIONS_CHUNK):
        print(f"  [{i}] {actions[i]}")
    print(f"\nq01={q01}")
    print(f"q99={q99}")
    print(f"mask={mask}")
    
    # Final post-processing
    for i in range(NUM_ACTIONS_CHUNK):
        a = actions[i].copy()
        a[-1] = 2.0*a[-1] - 1.0
        a[-1] = np.sign(a[-1])
        a[-1] *= -1.0
        print(f"  final[{i}] = {a}")

# ── OFFICIAL pipeline ──
print("\n\n=== OFFICIAL PREDICT_ACTION ===")
sys.path.insert(0, "/root/autodl-tmp/openvla-oft-official")
try:
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    official = OpenVLAForActionPrediction.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval().to(DEVICE)
    official.vision_backbone.set_num_images_in_input(2)
    with open(ckpt / "dataset_statistics.json") as f:
        official.norm_stats = json.load(f)
    
    with torch.inference_mode():
        inp1 = processor(prompt, img_pil).to(DEVICE, dtype=torch.bfloat16)
        inp2 = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
        inp1["pixel_values"] = torch.cat([inp1["pixel_values"], inp2["pixel_values"]], dim=1)
        print(f"Combined pixels: {inp1['pixel_values'].shape}")
        
        act_off, hid_off = official.predict_action(
            **inp1, unnorm_key=unnorm_key, proprio=proprio_normed,
            proprio_projector=pp, action_head=ah,
        )
        print(f"\nOfficial normalized actions:")
        for i in range(NUM_ACTIONS_CHUNK):
            print(f"  [{i}] {act_off[i]}")
        
        act_unnorm = np.where(mask, 0.5*(act_off+1)*(q99-q01+1e-8)+q01, act_off)
        print(f"\nOfficial unnormalized actions:")
        for i in range(NUM_ACTIONS_CHUNK):
            print(f"  [{i}] {act_unnorm[i]}")
            
except Exception as e:
    print(f"Official method failed: {e}")
    import traceback; traceback.print_exc()

env.close()
print("\nDone.")
