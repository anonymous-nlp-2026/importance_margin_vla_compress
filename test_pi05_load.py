import sys, dataclasses, os
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HOME"] = "./cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, './openpi-repo/src')

import torch
import safetensors.torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

@dataclasses.dataclass
class Pi05Config:
    pi05: bool = True
    paligemma_variant: str = 'gemma_2b'
    action_expert_variant: str = 'gemma_300m'
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    dtype: str = 'bfloat16'
    pytorch_compile_mode: str = None

print("Creating model...")
model = PI0Pytorch(config=Pi05Config())
print(f"Model created. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

print("Loading weights...")
safetensors.torch.load_model(model, './pi05-libero-finetuned/model.safetensors')
print("Weights loaded successfully")

model = model.to("cuda:0")
model.eval()
print("Model on cuda:0, eval mode")

# Check vision token count
# SigLIP So400m/14 with 224x224 input
n_patches = (224 // 14) ** 2
print(f"\nVision token count: {n_patches} per image (224/14 = 16, 16x16 = {n_patches})")
print(f"3 images (base + left_wrist + right_wrist): 3 x {n_patches} = {3*n_patches} total vision tokens")
print(f"For LIBERO (right_wrist masked): 2 real x {n_patches} = {2*n_patches} real + {n_patches} masked")
