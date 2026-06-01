"""Quick test: load OFT model with TIMM 0.9.16 and verify vision backbone works."""
import os, json, math, io, sys
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
from PIL import Image
from pathlib import Path

CKPT = "/root/autodl-tmp/openvla-oft-libero-object"
DEVICE = "cuda:0"

print(f"Loading model from {CKPT}...")
import timm
print(f"TIMM version: {timm.__version__}")

from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model.vision_backbone.set_num_images_in_input(2)
print("Model loaded successfully!")

# Create two dummy images and test forward
dummy_img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
dummy_img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the alphabet soup and place it in the basket?\nOut:"
inp1 = processor(prompt, dummy_img1).to(DEVICE, dtype=torch.bfloat16)
inp2 = processor(prompt, dummy_img2).to(DEVICE, dtype=torch.bfloat16)
dual_pixels = torch.cat([inp1["pixel_values"], inp2["pixel_values"]], dim=1)
print(f"Dual pixel_values shape: {dual_pixels.shape}")

with torch.inference_mode():
    patches = model.vision_backbone(dual_pixels)
    print(f"Vision output type: {type(patches)}, shape: {patches.shape}")
    print(f"Patches stats: mean={patches.float().mean():.4f}, std={patches.float().std():.4f}")

print("\nVision backbone works correctly with TIMM 0.9.16!")
