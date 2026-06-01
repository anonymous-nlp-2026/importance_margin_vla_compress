"""
Geometric Prior Pruning for SmolVLA

Training-free vision token selection based on Sobel edge detection.
Keeps tokens whose spatial region has high image gradient (object boundaries).

SmolVLA architecture:
  SigLIP (512x512, patch_size=16) -> 32x32 = 1024 tokens
  Connector pixel_shuffle (scale_factor=4) -> 8x8 = 64 tokens per camera
  LIBERO: 2 cameras -> 128 total visual tokens

Each 8x8 output token maps to a 4x4 block of SigLIP patches = 64x64 pixels
in the 512x512 input image = 32x32 pixels in the raw 256x256 observation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

GRID_H = 8
GRID_W = 8
TOKENS_PER_CAMERA = GRID_H * GRID_W  # 64


def compute_sobel_importance(
    image: np.ndarray,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
) -> np.ndarray:
    # 180-deg flip to match SmolVLA image preprocessing (torch.flip dims=[2,3])
    image = image[::-1, ::-1].copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    H, W = magnitude.shape
    cell_h = H / grid_h
    cell_w = W / grid_w

    importance = np.zeros(grid_h * grid_w, dtype=np.float64)
    for i in range(grid_h):
        for j in range(grid_w):
            r0 = int(i * cell_h)
            r1 = int((i + 1) * cell_h)
            c0 = int(j * cell_w)
            c1 = int((j + 1) * cell_w)
            importance[i * grid_w + j] = magnitude[r0:r1, c0:c1].mean()

    vmin, vmax = importance.min(), importance.max()
    if vmax > vmin:
        importance = (importance - vmin) / (vmax - vmin)
    else:
        importance[:] = 1.0

    return importance


def compute_geometric_mask(
    images: List[np.ndarray],
    k_ratio: float,
    device: torch.device,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
) -> torch.Tensor:
    all_imp = []
    for img in images:
        imp = compute_sobel_importance(img, grid_h, grid_w)
        all_imp.append(imp)

    importance = np.concatenate(all_imp)
    importance_t = torch.from_numpy(importance).float().to(device)

    N = importance_t.shape[0]
    k = max(1, int(N * k_ratio))
    _, topk_idx = torch.topk(importance_t, k)

    mask = torch.zeros(1, N, device=device)
    mask[0, topk_idx] = 1.0

    return mask


def find_connector(model):
    for name, module in model.named_modules():
        if name.endswith(".connector") or "multi_modal_projector" in name:
            return module
    return None


def register_geometric_hook(
    connector,
    mask: torch.Tensor,
    method: str = "true_removal",
):
    state = {"offset": 0}

    def hook_fn(module, input, output):
        tokens = output[0] if isinstance(output, tuple) else output
        if tokens.dim() != 3:
            return output

        n = tokens.shape[1]
        offset = state["offset"]
        if offset + n > mask.shape[1]:
            return output

        cam_mask = mask[:, offset : offset + n]
        state["offset"] = offset + n

        B, N, D = tokens.shape
        keep_mask = cam_mask[:B]

        if method == "true_removal":
            kept = keep_mask[0].bool()
            masked = tokens[:, kept, :]
        else:
            masked = tokens * keep_mask.unsqueeze(-1)

        if isinstance(output, tuple):
            return (masked,) + output[1:]
        return masked

    return connector.register_forward_hook(hook_fn)


class GeometricPriorPruner:
    def __init__(
        self,
        model,
        k_ratio: float = 0.5,
        method: str = "true_removal",
    ):
        self.k_ratio = k_ratio
        self.method = method
        self.connector = find_connector(model)
        if self.connector is None:
            raise RuntimeError("Cannot find connector module in model")
        self._current_images: List[np.ndarray] = []
        self._hook_handle = None
        self._last_mask: Optional[torch.Tensor] = None
        self._last_per_camera_importance: List[np.ndarray] = []

    def set_current_images(self, images: List[np.ndarray]):
        self._current_images = images

    def attach(self, device: torch.device) -> torch.Tensor:
        if not self._current_images:
            raise RuntimeError("No images set. Call set_current_images() first.")

        self._last_per_camera_importance = []
        for img in self._current_images:
            imp = compute_sobel_importance(img)
            self._last_per_camera_importance.append(imp)

        mask = compute_geometric_mask(
            self._current_images, self.k_ratio, device
        )
        self._last_mask = mask

        self._hook_handle = register_geometric_hook(
            self.connector, mask, method=self.method
        )
        return mask

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def remove(self):
        self.detach()

    def get_importance_stats(self) -> dict:
        if not self._last_per_camera_importance:
            return {}
        stats = {}
        for i, imp in enumerate(self._last_per_camera_importance):
            stats[f"cam{i}_mean"] = float(imp.mean())
            stats[f"cam{i}_std"] = float(imp.std())
            stats[f"cam{i}_min"] = float(imp.min())
            stats[f"cam{i}_max"] = float(imp.max())
            stats[f"cam{i}_nonzero"] = int((imp > 0.01).sum())
        if self._last_mask is not None:
            stats["total_kept"] = int(self._last_mask.sum().item())
            stats["total_tokens"] = self._last_mask.shape[1]
        return stats
