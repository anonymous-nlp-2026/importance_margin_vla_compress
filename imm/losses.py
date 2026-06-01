"""Importance Margin Maximization losses.

MarginMaximizationLoss: hinge loss  max(0, delta - (s_{(k)} - s_{(k+1)})).
IMMTotalLoss:           L_action + lambda * L_margin  with linear lambda warmup.

Inputs:  (B, N) importance scores, integer k.
Depends: soft_topk.SoftTopK.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from .soft_topk import SoftTopK


class MarginMaximizationLoss(nn.Module):
    """Hinge loss: L_margin = mean(max(0, delta - (s_{(k)} - s_{(k+1)})))."""

    def __init__(self, delta: float = 1.0, temperature: float = 1.0, method: str = "sigmoid"):
        super().__init__()
        self.delta = delta
        self.soft_topk = SoftTopK(temperature=temperature, method=method)

    def forward(self, importance_scores: Tensor, k: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        mask, s_k, s_k_plus_1 = self.soft_topk(importance_scores, k)

        margin = s_k - s_k_plus_1  # (B,)
        hinge = torch.clamp(self.delta - margin, min=0.0)  # (B,)
        loss = hinge.mean()

        info = {
            "margin": margin.detach(),
            "s_k": s_k.detach(),
            "s_k_plus_1": s_k_plus_1.detach(),
            "hinge": hinge.detach(),
            "mask": mask.detach(),
        }
        return loss, info


class IMMTotalLoss(nn.Module):
    """L_total = L_action + lambda * L_margin, with linear lambda warmup."""

    def __init__(
        self,
        lambda_margin: float = 0.1,
        delta: float = 1.0,
        warmup_steps: int = 1000,
        temperature: float = 1.0,
        method: str = "sigmoid",
    ):
        super().__init__()
        self.lambda_margin = lambda_margin
        self.warmup_steps = warmup_steps
        self.margin_loss = MarginMaximizationLoss(delta=delta, temperature=temperature, method=method)

    def _get_lambda(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.lambda_margin
        return self.lambda_margin * min(1.0, step / self.warmup_steps)

    def forward(
        self, action_loss: Tensor, importance_scores: Tensor, k: int, step: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        lam = self._get_lambda(step)
        margin_loss, margin_info = self.margin_loss(importance_scores, k)
        total = action_loss + lam * margin_loss

        info = {
            **margin_info,
            "action_loss": action_loss.detach(),
            "margin_loss": margin_loss.detach(),
            "total_loss": total.detach(),
            "lambda": torch.tensor(lam),
        }
        return total, info
