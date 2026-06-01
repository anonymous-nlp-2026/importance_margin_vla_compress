"""Evaluation metrics for importance-margin token compression."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional


def compute_margin(scores: Tensor, k: int) -> Tensor:
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    return sorted_scores[:, k - 1] - sorted_scores[:, k]


def topk_preservation_rate(scores_clean: Tensor, scores_perturbed: Tensor, k: int) -> Tensor:
    _, idx_clean = torch.topk(scores_clean, k, dim=-1)
    _, idx_pert = torch.topk(scores_perturbed, k, dim=-1)

    B, N = scores_clean.shape
    mask_clean = torch.zeros(B, N, device=scores_clean.device)
    mask_pert = torch.zeros(B, N, device=scores_clean.device)
    mask_clean.scatter_(1, idx_clean, 1.0)
    mask_pert.scatter_(1, idx_pert, 1.0)

    intersection = (mask_clean * mask_pert).sum(-1)
    union = (mask_clean + mask_pert).clamp(max=1.0).sum(-1)
    return intersection / union.clamp(min=1.0)


def kendall_tau_distance(scores_clean: Tensor, scores_perturbed: Tensor, k: int) -> Tensor:
    B, N = scores_clean.shape
    _, idx_clean = torch.topk(scores_clean, k, dim=-1)
    _, idx_pert = torch.topk(scores_perturbed, k, dim=-1)

    mask_union = torch.zeros(B, N, device=scores_clean.device)
    mask_union.scatter_(1, idx_clean, 1.0)
    mask_union.scatter_(1, idx_pert, 1.0)

    taus = []
    for b in range(B):
        union_idx = mask_union[b].nonzero(as_tuple=True)[0]
        n_union = union_idx.numel()
        if n_union < 2:
            taus.append(torch.tensor(1.0, device=scores_clean.device))
            continue

        r1 = scores_clean[b, union_idx]
        r2 = scores_perturbed[b, union_idx]

        diff1 = r1.unsqueeze(0) - r1.unsqueeze(1)
        diff2 = r2.unsqueeze(0) - r2.unsqueeze(1)
        sign_prod = torch.sign(diff1) * torch.sign(diff2)

        n_pairs = n_union * (n_union - 1) // 2
        tau = sign_prod.triu(diagonal=1).sum() / max(n_pairs, 1)
        taus.append(tau)

    return torch.stack(taus)


def certified_bound(margin: Tensor, lipschitz_const: float, epsilon: float, k: int) -> Tensor:
    bound = k * epsilon * lipschitz_const / margin.clamp(min=1e-8)
    return bound.clamp(0.0, 1.0)


def margin_statistics(scores: Tensor, k: int) -> Dict[str, Tensor]:
    m = compute_margin(scores, k)
    return {
        "median": m.median(),
        "mean": m.mean(),
        "std": m.std(),
        "min": m.min(),
        "max": m.max(),
    }


@torch.no_grad()
def compute_empirical_lipschitz(
    acis_module: nn.Module,
    visual_tokens: Tensor,
    a_query: Tensor,
    num_samples: int = 100,
    epsilon: float = 0.001,
) -> float:
    was_training = acis_module.training
    acis_module.eval()
    s_clean = acis_module(a_query, visual_tokens)

    max_ratio = 0.0
    for _ in range(num_samples):
        delta = torch.randn_like(visual_tokens)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1e-8) * epsilon
        s_pert = acis_module(a_query, visual_tokens + delta)

        score_diff = (s_pert - s_clean).norm(dim=-1)
        input_diff = delta.flatten(1).norm(dim=-1)
        ratio = (score_diff / input_diff.clamp(min=1e-8)).max().item()
        max_ratio = max(max_ratio, ratio)
    if was_training:
        acis_module.train()
    return max_ratio


def l_eps_delta_ratio(
    lipschitz_const: float, epsilon: float, margin: Tensor, k: int
) -> Tensor:
    return lipschitz_const * epsilon * k / margin.clamp(min=1e-8)
