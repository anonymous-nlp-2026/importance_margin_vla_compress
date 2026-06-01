"""Margin-aware token selection: soft masking (train) / hard prune+merge (inference)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional

from .soft_topk import SoftTopK


class MarginAwareTokenSelector(nn.Module):
    def __init__(self, temperature: float = 1.0, merge_weight_temp: float = 0.5):
        super().__init__()
        self.soft_topk = SoftTopK(temperature=temperature, method="sigmoid")
        self.merge_weight_temp = merge_weight_temp

    def forward(
        self,
        importance_scores: Tensor,
        visual_tokens: Tensor,
        k: int,
        training: bool = True,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if training:
            return self._soft_forward(importance_scores, visual_tokens, k)
        return self._hard_forward(importance_scores, visual_tokens, k)

    def _soft_forward(
        self, scores: Tensor, tokens: Tensor, k: int
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        mask, s_k, s_k1 = self.soft_topk(scores, k)
        selected = tokens * mask.unsqueeze(-1)
        info = {
            "mode": torch.tensor(0),
            "mask_sum": mask.sum(-1).detach(),
            "margin": (s_k - s_k1).detach(),
        }
        return selected, mask, info

    def _hard_forward(
        self, scores: Tensor, tokens: Tensor, k: int
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        B, N, D = tokens.shape

        sorted_scores, sorted_idx = torch.sort(scores, dim=-1, descending=True)
        s_k = sorted_scores[:, k - 1]          # (B,) k-th largest score
        s_k_plus_1 = sorted_scores[:, k]       # (B,) (k+1)-th largest score
        delta_k = s_k - s_k_plus_1             # (B,) margin

        threshold_high = (s_k + delta_k / 2).unsqueeze(-1)   # (B, 1)
        threshold_low = (s_k - delta_k / 2).unsqueeze(-1)    # (B, 1)

        preserve_idx = sorted_idx[:, :k]                                 # (B, k)
        is_topk = torch.zeros(B, N, device=scores.device, dtype=torch.bool)
        is_topk.scatter_(1, preserve_idx, True)

        borderline_mask = ~is_topk & (scores >= threshold_low) & (scores <= threshold_high)
        prune_mask = ~is_topk & (scores < threshold_low)

        binary_mask = is_topk.float()
        selected = tokens * binary_mask.unsqueeze(-1)  # (B, N, D) - same shape as soft

        info = {
            "mode": torch.tensor(1),  # 1 = hard
            "num_preserved": is_topk.sum(-1).float().detach(),
            "num_pruned": prune_mask.sum(-1).float().detach(),
            "num_merged": borderline_mask.sum(-1).float().detach(),
            "delta_k": delta_k.detach(),
        }
        return selected, binary_mask, info

    def _merge_borderline_tokens(
        self,
        tokens: Tensor,
        scores: Tensor,
        preserved_tokens: Tensor,
        preserve_idx: Tensor,
        borderline_mask: Tensor,
    ) -> Tensor:
        B, N, D = tokens.shape
        K = preserved_tokens.shape[1]

        num_borderline = borderline_mask.sum(-1)
        if num_borderline.sum() == 0:
            return preserved_tokens

        preserved_norm = F.normalize(preserved_tokens, dim=-1)
        tokens_norm = F.normalize(tokens, dim=-1)

        sim = torch.bmm(tokens_norm, preserved_norm.transpose(1, 2))

        best_match = sim.argmax(dim=-1)

        preserve_scores = torch.gather(scores, 1, preserve_idx)

        merged = preserved_tokens.clone()
        for b in range(B):
            border_idx = borderline_mask[b].nonzero(as_tuple=True)[0]
            if border_idx.numel() == 0:
                continue
            for idx in border_idx:
                target = best_match[b, idx].item()
                w_border = torch.sigmoid(scores[b, idx] / (self.merge_weight_temp + 1e-8))
                w_orig = torch.sigmoid(preserve_scores[b, target] / (self.merge_weight_temp + 1e-8))
                total_w = w_orig + w_border + 1e-8
                merged[b, target] = (w_orig / total_w) * merged[b, target] + (w_border / total_w) * tokens[b, idx]

        return merged
