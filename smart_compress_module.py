"""Vision Token Compressor for SmolVLA (Cross-attention Pooling).

Compress N vision tokens to M tokens (M < N) using learnable queries +
cross-attention, similar to Q-Former / Perceiver Resampler.

Architecture:
  SigLIP(frozen) -> connector(frozen) -> 128 tokens
  -> VisionTokenCompressor(trainable) -> M tokens
  -> VLM+Expert(VLM frozen, Expert trainable) -> action loss

SmartCompressWrapper integrates the compressor into SmolVLA's forward pass.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )
except ImportError:
    ACTION = "action"
    OBS_STATE = "observation.state"
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: Tensor, kv: Tensor) -> Tensor:
        q = self.norm_q(queries)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_n, kv_n)
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class VisionTokenCompressor(nn.Module):
    """Cross-attention pooling: compress N vision tokens to M tokens.

    M learnable query tokens attend to all N vision tokens via multi-layer
    cross-attention, producing M compressed tokens that retain information
    from all original tokens through soft attention weighting.

    Args:
        hidden_dim: dimension of vision tokens (= text_config.hidden_size).
        num_queries: M, number of compressed output tokens.
        num_heads: attention heads per cross-attention layer.
        num_layers: number of cross-attention + FFN blocks.
        dropout: dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 960,
        num_queries: int = 32,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_tokens: Tensor) -> Tensor:
        """
        Args:
            vision_tokens: (B, N, D) concatenated vision tokens from all cameras.
        Returns:
            (B, M, D) compressed tokens.
        """
        B = vision_tokens.shape[0]
        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, vision_tokens)
        return self.output_norm(queries)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def _make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


class SmartCompressWrapper(nn.Module):
    """Wraps SmolVLAPolicy with VisionTokenCompressor.

    Data flow (training):
      images -> SigLIP+connector (frozen) -> 128 vis tokens
      -> VisionTokenCompressor (trainable) -> M tokens
      -> [M tokens, lang, state] prefix + [action+time] suffix
      -> VLM (frozen) + Expert (trainable) -> action prediction -> MSE loss
    """

    def __init__(
        self,
        smolvla_policy: Any,
        compress_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.policy = smolvla_policy
        self.base_model = self._unwrap_to_vla(smolvla_policy.model)

        hidden_dim = self.base_model.vlm_with_expert.config.text_config.hidden_size
        cfg = compress_config or {}

        self.compressor = VisionTokenCompressor(
            hidden_dim=hidden_dim,
            num_queries=cfg.get("num_queries", 32),
            num_heads=cfg.get("num_heads", 12),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )

    @staticmethod
    def _unwrap_to_vla(model: nn.Module) -> nn.Module:
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                return model.base_model.model
        except ImportError:
            pass
        return model

    # ----- vision token helpers -----

    def _get_visual_tokens(self, images: List[Tensor]) -> Tuple[Tensor, List[int]]:
        all_vis: list[Tensor] = []
        tokens_per_camera: list[int] = []
        for img in images:
            vis = self.base_model.vlm_with_expert.embed_image(img)
            all_vis.append(vis)
            tokens_per_camera.append(vis.shape[1])
        return torch.cat(all_vis, dim=1), tokens_per_camera

    # ----- prefix builder (compressed) -----

    def _build_prefix_compressed(
        self,
        compressed_vis: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build prefix embeddings substituting original vision tokens with
        M compressed tokens from the VisionTokenCompressor."""
        bm = self.base_model
        device = compressed_vis.device
        bsize = compressed_vis.shape[0]
        M = compressed_vis.shape[1]

        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks_list: list[int] = []

        # compressed vision tokens (apply same sqrt-dim scaling as original)
        dim = compressed_vis.shape[-1]
        vis_scaled = compressed_vis * torch.tensor(
            dim**0.5, dtype=compressed_vis.dtype, device=device
        )
        vis_mask = torch.ones(bsize, M, dtype=torch.bool, device=device)
        embs.append(vis_scaled)
        pad_masks.append(vis_mask)
        att_masks_list += [0] * M

        # language tokens
        lang_emb = bm.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks_list += [0] * lang_emb.shape[1]

        # state (causal block boundary)
        state_emb = bm.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        states_seq = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq, dtype=torch.bool, device=device)
        embs.append(state_emb)
        pad_masks.append(state_mask)
        att_masks_list += [1] * states_seq

        embs_cat = torch.cat(embs, dim=1)
        pad_cat = torch.cat(pad_masks, dim=1)
        att_cat = torch.tensor(att_masks_list, dtype=torch.bool, device=device)
        att_cat = att_cat.unsqueeze(0).expand(bsize, -1)

        return embs_cat, pad_cat, att_cat

    # ----- forward (training) -----

    def forward(
        self, batch: Dict[str, Tensor], step: int = 0
    ) -> Tuple[Tensor, Dict[str, Any]]:
        bm = self.base_model
        device = next(iter(self.compressor.parameters())).device

        # 1. vision tokens (frozen SigLIP + connector)
        with torch.no_grad():
            images, img_masks = self.policy.prepare_images(batch)
            vis_tokens, _ = self._get_visual_tokens(images)

        # 2. compress 128 -> M (trainable)
        vis_dtype = vis_tokens.dtype
        compressed_vis = self.compressor(vis_tokens.detach().float())
        compressed_vis = compressed_vis.to(dtype=vis_dtype)

        # 3. build prefix (use policy helpers for proper padding)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        state = self.policy.prepare_state(batch)

        prefix_embs, prefix_pad, prefix_att = self._build_prefix_compressed(
            compressed_vis, lang_tokens, lang_masks, state
        )

        # 4. flow matching noise + time
        actions = self.policy.prepare_action(batch)
        noise = bm.sample_noise(actions.shape, actions.device)
        time = bm.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 5. suffix
        suffix_embs, suffix_pad, suffix_att = bm.embed_suffix(x_t, time)

        # 6. VLM + Expert forward
        all_pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        all_att = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = _make_att_2d_masks(all_pad, all_att)
        position_ids = torch.cumsum(all_pad.to(torch.int32), dim=1) - 1

        (_, suffix_out), _ = bm.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        # 7. action projection + loss
        chunk_size = bm.config.chunk_size
        suffix_out = suffix_out[:, -chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = bm.action_out_proj(suffix_out)

        action_dim = self.policy.config.action_feature.shape[0]
        action_loss = F.mse_loss(
            u_t[:, :, :action_dim],
            v_t[:, :, :action_dim],
        )

        info = {
            "action_loss": action_loss.detach(),
            "num_compressed_tokens": compressed_vis.shape[1],
            "compression_ratio": vis_tokens.shape[1] / compressed_vis.shape[1],
        }
        return action_loss, info

    # ----- inference forward (with KV cache) -----

    @torch.no_grad()
    def forward_inference(
        self, batch: Dict[str, Tensor], noise: Optional[Tensor] = None
    ) -> Tensor:
        """Inference: denoise actions using compressed vision prefix + KV cache."""
        bm = self.base_model
        device = next(iter(self.compressor.parameters())).device

        images, img_masks = self.policy.prepare_images(batch)
        vis_tokens, _ = self._get_visual_tokens(images)
        compressed_vis = self.compressor(vis_tokens.float()).to(vis_tokens.dtype)

        lang_tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        state = self.policy.prepare_state(batch)

        prefix_embs, prefix_pad, prefix_att = self._build_prefix_compressed(
            compressed_vis, lang_tokens, lang_masks, state
        )

        prefix_att_2d = _make_att_2d_masks(prefix_pad, prefix_att)
        prefix_position_ids = torch.cumsum(prefix_pad.to(torch.int32), dim=1) - 1

        # fill KV cache with prefix
        _, past_key_values = bm.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=bm.config.use_cache,
            fill_kv_cache=True,
        )

        # iterative denoising
        bsize = state.shape[0]
        actions_shape = (bsize, bm.config.chunk_size, bm.config.max_action_dim)
        if noise is None:
            noise = bm.sample_noise(actions_shape, device)

        num_steps = bm.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise

        for s in range(num_steps):
            t_val = 1.0 + s * dt
            timestep = torch.tensor(t_val, dtype=torch.float32, device=device).expand(bsize)

            suffix_embs, suffix_pad, suffix_att = bm.embed_suffix(x_t, timestep)
            suffix_len = suffix_pad.shape[1]
            prefix_len = prefix_pad.shape[1]

            prefix_pad_2d = prefix_pad[:, None, :].expand(bsize, suffix_len, prefix_len)
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

            prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
            pos_ids = prefix_offsets + torch.cumsum(suffix_pad.to(torch.int32), dim=1) - 1

            outputs_embeds, _ = bm.vlm_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=bm.config.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -bm.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = bm.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t

        return x_t

    # ----- baseline forward (no compression, for comparison) -----

    def forward_baseline(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        loss, loss_dict = self.policy.forward(batch)
        return loss, {"action_loss": loss.detach()}
