"""IMMSmolVLAWrapper — wraps SmolVLA forward pass with ACIS scoring + token selection.

Data flow:
  Original:  images -> SigLIP -> connector -> embed_prefix -> VLM+Expert -> loss
  Modified:  images -> SigLIP -> connector -> [ACIS -> soft top-k -> mask] -> rebuild_prefix -> VLM+Expert -> loss + L_margin

Takes a full SmolVLAPolicy (not just model) to use its batch preprocessing.
Replicates embed_prefix internally to inject masked visual tokens before the
VLM+Expert forward pass, without modifying any lerobot source code.

Depends: acis.ACIS, losses.IMMTotalLoss, token_selection.MarginAwareTokenSelector.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple

from .acis import ACIS
from .losses import IMMTotalLoss
from .token_selection import MarginAwareTokenSelector

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


def _pad_tensor(tensor: Tensor, max_len: int, pad_value: float = 0) -> Tensor:
    """Pad tensor along dim=1 to max_len."""
    b, d = tensor.shape[:2]
    if d >= max_len:
        return tensor
    padded = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded[:, :d] = tensor
    return padded


def _make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Build 2D causal/prefix attention mask from 1D pad + attention-region masks.

    att_masks uses cumsum convention: 0 = bidirectional block, 1 = new causal block.
    """
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


class IMMSmolVLAWrapper(nn.Module):
    """Wraps SmolVLAPolicy with ACIS importance scoring and margin-based token selection.

    Args:
        smolvla_policy: SmolVLAPolicy instance (may have LoRA applied via PeftModel).
        acis_config: kwargs for ACIS (visual_dim, action_dim, num_heads, dropout).
        imm_config: kwargs for IMMTotalLoss + k_ratio.
    """

    def __init__(
        self,
        smolvla_policy: Any,
        acis_config: Optional[Dict[str, Any]] = None,
        imm_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.policy = smolvla_policy
        self.base_model = self._unwrap_to_vla(smolvla_policy.model)

        acis_config = dict(acis_config or {})
        imm_config = dict(imm_config or {})

        if "visual_dim" not in acis_config or "action_dim" not in acis_config:
            vlm_hidden = self.base_model.vlm_with_expert.config.text_config.hidden_size
            expert_hidden = self.base_model.vlm_with_expert.expert_hidden_size
            acis_config.setdefault("visual_dim", vlm_hidden)
            acis_config.setdefault("action_dim", expert_hidden)

        self.acis = ACIS(**acis_config)

        self.k_ratio = imm_config.pop("k_ratio", 0.5)
        self.gradient_isolation = imm_config.pop("gradient_isolation", False)
        selector_temp = imm_config.pop("selector_temperature", imm_config.get("temperature", 1.0))

        self.imm_loss = IMMTotalLoss(**imm_config)
        self.token_selector = MarginAwareTokenSelector(temperature=selector_temp)

    @staticmethod
    def _unwrap_to_vla(model: nn.Module) -> nn.Module:
        """Unwrap PeftModel -> LoraModel -> VLAFlowMatching."""
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                return model.base_model.model
        except ImportError:
            pass
        return model

    def _get_visual_tokens(self, images: List[Tensor]) -> Tuple[Tensor, List[int]]:
        """Encode each camera image through SigLIP+connector.

        Returns:
            vis_tokens: (B, sum(n_i), D) concatenated visual tokens from all cameras.
            tokens_per_camera: list of token counts per camera.
        """
        all_vis = []
        tokens_per_camera = []
        for img in images:
            vis = self.base_model.vlm_with_expert.embed_image(img)
            all_vis.append(vis)
            tokens_per_camera.append(vis.shape[1])
        return torch.cat(all_vis, dim=1), tokens_per_camera

    def _build_prefix_with_vis_tokens(
        self,
        vis_tokens: Tensor,
        tokens_per_camera: List[int],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Reconstruct prefix embeddings using pre-computed (possibly masked) visual tokens.

        Replicates VLAFlowMatching.embed_prefix() but substitutes the visual token
        computation with externally provided tokens (after ACIS masking).

        Returns:
            embs:      (B, prefix_length, D)
            pad_masks: (B, prefix_length) bool
            att_masks: (B, prefix_length) bool/float
        """
        bm = self.base_model
        device = vis_tokens.device
        bsize = vis_tokens.shape[0]

        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks_list: list[int] = []

        offset = 0
        for cam_idx, (n_tokens, img_mask) in enumerate(zip(tokens_per_camera, img_masks)):
            if bm.add_image_special_tokens:
                start_tok = (
                    bm.vlm_with_expert.embed_language_tokens(
                        bm.global_image_start_token.to(device=device)
                    )
                    .unsqueeze(0)
                    .expand(bsize, -1, -1)
                )
                start_mask = torch.ones(bsize, start_tok.shape[1], dtype=torch.bool, device=device)
                embs.append(start_tok)
                pad_masks.append(start_mask)
                att_masks_list += [0] * start_tok.shape[1]

            img_emb = vis_tokens[:, offset : offset + n_tokens, :]
            offset += n_tokens

            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=device
            )

            cam_mask = img_mask[:, None].expand(bsize, n_tokens)
            embs.append(img_emb)
            pad_masks.append(cam_mask)
            att_masks_list += [0] * n_tokens

            if bm.add_image_special_tokens:
                end_tok = (
                    bm.vlm_with_expert.embed_language_tokens(
                        bm.image_end_token.to(device=device)
                    )
                    .unsqueeze(0)
                    .expand(bsize, -1, -1)
                )
                end_mask = torch.ones(bsize, end_tok.shape[1], dtype=torch.bool, device=device)
                embs.append(end_tok)
                pad_masks.append(end_mask)
                att_masks_list += [0] * end_tok.shape[1]

        lang_emb = bm.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks_list += [0] * lang_emb.shape[1]

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
        att_cat = torch.tensor(att_masks_list, dtype=torch.bool, device=device).unsqueeze(0)

        prefix_len = bm.prefix_length
        if pad_cat.shape[1] < prefix_len:
            embs_cat = _pad_tensor(embs_cat, prefix_len, pad_value=0)
            pad_cat = _pad_tensor(pad_cat, prefix_len, pad_value=0)
            att_cat = _pad_tensor(att_cat, prefix_len, pad_value=0)

        att_cat = att_cat.expand(bsize, -1)
        return embs_cat, pad_cat, att_cat

    def forward(
        self, batch: Dict[str, Tensor], step: int = 0
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Full IMM forward: ACIS scoring -> token selection -> VLM+Expert -> action + margin loss."""
        bm = self.base_model

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.policy.prepare_action(batch)

        vis_tokens, tokens_per_cam = self._get_visual_tokens(images)

        noise = bm.sample_noise(actions.shape, actions.device)
        time = bm.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_embs, suffix_pad, suffix_att = bm.embed_suffix(x_t, time)

        vis_for_acis = vis_tokens.detach() if self.gradient_isolation else vis_tokens
        suffix_for_acis = suffix_embs.detach() if self.gradient_isolation else suffix_embs
        importance_scores = self.acis(suffix_for_acis, vis_for_acis)

        N_vis = vis_tokens.shape[1]
        k = max(1, int(N_vis * self.k_ratio))
        scores_for_select = importance_scores.detach() if self.gradient_isolation else importance_scores
        selected_tokens, mask, sel_info = self.token_selector(
            scores_for_select, vis_tokens, k, training=self.training
        )

        prefix_embs, prefix_pad, prefix_att = self._build_prefix_with_vis_tokens(
            selected_tokens, tokens_per_cam, img_masks, lang_tokens, lang_masks, state
        )

        all_pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        all_att = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = _make_att_2d_masks(all_pad, all_att)
        position_ids = torch.cumsum(all_pad, dim=1) - 1

        (_, suffix_out), _ = bm.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        chunk_size = bm.config.chunk_size
        suffix_out = suffix_out[:, -chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = bm.action_out_proj(suffix_out)

        action_dim = self.policy.config.action_feature.shape[0]
        action_loss = F.mse_loss(
            u_t[:, :, :action_dim],
            v_t[:, :, :action_dim],
        )

        total_loss, imm_info = self.imm_loss(action_loss, importance_scores, k, step)

        info = {
            **sel_info,
            **imm_info,
            "importance_scores": importance_scores.detach(),
            "k": k,
            "k_ratio": self.k_ratio,
            "lipschitz_estimate": self.acis.get_lipschitz_estimate(),
        }
        return total_loss, info

    def forward_acis_only(
        self, batch: Dict[str, Tensor], step: int = 0
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Two-stage Stage 1: train ACIS with margin_loss only, skip VLM forward."""
        bm = self.base_model

        with torch.no_grad():
            images, _ = self.policy.prepare_images(batch)
            vis_tokens, _ = self._get_visual_tokens(images)
            actions = self.policy.prepare_action(batch)

            noise = bm.sample_noise(actions.shape, actions.device)
            time = bm.sample_time(actions.shape[0], actions.device)
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            suffix_embs, _, _ = bm.embed_suffix(x_t, time)

        importance_scores = self.acis(suffix_embs, vis_tokens)

        N_vis = vis_tokens.shape[1]
        k = max(1, int(N_vis * self.k_ratio))

        margin_loss, margin_info = self.imm_loss.margin_loss(importance_scores, k)

        info = {
            **margin_info,
            "action_loss": torch.tensor(0.0, device=margin_loss.device),
            "margin_loss": margin_loss.detach(),
            "total_loss": margin_loss.detach(),
            "lambda": torch.tensor(1.0),
            "importance_scores": importance_scores.detach(),
            "k": k,
            "k_ratio": self.k_ratio,
            "lipschitz_estimate": self.acis.get_lipschitz_estimate(),
        }
        return margin_loss, info

    def forward_baseline(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
        """Baseline forward: delegate to the original SmolVLAPolicy (no ACIS)."""
        loss, loss_dict = self.policy.forward(batch)
        return loss, {"action_loss": loss.detach().item() if loss.dim() == 0 else loss.detach()}

    @torch.no_grad()
    def compute_importance_scores(
        self, batch: Dict[str, Tensor],
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """Extract ACIS scores without running full forward (for evaluation)."""
        bm = self.base_model

        images, _ = self.policy.prepare_images(batch)
        vis_tokens, _ = self._get_visual_tokens(images)

        actions = self.policy.prepare_action(batch)
        if noise is None:
            noise = bm.sample_noise(actions.shape, actions.device)
        if time is None:
            time = bm.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        suffix_embs, _, _ = bm.embed_suffix(x_t, time)
        scores = self.acis(suffix_embs, vis_tokens)

        N_vis = vis_tokens.shape[1]
        k = max(1, int(N_vis * self.k_ratio))
        return scores, k
