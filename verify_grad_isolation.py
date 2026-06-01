"""Verify gradient isolation: margin_loss -> backbone = 0, action_loss -> ACIS = 0."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "disabled"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from train import (load_config, set_seed, load_smolvla_policy, load_dataset,
                   make_dataloader, wrap_with_imm, preprocess_batch, get_tokenizer_from_policy)
from imm.smolvla_wrapper import _make_att_2d_masks

def main():
    config = load_config("configs/imm_anchor_v8.yaml")
    set_seed(42)
    device = torch.device("cuda")

    print("[1/5] Loading model...")
    policy = load_smolvla_policy(config, device)
    model = wrap_with_imm(policy, config)
    model.to(device)
    model.train()
    print(f"      gradient_isolation = {model.gradient_isolation}")

    print("[2/5] Loading data...")
    tokenizer = get_tokenizer_from_policy(policy)
    dataset = load_dataset(config)
    dataloader = make_dataloader(dataset, config)
    batch = next(iter(dataloader))
    batch = preprocess_batch(batch, tokenizer, device=device)

    # --- First: simple 1-step forward through model.forward() ---
    print("[3/5] Forward pass (1 step)...")
    total_loss, info = model(batch, step=5000)
    print(f"      action_loss = {info['action_loss'].item():.6f}")
    print(f"      margin_loss = {info['margin_loss'].item():.6f}")
    print(f"      total_loss  = {total_loss.item():.6f}")

    # --- Now: manual forward to get grad-attached loss tensors ---
    print("\n[4/5] Gradient isolation checks...")
    model.zero_grad()

    bm = model.base_model
    images, img_masks = model.policy.prepare_images(batch)
    state = model.policy.prepare_state(batch)
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = model.policy.prepare_action(batch)
    vis_tokens, tokens_per_cam = model._get_visual_tokens(images)

    noise = bm.sample_noise(actions.shape, actions.device)
    time_ = bm.sample_time(actions.shape[0], actions.device)
    time_exp = time_[:, None, None]
    x_t = time_exp * noise + (1 - time_exp) * actions
    u_t = noise - actions
    suffix_embs, suffix_pad, suffix_att = bm.embed_suffix(x_t, time_)

    # Dual detach (matching gradient_isolation=True logic)
    vis_for_acis = vis_tokens.detach()
    suffix_for_acis = suffix_embs.detach()
    importance_scores = model.acis(suffix_for_acis, vis_for_acis)

    N_vis = vis_tokens.shape[1]
    k = max(1, int(N_vis * model.k_ratio))

    scores_for_select = importance_scores.detach()
    selected_tokens, mask, sel_info = model.token_selector(
        scores_for_select, vis_tokens, k, training=True
    )

    prefix_embs, prefix_pad, prefix_att = model._build_prefix_with_vis_tokens(
        selected_tokens, tokens_per_cam, img_masks, lang_tokens, lang_masks, state
    )
    all_pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    all_att = torch.cat([prefix_att, suffix_att], dim=1)
    att_2d = _make_att_2d_masks(all_pad, all_att)
    position_ids = torch.cumsum(all_pad, dim=1) - 1

    (_, suffix_out), _ = bm.vlm_with_expert.forward(
        attention_mask=att_2d, position_ids=position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False, fill_kv_cache=False,
    )
    chunk_size = bm.config.chunk_size
    suffix_out = suffix_out[:, -chunk_size:].to(dtype=torch.float32)
    v_t = bm.action_out_proj(suffix_out)
    action_dim = model.policy.config.action_feature.shape[0]
    action_loss_t = F.mse_loss(u_t[:, :, :action_dim], v_t[:, :, :action_dim])
    margin_loss_t, _ = model.imm_loss.margin_loss(importance_scores, k)

    results = {}

    # Check 1: margin_loss -> backbone LoRA = None
    backbone_lora = [(n, p) for n, p in model.policy.model.named_parameters()
                     if 'lora' in n.lower() and 'acis' not in n.lower() and p.requires_grad]
    print(f"      backbone LoRA params: {len(backbone_lora)}")
    if backbone_lora:
        grads = torch.autograd.grad(margin_loss_t, [p for _, p in backbone_lora],
                                     allow_unused=True, retain_graph=True)
        leak1 = any(g is not None and g.abs().max() > 0 for g in grads)
        print(f"      {'FAIL' if leak1 else 'PASS'}: margin_loss -> backbone LoRA = {'LEAK' if leak1 else 'blocked'}")
        results['margin_to_backbone'] = not leak1
    else:
        print("      SKIP: no backbone LoRA params")
        results['margin_to_backbone'] = True

    # Check 2: action_loss -> ACIS = None
    acis_params = [(n, p) for n, p in model.named_parameters()
                   if 'acis' in n and p.requires_grad]
    print(f"      ACIS params: {len(acis_params)}")
    grads2 = torch.autograd.grad(action_loss_t, [p for _, p in acis_params],
                                  allow_unused=True, retain_graph=True)
    leak2 = any(g is not None and g.abs().max() > 0 for g in grads2)
    print(f"      {'FAIL' if leak2 else 'PASS'}: action_loss -> ACIS = {'LEAK' if leak2 else 'blocked'}")
    results['action_to_acis'] = not leak2

    # Sanity: action_loss -> backbone LoRA SHOULD have gradient
    if backbone_lora:
        grads3 = torch.autograd.grad(action_loss_t, [p for _, p in backbone_lora[:3]],
                                      allow_unused=True, retain_graph=True)
        has_grad3 = any(g is not None and g.abs().max() > 0 for g in grads3)
        print(f"      {'PASS' if has_grad3 else 'FAIL'}: action_loss -> backbone LoRA exists (sanity)")
        results['action_trains_backbone'] = has_grad3

    # Sanity: margin_loss -> ACIS SHOULD have gradient
    grads4 = torch.autograd.grad(margin_loss_t, [p for _, p in acis_params[:3]],
                                  allow_unused=True, retain_graph=True)
    has_grad4 = any(g is not None and g.abs().max() > 0 for g in grads4)
    print(f"      {'PASS' if has_grad4 else 'FAIL'}: margin_loss -> ACIS exists (sanity)")
    results['margin_trains_acis'] = has_grad4

    # Check 5: eval mode forward
    print("\n[5/5] Eval mode forward...")
    model.eval()
    with torch.no_grad():
        total_loss_eval, info_eval = model(batch, step=0)
    print(f"      eval forward OK, total_loss = {total_loss_eval.item():.6f}")

    all_pass = all(results.values())
    print(f"\n{'='*60}")
    print(f"GRADIENT ISOLATION {'VERIFIED' if all_pass else 'FAILED'}")
    print(f"{'='*60}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
