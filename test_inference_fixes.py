"""test_inference_fixes.py - Inference-time fixes for flow matching ODE mode collapse."""

import os, sys, logging
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from test_mode_collapse import load_config, load_model, load_dataset_batches, preprocess_batch
from train import get_tokenizer_from_policy


def get_inner(policy):
    m = policy.model
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            return m.base_model.model
    except ImportError:
        pass
    return m


def verdict(s):
    if s > 0.05: return "PASS"
    if s < 0.01: return "FAIL"
    return "WARN"


@torch.no_grad()
def sample_diversity(policy, single_batch, num_samples=10,
                     num_steps=None, sde_sigma=0.0, temperature=1.0, t_stop=0.0):
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    inner = get_inner(policy)

    images, img_masks = policy.prepare_images(single_batch)
    state = policy.prepare_state(single_batch)
    lang_tokens = single_batch["observation.language.tokens"]
    lang_masks = single_batch["observation.language.attention_mask"]

    N = num_samples
    def rep(t, n):
        if isinstance(t, torch.Tensor):
            return t.expand(n, *(-1,)*(t.dim()-1)).contiguous()
        if isinstance(t, list):
            return [x.expand(n, *(-1,)*(x.dim()-1)).contiguous() for x in t]
        return t

    imgs = rep(images, N)
    imasks = rep(img_masks, N)
    st = state.expand(N, -1).contiguous()
    lt = lang_tokens.expand(N, -1).contiguous()
    lm = lang_masks.expand(N, -1).contiguous()

    device = st.device
    actions_shape = (N, inner.config.chunk_size, inner.config.max_action_dim)
    noise = inner.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = inner.embed_prefix(
        imgs, imasks, lt, lm, state=st
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, kv_cache = inner.vlm_with_expert.forward(
        attention_mask=prefix_att_2d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=inner.config.use_cache,
        fill_kv_cache=True,
    )

    ns = num_steps if num_steps is not None else inner.config.num_steps
    dt = -1.0 / ns
    x_t = noise

    for step in range(ns):
        t = 1.0 + step * dt
        if t < t_stop:
            break
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(N)
        v_t = inner.denoise_step(
            x_t=x_t, prefix_pad_masks=prefix_pad_masks,
            past_key_values=kv_cache, timestep=t_tensor,
        )
        x_t = x_t + dt * v_t * temperature
        if sde_sigma > 0:
            x_t = x_t + torch.randn_like(x_t) * sde_sigma * (abs(dt)**0.5)

    adim = policy.config.action_feature.shape[0]
    actions = x_t[:, :, :adim]
    return actions.std(dim=0).mean().item()


def main():
    device = torch.device("cuda:0")
    ckpt = "checkpoints/baseline/step_029999/checkpoint.pt"
    cfg_path = "configs/baseline.yaml"
    NS = 10

    log.info("Loading model...")
    policy, config = load_model(ckpt, cfg_path, device)
    tokenizer = get_tokenizer_from_policy(policy)
    inner = get_inner(policy)
    default_steps = inner.config.num_steps

    log.info("Loading dataset...")
    batches = load_dataset_batches(config, 3, 4)
    batch = preprocess_batch(dict(batches[0]), tokenizer, device)
    single = {k: v[:1] for k, v in batch.items() if isinstance(v, torch.Tensor)}

    print(f"\n=== cuda:0 Inference Fixes (baseline_v5 final) ===")
    print(f"Default solver: Euler, steps={default_steps}")

    results = []

    # 1. ODE steps sweep
    log.info("--- Test 1: ODE Steps Sweep ---")
    for steps in [3, 5, 10, 20, 50, 100]:
        s = sample_diversity(policy, single, NS, num_steps=steps)
        r = f"Steps={steps}: std={s:.4f} [{verdict(s)}]"
        results.append(r); print(r); sys.stdout.flush()

    # 2. SDE
    log.info("--- Test 2: SDE Sampling ---")
    for sigma in [0.01, 0.05, 0.1]:
        s = sample_diversity(policy, single, NS, sde_sigma=sigma)
        r = f"SDE sigma={sigma}: std={s:.4f} [{verdict(s)}]"
        results.append(r); print(r); sys.stdout.flush()

    # 3. Temperature
    log.info("--- Test 3: Temperature Scaling ---")
    for temp in [0.5, 0.8, 1.2]:
        s = sample_diversity(policy, single, NS, temperature=temp)
        r = f"Temp={temp}: std={s:.4f} [{verdict(s)}]"
        results.append(r); print(r); sys.stdout.flush()

    # 4. Early stop
    log.info("--- Test 4: Early Stop ---")
    for ts in [0.01, 0.05, 0.1]:
        s = sample_diversity(policy, single, NS, t_stop=ts)
        r = f"Early stop t={ts}: std={s:.4f} [{verdict(s)}]"
        results.append(r); print(r); sys.stdout.flush()

    # 5. LoRA merge (destructive - last)
    log.info("--- Test 5: LoRA Merge ---")
    try:
        from peft import PeftModel
        if isinstance(policy.model, PeftModel):
            policy.model = policy.model.merge_and_unload()
            s = sample_diversity(policy, single, NS)
            r = f"LoRA merge: std={s:.4f} [{verdict(s)}]"
        else:
            r = "LoRA merge: N/A"
    except Exception as e:
        r = f"LoRA merge: ERROR ({e})"
    results.append(r); print(r); sys.stdout.flush()

    print(f"\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
