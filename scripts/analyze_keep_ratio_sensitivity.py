"""Analysis 2: Keep ratio sensitivity scan — ACIS vs random token selection."""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/importance_margin_vla_compress')
os.chdir('/root/autodl-tmp/importance_margin_vla_compress')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda:0')
    from evaluate import load_config, load_model_and_checkpoint
    from train import load_dataset, make_dataloader, get_tokenizer_from_policy, preprocess_batch

    print('Loading v8 step-2K checkpoint...')
    config = load_config('configs/imm_anchor_v8.yaml')
    model, _ = load_model_and_checkpoint(config, 'checkpoints/imm_anchor_v8/step_002000', device)
    model.eval()

    dataset = load_dataset(config)
    dl_cfg = {**config, 'training': {**config['training'], 'batch_size': 4}}
    tokenizer = get_tokenizer_from_policy(model.policy)

    # Pre-cache 100 batches for consistency
    print('Pre-caching 100 batches...')
    torch.manual_seed(42)
    dl = make_dataloader(dataset, dl_cfg)
    cached = []
    for i, batch in enumerate(dl):
        if i >= 100:
            break
        batch = preprocess_batch(batch, tokenizer, device=device)
        cached.append({k: v.cpu() for k, v in batch.items()})
    print(f'Cached {len(cached)} batches.')

    keep_ratios = [0.3, 0.5, 0.7, 0.9]
    results = {}

    # Sweep with ACIS scores
    for kr in keep_ratios:
        model.k_ratio = kr
        losses = []
        for i, batch_cpu in enumerate(cached):
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            with torch.no_grad():
                _, info = model(batch, step=999999)
            losses.append(info['action_loss'].item())
        avg = np.mean(losses)
        results[('acis', kr)] = avg
        print(f'  ACIS  keep_ratio={kr:.1f}: action_loss={avg:.6f}')

    # Sweep with random scores (monkey-patch ACIS)
    original_acis_forward = model.acis.forward
    def random_acis(a_query, visual_tokens):
        B, N = visual_tokens.shape[:2]
        return torch.randn(B, N, device=visual_tokens.device, dtype=visual_tokens.dtype)
    model.acis.forward = random_acis

    for kr in keep_ratios:
        model.k_ratio = kr
        losses = []
        for i, batch_cpu in enumerate(cached):
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            with torch.no_grad():
                _, info = model(batch, step=999999)
            losses.append(info['action_loss'].item())
        avg = np.mean(losses)
        results[('random', kr)] = avg
        print(f'  Random keep_ratio={kr:.1f}: action_loss={avg:.6f}')

    model.acis.forward = original_acis_forward

    # Also compute full (no pruning) baseline
    model.k_ratio = 0.99
    losses_full = []
    for i, batch_cpu in enumerate(cached):
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        with torch.no_grad():
            _, info = model(batch, step=999999)
        losses_full.append(info['action_loss'].item())
    full_loss = np.mean(losses_full)
    print(f'  Full (no pruning, k_ratio=0.99): action_loss={full_loss:.6f}')

    # Plot
    acis_losses = [results[('acis', kr)] for kr in keep_ratios]
    rand_losses = [results[('random', kr)] for kr in keep_ratios]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(keep_ratios, acis_losses, 'o-', color='#1976D2', lw=2, ms=8,
            label='IMM v8 (ACIS)', zorder=3)
    ax.plot(keep_ratios, rand_losses, 's--', color='#757575', lw=2, ms=8,
            label='Random selection', zorder=3)
    ax.axhline(full_loss, color='#43A047', ls=':', lw=1.5, label=f'No pruning = {full_loss:.4f}')

    for kr, al, rl in zip(keep_ratios, acis_losses, rand_losses):
        ax.annotate(f'{al:.4f}', (kr, al), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=9, color='#1976D2')
        ax.annotate(f'{rl:.4f}', (kr, rl), textcoords='offset points',
                    xytext=(0, -16), ha='center', fontsize=9, color='#555555')

    ax.set_xlabel('Keep Ratio', fontsize=13)
    ax.set_ylabel('Action Loss (MSE)', fontsize=13)
    ax.set_title('Keep Ratio Sensitivity: ACIS vs Random (v8, Step 2K)', fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=11)
    ax.set_xticks(keep_ratios)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = 'artifacts/v8_step2k_diagnostics/keep_ratio_sensitivity.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()

    # Save raw data
    import json
    data = {
        'keep_ratios': keep_ratios,
        'acis_losses': acis_losses,
        'random_losses': rand_losses,
        'full_loss': full_loss,
        'n_batches': len(cached),
    }
    with open('artifacts/v8_step2k_diagnostics/keep_ratio_sensitivity.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('Data saved.')

if __name__ == '__main__':
    main()
