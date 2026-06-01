"""Analysis 1: ACIS importance score distribution at v8 step-2K."""
import sys, os
sys.path.insert(0, '.')
os.chdir('.')

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
    dl = make_dataloader(dataset, dl_cfg)
    tokenizer = get_tokenizer_from_policy(model.policy)

    all_scores = []
    N_BATCHES = 100
    print(f'Collecting importance scores over {N_BATCHES} batches...')
    for i, batch in enumerate(dl):
        if i >= N_BATCHES:
            break
        batch = preprocess_batch(batch, tokenizer, device=device)
        with torch.no_grad():
            scores, k = model.compute_importance_scores(batch)
        all_scores.append(scores.cpu().numpy())
        if (i + 1) % 25 == 0:
            print(f'  batch {i+1}/{N_BATCHES}')

    all_scores = np.concatenate(all_scores, axis=0)
    flat = all_scores.flatten()
    N_vis = all_scores.shape[1]

    med = np.median(flat)
    mean = np.mean(flat)
    std = np.std(flat)
    print(f'N_vis={N_vis}, n_samples={all_scores.shape[0]}')
    print(f'Stats: mean={mean:.4f}, median={med:.4f}, std={std:.4f}')
    print(f'  min={flat.min():.4f}, max={flat.max():.4f}')

    # Check for bimodality via KDE
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    kde = gaussian_kde(flat)
    x_grid = np.linspace(flat.min() - 0.5 * std, flat.max() + 0.5 * std, 300)
    density = kde(x_grid)
    peaks_idx, props = find_peaks(density, prominence=0.005)
    print(f'Detected {len(peaks_idx)} peak(s) in KDE')
    for pi, p in enumerate(peaks_idx):
        print(f'  Peak {pi+1}: score={x_grid[p]:.4f}, density={density[p]:.4f}')

    # Per-sample score spread
    per_sample_range = all_scores.max(axis=1) - all_scores.min(axis=1)
    print(f'Per-sample score range: mean={per_sample_range.mean():.4f}, std={per_sample_range.std():.4f}')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=50, alpha=0.7, density=True, label='IMM v8 (step 2K)',
            color='#1976D2', edgecolor='white', linewidth=0.5)
    rand = np.random.uniform(flat.min(), flat.max(), size=len(flat))
    ax.hist(rand, bins=50, alpha=0.25, density=True, label='Random (uniform)',
            color='#757575', edgecolor='white', linewidth=0.5)

    ax.axvline(med, color='#E53935', ls='--', lw=1.5, label=f'Median = {med:.3f}')
    ax.axvline(mean, color='#43A047', ls='--', lw=1.5, label=f'Mean = {mean:.3f}')

    if len(peaks_idx) >= 2:
        ax.annotate('Bimodal / multi-peak structure detected',
                     xy=(0.5, 0.95), xycoords='axes fraction',
                     fontsize=11, ha='center', style='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8))

    ax.set_xlabel('Importance Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('ACIS Importance Score Distribution (v8, Step 2K)', fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = 'artifacts/v8_step2k_diagnostics/margin_distribution.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()

    # Save raw stats
    import json
    stats = {
        'n_vis': int(N_vis), 'n_samples': int(all_scores.shape[0]),
        'mean': float(mean), 'median': float(med), 'std': float(std),
        'min': float(flat.min()), 'max': float(flat.max()),
        'n_peaks': len(peaks_idx),
        'peaks': [{'score': float(x_grid[p]), 'density': float(density[p])} for p in peaks_idx],
        'per_sample_range_mean': float(per_sample_range.mean()),
    }
    with open('artifacts/v8_step2k_diagnostics/margin_distribution_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print('Stats saved.')

if __name__ == '__main__':
    main()
