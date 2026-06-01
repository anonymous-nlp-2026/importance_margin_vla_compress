"""Analysis 3: Token-level importance heatmap visualization."""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/importance_margin_vla_compress')
os.chdir('/root/autodl-tmp/importance_margin_vla_compress')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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

    # Get first batch
    batch = next(iter(dl))
    batch = preprocess_batch(batch, tokenizer, device=device)

    # Get importance scores
    with torch.no_grad():
        scores, k = model.compute_importance_scores(batch)
    scores_np = scores.cpu().numpy()  # (B, N_vis)
    N_vis = scores_np.shape[1]
    print(f'N_vis = {N_vis}, k = {k}')

    # Get original images for overlay
    img_key = 'observation.images.top'
    if img_key in batch:
        images = batch[img_key].cpu().numpy()  # (B, C, H, W) or (B, T, C, H, W)
        if images.ndim == 5:
            images = images[:, -1]  # last timestep
        # De-normalize: assume [0,1] or [-1,1] range
        if images.min() < 0:
            images = (images + 1) / 2
        images = np.clip(images, 0, 1)
        images = images.transpose(0, 2, 3, 1)  # (B, H, W, C)
        has_images = True
        print(f'Image shape: {images.shape}')
    else:
        has_images = False
        print('No images found in batch, will plot heatmap only.')

    # Determine spatial grid
    grid_h = int(np.sqrt(N_vis))
    grid_w = N_vis // grid_h
    if grid_h * grid_w != N_vis:
        # Not a perfect square — try common factorizations
        for h in range(int(np.sqrt(N_vis)), 0, -1):
            if N_vis % h == 0:
                grid_h, grid_w = h, N_vis // h
                break
    print(f'Using spatial grid: {grid_h} x {grid_w} = {grid_h * grid_w}')

    n_samples = min(3, scores_np.shape[0])
    fig, axes = plt.subplots(2 if has_images else 1, n_samples,
                              figsize=(4 * n_samples, 8 if has_images else 4))
    if n_samples == 1:
        axes = np.array(axes).reshape(-1, 1) if has_images else np.array([axes]).reshape(1, 1)
    elif not has_images:
        axes = axes.reshape(1, -1)

    vmin = scores_np[:n_samples].min()
    vmax = scores_np[:n_samples].max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    for si in range(n_samples):
        s = scores_np[si]
        hmap = s.reshape(grid_h, grid_w)

        if has_images:
            # Top row: original image with heatmap overlay
            ax_img = axes[0, si]
            ax_img.imshow(images[si])
            # Resize heatmap to image size for overlay
            from PIL import Image
            hmap_resized = np.array(Image.fromarray(hmap).resize(
                (images.shape[2], images.shape[1]), Image.BILINEAR))
            ax_img.imshow(hmap_resized, cmap='viridis', alpha=0.5, norm=norm)
            ax_img.set_title(f'Sample {si+1} (overlay)', fontsize=12)
            ax_img.axis('off')

            # Bottom row: raw heatmap
            ax_hm = axes[1, si]
        else:
            ax_hm = axes[0, si]

        im = ax_hm.imshow(hmap, cmap='viridis', norm=norm, interpolation='nearest')
        ax_hm.set_title(f'Sample {si+1} ({grid_h}x{grid_w})', fontsize=12)
        ax_hm.set_xlabel('Token col', fontsize=10)
        ax_hm.set_ylabel('Token row', fontsize=10)
        ax_hm.tick_params(labelsize=9)

        # Annotate each cell with score
        if N_vis <= 100:
            for r in range(grid_h):
                for c in range(grid_w):
                    val = hmap[r, c]
                    color = 'white' if val < (vmin + vmax) / 2 else 'black'
                    ax_hm.text(c, r, f'{val:.2f}', ha='center', va='center',
                              fontsize=6, color=color)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Importance Score')

    plt.suptitle('Token Importance Heatmap (v8, Step 2K)', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    out = 'artifacts/v8_step2k_diagnostics/token_importance_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()

    # Print summary stats per sample
    for si in range(n_samples):
        s = scores_np[si]
        topk_idx = np.argsort(s)[-k:]
        print(f'Sample {si+1}: min={s.min():.4f}, max={s.max():.4f}, '
              f'range={s.max()-s.min():.4f}, top-k indices={sorted(topk_idx.tolist())}')

if __name__ == '__main__':
    main()
