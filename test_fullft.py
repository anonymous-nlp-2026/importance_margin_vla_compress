import torch, sys
sys.path.insert(0, '.')
from train import load_config, load_smolvla_policy

config = load_config('configs/libero_full_ft.yaml')
device = torch.device('cuda:0')
print('Loading SmolVLA with full expert FT config...')
policy = load_smolvla_policy(config, device)
print('Model loaded successfully')

trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
total = sum(p.numel() for p in policy.parameters())
print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

trained_names = [n for n, p in policy.named_parameters() if p.requires_grad]
frozen_names = [n for n, p in policy.named_parameters() if not p.requires_grad]
print(f'Trainable modules: {len(trained_names)}')
print(f'Frozen modules: {len(frozen_names)}')
for n in trained_names[:10]:
    print(f'  TRAIN: {n}')
if len(trained_names) > 10:
    print(f'  ... and {len(trained_names)-10} more')
for n in frozen_names[:5]:
    print(f'  FROZEN: {n}')
if len(frozen_names) > 5:
    print(f'  ... and {len(frozen_names)-5} more')
print('VALIDATION_DONE')
