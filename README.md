# Supplementary Code

**Why Token Pruning Crashes Some Robot Policies but Not Others: Vision Connectors and VLA Token Compressibility**

Submitted to CoRL 2026.

## Directory Structure

```
.
├── imm/                    # Importance margin module (exploratory; not used in main results)
│   ├── __init__.py
│   ├── acis.py             # Attention-Conditioned Importance Scoring
│   ├── losses.py           # Training losses (importance margin loss, distillation)
│   ├── metrics.py          # Evaluation metrics
│   ├── smolvla_wrapper.py  # SmolVLA model wrapper with token compression
│   ├── soft_topk.py        # Differentiable top-k token selection
│   └── token_selection.py  # Token selection strategies (L2-norm, attention, random)
├── configs/                # Training & evaluation YAML configs
├── scripts/                # Analysis and visualization utilities
├── results/                # SmolVLA MetaWorld evaluation results (JSON)
├── eval_results/           # LIBERO evaluation results across all VLAs (JSON)
│
├── train.py                           # SmolVLA training (LoRA fine-tuning)
├── train_smart_compress.py            # SmolVLA training with learned token compression
├── train_twostage.py                  # Two-stage training (compress then recover)
├── train_baseline_control.py          # Baseline control training
├── train_openvla_smart_compress.py    # OpenVLA training with learned compression
├── train_openvla_smart_compress_lora.py # OpenVLA LoRA variant
│
├── evaluate.py                        # SmolVLA MetaWorld closed-loop evaluation
├── eval_official_libero.py            # SmolVLA LIBERO evaluation (official protocol)
├── eval_oft_libero.py                 # SmolVLA OFT LIBERO evaluation
├── eval_openvla_libero.py             # OpenVLA LIBERO evaluation
├── eval_openvla_l2norm_libero.py      # OpenVLA L2-norm pruning evaluation
├── eval_openvla_zeromask_libero.py    # OpenVLA zero-mask pruning evaluation
├── eval_openvla_smart_compress.py     # OpenVLA learned compression evaluation
├── eval_openvla_acis_prune.py         # OpenVLA ACIS pruning evaluation
├── eval_pi05_libero.py                # π0.5 LIBERO evaluation
├── eval_pi05_pruning_libero.py        # π0.5 pruning evaluation
├── eval_smolvla_pruning_metaworld.py  # SmolVLA pruning on MetaWorld
├── eval_xvla_pruning_libero.py        # Cross-VLA L2-norm pruning on LIBERO
├── eval_xvla_attention_pruning_libero.py # Cross-VLA attention pruning on LIBERO
│
├── smart_compress_module.py           # Learned compression module (gating network)
├── openvla_vtc_adapter.py             # OpenVLA vision token compression adapter
├── geometric_prior_pruning.py         # Geometric prior-based pruning
├── _lerobot_compat.py                 # LeRobot compatibility layer
│
├── analyze_gate.py                    # Gate activation analysis
├── analyze_openvla_oft.py             # OpenVLA OFT analysis
├── analyze_openvla_oft_token_importance.py  # Token importance in OFT models
├── analyze_token_cosine_similarity.py # Token cosine similarity analysis
├── pi0_attention_entropy.py           # π0 attention entropy measurement
├── pi0_token_importance.py            # π0 token importance measurement
├── visualize_importance.py            # Importance score visualization
├── measure_inference_latency.py       # Inference latency benchmarking
└── measure_smolvla_entropy_perlayer*.py # Per-layer entropy measurement
```

## Environment

- Python 3.10+
- PyTorch 2.1+ with CUDA 12.x
- Key dependencies: `transformers`, `lerobot`, `libero`, `mujoco`, `einops`, `accelerate`, `wandb`, `peft`
- For π0.5 evaluation: `physical-intelligence/openpi`

## Reproducing Experiments

### 1. Token Pruning Evaluation (Table 1–2, Figure 3)

Run closed-loop pruning evaluation on LIBERO for each VLA:

```bash
# SmolVLA on LIBERO (L2-norm pruning, keep ratio k)
python eval_official_libero.py --keep_ratio 0.5 --suite libero_object

# OpenVLA on LIBERO
python eval_openvla_l2norm_libero.py --keep_ratio 0.5 --suite libero_object

# π0.5 on LIBERO
python eval_pi05_pruning_libero.py --prune_mode l2norm --keep_ratio 0.5 --suite libero_10
```

### 2. Learned Compression (Table 3)

Train and evaluate the learned compression module:

```bash
# Train SmolVLA with Smart Compress
python train_smart_compress.py --config configs/smart_compress_m64.yaml

# Evaluate
python eval_smart_compress_libero.py --checkpoint <path>
```

### 3. Analysis Scripts

```bash
# Token importance visualization
python visualize_importance.py

# Attention entropy analysis
python pi0_attention_entropy.py

# Inference latency measurement
python measure_inference_latency.py
```

## Notes

- Pre-trained model weights and datasets are **not** included in this supplementary material.
- SmolVLA weights: `lerobot/smolvla_base` (HuggingFace)
- OpenVLA weights: `openvla/openvla-7b-finetuned-libero-object` (HuggingFace)
- π0.5 weights: via `openpi` package
- LIBERO benchmark: `https://github.com/Lifelong-Robot-Learning/LIBERO`
- MetaWorld benchmark: `https://github.com/Farama-Foundation/Metaworld`
