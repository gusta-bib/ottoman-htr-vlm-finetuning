# Training Settings

All models use QLoRA (NF4 base, bf16 compute) with FlashAttention-2 on a single NVIDIA RTX 4070 Ti SUPER (16 GB).

## High HP Configuration (Sections 3.3, 4.2–4.4)
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| Learning rate | 1e-4 |
| min_pixels | 100,352 (128 × 28 × 28) |
| max_pixels | 602,112 (768 × 28 × 28) |
| Epochs | 3 |
| Effective batch size | 16 (batch 1 × grad_accum 16) |
| save_steps / eval_steps | 250 |
| metric_for_best_model | eval_loss |
| load_best_model_at_end | True |
| Early stopping patience | 8 |

## Low HP Configuration (Section 4.3)
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Learning rate | 3e-5 |
| min_pixels | 3,136 (4 × 28 × 28) |
| max_pixels | 200,704 (256 × 28 × 28) |
| save_steps | 500 (9B) or 250 (2B, 4B) |
| Other settings | Same as High HP |

## Per-Run Details

| Run | Script | r | alpha | LR | min_px | max_px | save/eval_steps | Checkpoint Selection |
|-----|--------|---|-------|-----|--------|--------|-----------------|---------------------|
| 2B Subset 1 | `finetune_qwen35_2b_latin_v1_15042.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 2B Subset 2 | `finetune_qwen35_2b_latin_v2_15042_7740.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 2B Subset 3 | `finetune_qwen35_2b_latin_v3_15042_7740_4678.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 4B Subset 1 | `finetune_qwen35_4b_latin_v1_15042.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 4B Subset 2 | `finetune_qwen35_4b_latin_v2_15042_7740.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 4B Subset 3 | `finetune_qwen35_4b_latin_v3_15042_7740_4678.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 9B Subset 1 | `finetune_qwen35_9b_latin_v1_15042.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 9B Subset 2 | `finetune_qwen35_9b_latin_v2.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 9B Subset 3 | `finetune_qwen35_9b_latin_v3.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 2B Low HP | `finetune_qwen35_2b_latin_v2_15042_7740_test.py` | 32 | 64 | 3e-5 | 3,136 | 200,704 | 250 | Lowest eval_loss |
| 4B Low HP | `finetune_qwen35_4b_latin_v2_15042_7740_test.py` | 32 | 64 | 3e-5 | 3,136 | 200,704 | 250 | Lowest eval_loss |
| 9B Low HP | `finetune_qwen35_9b_latin_v1.py` | 32 | 64 | 3e-5 | 3,136 | 200,704 | 500 | Lowest eval_loss |
| 4B Ablation: Lower image limit | `finetune_..._ablasion_lowres_lora64.py` | 64 | 128 | 1e-4 | 3,136 | 200,704 | 250 | **Final checkpoint** |
| 4B Ablation: Lower rank | `finetune_..._ablasion_highres_lora32.py` | 32 | 64 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 4B Ablation: Lower LR | `finetune_..._ablasion_lr3e5.py` | 64 | 128 | 3e-5 | 100,352 | 602,112 | 250 | Lowest eval_loss |
| 4B Iso-compute | `finetune_qwen35_4b_latin_v1_iso_compute_4272.py` | 64 | 128 | 1e-4 | 100,352 | 602,112 | 250 | Lowest eval_loss |

**Note:** The lower-image-limit ablation used its final checkpoint (not the best-eval-loss checkpoint) because its validation loss did not decrease monotonically under the reduced resolution.
