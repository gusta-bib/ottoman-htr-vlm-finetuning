# Training Scripts & Manuscript Run Mapping

| Script Name | Model | Configuration | Dataset Subset | Manuscript Reference |
|---|---|---|---|---|
| `finetune_qwen35_2b_latin_v1_15042.py` | Qwen3.5-2B | High HP | Subset 1 (15.0k) | Table 5 |
| `finetune_qwen35_2b_latin_v2_15042_7740.py` | Qwen3.5-2B | High HP | Subset 2 (22.8k) | Table 5 |
| `finetune_qwen35_2b_latin_v3_15042_7740_4678.py` | Qwen3.5-2B | High HP | Subset 3 (27.5k) | Table 5, Table 8 |
| `finetune_qwen35_4b_latin_v1_15042.py` | Qwen3.5-4B | High HP | Subset 1 (15.0k) | Table 5 |
| `finetune_qwen35_4b_latin_v2_15042_7740.py` | Qwen3.5-4B | High HP | Subset 2 (22.8k) | Table 5, Table 6 |
| `finetune_qwen35_4b_latin_v3_15042_7740_4678.py` | Qwen3.5-4B | High HP | Subset 3 (27.5k) | Table 5, Table 8 |
| `finetune_qwen35_9b_latin_v1_15042.py` | Qwen3.5-9B | High HP | Subset 1 (15.0k) | Table 5 |
| `finetune_qwen35_9b_latin_v2.py` | Qwen3.5-9B | High HP | Subset 2 (22.8k) | Table 5 |
| `finetune_qwen35_9b_latin_v3.py` | Qwen3.5-9B | High HP | Subset 3 (27.5k) | Table 5, Table 8 |
| `finetune_qwen35_2b_latin_v2_15042_7740_test.py` | Qwen3.5-2B | Low HP | Subset 2 (22.8k) | Section 4.3, Figure 3 |
| `finetune_qwen35_4b_latin_v2_15042_7740_test.py` | Qwen3.5-4B | Low HP | Subset 2 (22.8k) | Section 4.3, Figure 3, Table 6 |
| `finetune_qwen35_9b_lowhp_subset1.py` | Qwen3.5-9B | Low HP | Subset 1 (15.0k) | Section 4.3, Figure 3 |
| `ablation/finetune_qwen35_4b_latin_v2_15042_7740_ablasion_lowres_lora64.py` | Qwen3.5-4B | Lower image limit (200k px) | Subset 2 (22.8k) | Table 6 |
| `ablation/finetune_qwen35_4b_latin_v2_15042_7740_ablasion_highres_lora32.py` | Qwen3.5-4B | Lower rank (r=32) | Subset 2 (22.8k) | Table 6 |
| `ablation/finetune_qwen35_4b_latin_v2_15042_7740_ablasion_lr3e5.py` | Qwen3.5-4B | Lower LR (3e-5) | Subset 2 (22.8k) | Table 6 |
| `finetune_qwen35_4b_latin_v1_iso_compute_4272.py` | Qwen3.5-4B | Iso-compute | Subset 1 match | Section 4.4 |

See `docs/TRAINING_SETTINGS.md` for hyperparameter details for each configuration.
