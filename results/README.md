# Evaluation Results

All per-line evaluation outputs are in `results/eval/`.

## File Naming Convention
`eval_{model}_{quantization}_{subset}_{benchmark}_{results|metrics|report}.{csv|json|jsonl|txt}`

## Notes
- **9B Low HP (Subset 1):** The files `eval_qwen35_9b_lowhp_gptq_w4a16_subset1_*` contain the re-evaluated results using the published GPTQ W4A16 pipeline with corrected inference image bounds (100,352–602,112 px). These supersede earlier evaluations that used the model's training bounds.
- **Table 8 (Throughput):** Throughput values come from the Subset 3 model evaluations.
- **Figure 6b:** CER and throughput values come from Subset 2 4B checkpoints (High HP, three ablations, Low HP reference).
- **LR ablation:** `eval_qwen35_4b_latin_gptq_w4a16_v2_ablation_lr3e5_*` contains results for the lower-learning-rate (3e-5) ablation run.
