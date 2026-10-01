# Iso-Compute Control and Ablation Studies

This directory contains evaluation outputs and runner scripts for the iso-compute control experiment (Table 6).

## Files in this Directory

- `run_qwen35_4b_v1_iso_compute.sh`: Execution script for training the Qwen3.5-4B model on Block A alone (15,042 lines) for 4,272 optimizer steps (about 4.5 epochs, slightly more than the Subset 2 run of 4,173 steps, compared to 2,823 steps for Subset 1).
- `eval_qwen35_4b_latin_gptq_w4a16_v1_iso_compute_4272_815_rika_metrics.json`: Summary evaluation metrics on the ID benchmark (815 lines).
- `eval_qwen35_4b_latin_gptq_w4a16_v1_iso_compute_4272_815_rika_report.txt`: Evaluation report on the ID benchmark.
- `eval_qwen35_4b_latin_gptq_w4a16_v1_iso_compute_4272_815_rika_results.csv`: Line-by-line prediction results on the ID benchmark.
- `eval_qwen35_4b_latin_gptq_w4a16_v1_iso_compute_4272_osmanlicacom_1575_metrics.json`: Summary evaluation metrics on the OOD benchmark (1,575 lines).
- `eval_qwen35_4b_latin_gptq_w4a16_v1_iso_compute_4272_osmanlicacom_1575_report.txt`: Evaluation report on the OOD benchmark.
- `eval_qwen35_4b_latin_gptq_w4a16_v1_iso_compute_4272_osmanlicacom_1575_results.csv`: Line-by-line prediction results on the OOD benchmark.
