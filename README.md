# Fine-Tuning Vision-Language Models for Ottoman Handwritten Text Recognition

> Code, per-line evaluation outputs, and analysis scripts for an anonymized manuscript under review at *Information Processing & Management*.

## Repository Map

```
.
├── data_preparation/        # Data alignment and phonetic skeleton scripts (Section 3.2)
├── training/                # Fine-tuning scripts and logs for all runs (Section 3.3–3.5)
│   └── ablation/            # Single-factor ablation scripts (Section 4.5)
├── evaluation/              # Evaluation scripts: vLLM GPTQ W4A16 pipeline (Section 3.6)
│   ├── benchmarks/          # Benchmark manifests and access instructions (Section 3.4)
│   ├── stats/               # Bootstrap CI and paired-test scripts (Section 3.8)
│   └── visual_attention_mass/  # Attention analysis (Section 4.6)
├── results/eval/            # Per-line CSV/JSONL outputs and metrics JSON for every run
├── figures/                 # Figure generation scripts and outputs (Figures 1–6)
├── docs/                    # Training settings, data provenance, limitations
├── archive/                 # Unused scripts (not cited in manuscript)
└── segmentation_quality/    # Line-detection quality analysis
```

## Reproducing Results

### Environment
```bash
conda env create -f environment.yml
conda activate ottoman-htr
```

### Tables 5 and 6 (Main CER Results and Ablations)
Per-line outputs are in `results/eval/`. To recompute micro CER and bootstrap CIs:
```bash
python evaluation/stats/compute_main_results_ci.py
python evaluation/stats/run_micro_bootstrap.py
```

### Table 8 (Throughput)
Throughput values are recorded in the `*_metrics.json` files in `results/eval/`.

### Figures 1–4
```bash
python figures/generate_figures_1_to_4.py
```

### Figure 5 (Attention Mass)
```bash
python figures/generate_figure5_attention.py
```

### Figure 6 (Throughput vs CER)
```bash
python figures/generate_figure6_throughput.py
```

## Data Access
See `evaluation/benchmarks/README.md` for benchmark dataset access and verification instructions.

## Citation
See `CITATION.cff`. Author details will be added after review.

## License
MIT License. See `LICENSE`.
