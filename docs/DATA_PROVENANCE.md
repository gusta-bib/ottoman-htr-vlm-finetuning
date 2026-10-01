# Data Provenance

## Training Data Construction (Section 3.2)

### Block A: Aligned Ground-Truth Lines (15,042 lines)
- Source: 300 DPI page scans of Ottoman court registers (*şer'iyye sicilleri*)
- Line detection: AHDoc model
- Reference text: Graduate-thesis Latin transliterations (TTK publications)
- Alignment: Monotonic sliding-window matching via phonetic consonant skeletons (RapidFuzz)
- Quality filter: alignment score ≥ 70 → 16,829 lines; after deduplication and leakage removal → 15,042 lines

### Block B: Pseudo-Labeled Lines (7,740 lines)
- Generator: 9B Low HP model (Qwen3.5-9B, LoRA r=32, alpha=64, LR 3e-5, max 200,704 px), trained on Subset 1 (Block A)
- Source: 18,215 aligned line images from Block A inference
- Selection: 7,740 lines with highest alignment confidence
- Used in: Subset 2 = Block A + Block B = 22,782 lines (22,241 training + 1,222 validation after 541-line leakage removal)

### Block C: Pseudo-Labeled Lines (4,678 lines)
- Generator: 9B High HP model (Qwen3.5-9B, LoRA r=64, alpha=128, LR 1e-4, max 602,112 px), trained on Subset 2
- Used in: Subset 3 = Subset 2 + Block C = 27,460 lines (training + validation)

## Evaluation Benchmarks (Section 3.4)

### In-Distribution (ID): 815 lines
- Held-out lines from the same court registers
- See `evaluation/benchmarks/id_815_manifest.csv`

### Out-of-Distribution (OOD): 1,575 lines
- Source: "Ottoman Handwritten (Riqa) OCR-Transliteration Dataset", Osmanlica.com
- URL: https://www.osmanlica.com/test (version of 7 April 2021)
- See `evaluation/benchmarks/ood_1575_manifest.csv`
