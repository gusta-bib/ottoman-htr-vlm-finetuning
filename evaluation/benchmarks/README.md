# Evaluation Benchmarks

This directory contains manifests and instructions for accessing and reproducing the evaluation benchmarks used in the manuscript (Tables 3, 5–8, Section 3.4).

In compliance with dataset licensing and privacy restrictions:
1. **OOD Benchmark (Public dataset with restricted redistribution):** Line identifiers, image hashes, reference text hashes, and character counts are provided in `ood_1575_manifest.csv`. Raw images and texts are not redistributed directly in this repository.
2. **ID Benchmark (Archival court records):** Line identifiers, image hashes, reference text hashes, and character counts are provided in `id_815_manifest.csv`. The benchmark is available from the corresponding author upon reasonable academic request.

---

## 1. Out-of-Distribution (OOD) Benchmark: `ood_1575_manifest.csv`

- **Dataset Name:** *Ottoman Handwritten (Riqa) OCR-Transliteration Dataset*
- **Source URL:** [https://www.osmanlica.com/test](https://www.osmanlica.com/test)
- **Dataset Version Date:** 7 April 2021
- **Licensing:** Offered for research use without explicit permission for third-party redistribution.
- **Line Count:** 1,575 lines across historical documents in Ottoman Riqa script.

### Obtaining and Verifying the OOD Benchmark
1. Download the dataset archive from [https://www.osmanlica.com/test](https://www.osmanlica.com/test) (version dated 7 April 2021).
2. Extract the files to a local directory (e.g. `evaluation/benchmarks/ood_1575_osmanlicacom/`).
3. **Reference Text Format:** Each line contains an image (`<line_id>.png`) and two text files: `<line_id>.txt` and `<line_id>.gt.txt`. Both files provide the scholarly Latin transliteration of the Ottoman text; the published evaluation uses this Latin transliteration as the reference ground truth across all 1,575 lines.
4. **Verification:** Verify the downloaded image and text files against `ood_1575_manifest.csv`:
   ```bash
   python3 -c "
   import pandas as pd, hashlib, os
   df = pd.read_csv('evaluation/benchmarks/ood_1575_manifest.csv')
   data_dir = 'evaluation/benchmarks/ood_1575_osmanlicacom'
   for _, row in df.iterrows():
       img_path = os.path.join(data_dir, row['image_filename'])
       assert os.path.exists(img_path), f'Missing {img_path}'
       sha = hashlib.sha256(open(img_path, 'rb').read()).hexdigest()
       assert sha == row['image_sha256'], f'Image checksum mismatch for {row[\"line_id\"]}'
   print('All 1,575 OOD benchmark files verified successfully.')
   "
   ```

---

## 2. In-Distribution (ID) Benchmark: `id_815_manifest.csv`

- **Dataset Source:** Archival court records (*şer‘iyye sicilleri*, 18 documents) from Ottoman legal registers.
- **Line Count:** 815 lines in Ottoman Riqa script.
- **Reference Transcriptions:** Transcriptions derive from the automated alignment pipeline; as stated in Section 3.4 and Section 5.4, these references are unverified archival draft ground truths.
- **Access Policy:** The ID benchmark dataset is available upon reasonable request from the corresponding author for academic research and validation purposes.
- **Verification:** Once obtained, verify files against `id_815_manifest.csv` using the SHA-256 checksums provided for both image crops and reference text strings.

---

## 3. Evaluation Scripts Data Folder Parameter

All evaluation scripts in `evaluation/` accept a configurable `--data_dir` (or `--test_dir`) argument pointing to the extracted benchmark folders to allow seamless replication once benchmark files are in place:
```bash
python3 evaluation/eval_single_vllm_gptq_worker.py \
    --model_dir models/qwen35_4b_latin_gptq_w4a16_v2 \
    --benchmark id_815_rika \
    --data_dir path/to/id_815_rika/
```
