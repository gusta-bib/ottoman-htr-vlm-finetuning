"""
Azra - Latin (Qwen3.5-9B-GPTQ-W4A16) - Tüm Veri Seti Çıkarım ve Hizalama Scripti
Yerel Sistem: NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM) - vLLM Optimize (Thinking/Reasoning Kapalı)
"""

import os
import re
import sys
import time
import json
import difflib
import shutil
import concurrent.futures
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ============================================================
# 1. KLASÖR VE MODEL YOLLARI
# ============================================================
BASE_DATASET_DIR = "/home/fatih/Documents/ottoman/data-set"

LINE_BASED_DIR = os.path.join(BASE_DATASET_DIR, "line_based_data")
ANNOTATIONS_DIR = os.path.join(BASE_DATASET_DIR, "annotations_transliteration")

MODEL_PATH = "/home/fatih/Documents/ottoman/models/Azra - Latin - GPTQ-W4A16"

OUTPUT_DIR = os.path.join(BASE_DATASET_DIR, "ground_truth_line_based_dataset_azra_latin")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAILED_LOG_PATH = os.path.join(OUTPUT_DIR, "failed_lines_log.txt")

# ⚠️ Fine-Tuning ile birebir aynı prompt
PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)

# ============================================================
# 2. DOSYA YAZMA VE YARDIMCI FONKSİYONLAR
# ============================================================
SAVE_BATCH_SIZE = 500

# Qwen3.5-VL vision token limiti: büyük görseller çok fazla token üretip
# HF processor truncation hatasına yol açıyor. Görselleri ön-boyutlandır.
MAX_IMAGE_PIXELS = 150000  # ~192 vision token üretir (güvenli sınır)

def resize_if_needed(img, max_pixels=MAX_IMAGE_PIXELS):
    """Büyük görselleri küçülterek vision token sayısını sınırla"""
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w = max(28, int(w * scale))
        new_h = max(28, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def save_pair(args):
    src_img, dest_img, txt_path, text = args
    try:
        shutil.copy2(src_img, dest_img)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"⚠️ Yazma hatası ({src_img}): {e}")

def clean_prediction_text(pred):
    """Thinking bloklarını ve gereksiz boşlukları ayıklar"""
    if "</think>" in pred:
        pred = pred.split("</think>")[-1]
    pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL)
    pred = pred.replace('\n', ' ').replace('\t', ' ')
    return " ".join(pred.split()).strip()

def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    return " ".join(text.split())

def align_ocr_to_ground_truth_v5(model_predictions, ground_truth_text, min_ratio=0.60):
    aligned_pairs = []
    gt_clean = clean_text(ground_truth_text)
    gt_words = gt_clean.split()
    current_word_idx = 0

    for i, pred in enumerate(model_predictions):
        pred_clean = clean_text(pred)
        if not pred_clean or len(pred_clean) < 3:
            aligned_pairs.append({"line_idx": i, "matched_gt": None, "status": "SKIPPED_GARBAGE"})
            continue

        pred_words = pred_clean.split()
        pred_word_count = len(pred_words)
        best_ratio = 0
        best_match_idx = -1
        best_match_word_count = 0
        best_matched_text = ""

        search_limit = min(current_word_idx + 45, len(gt_words))
        for start_idx in range(current_word_idx, search_limit):
            min_len = max(1, pred_word_count - 3)
            max_len = pred_word_count + 5

            for length in range(min_len, max_len):
                if start_idx + length > len(gt_words):
                    break
                candidate_str = " ".join(gt_words[start_idx : start_idx + length])
                ratio = difflib.SequenceMatcher(None, pred_clean.lower(), candidate_str.lower()).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match_idx = start_idx
                    best_match_word_count = length
                    best_matched_text = candidate_str

        if best_ratio >= min_ratio:
            aligned_pairs.append({"line_idx": i, "matched_gt": best_matched_text, "status": "OK", "ratio": best_ratio})
            current_word_idx = best_match_idx + best_match_word_count
        else:
            aligned_pairs.append({"line_idx": i, "matched_gt": None, "status": f"LOW_CONFIDENCE ({best_ratio:.2f})"})

    return aligned_pairs

# ============================================================
# 3. ANA İŞLEM FONKSİYONU
# ============================================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ Model ve sistem yapılandırması kontrol ediliyor...")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: Model klasörü bulunamadı: {MODEL_PATH}")
        sys.exit(1)

    config_path = os.path.join(MODEL_PATH, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not config.get("tie_word_embeddings", False):
            config["tie_word_embeddings"] = True
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print("✅ config.json güncellendi ('tie_word_embeddings': True).")
        else:
            print("ℹ️ config.json güncel ('tie_word_embeddings': True).")

    # vLLM Motorunu Başlat
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 vLLM Motoru Başlatılıyor (Azra - Latin - GPTQ-W4A16)...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    try:
        processor.image_processor.min_pixels = 3136
        processor.image_processor.max_pixels = 256 * 28 * 28
    except Exception:
        pass

    llm = LLM(
        model=MODEL_PATH,
        quantization="compressed-tensors",
        dtype="auto",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1}
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        repetition_penalty=1.15,
        stop_token_ids=[processor.tokenizer.eos_token_id]
    )

    MAX_BATCH_SIZE = 32  # Büyük dokümanları alt gruplara böl (token mismatch hatasını önler)

    def predict_vllm_batch(image_paths):
        all_clean_outputs = []

        # Alt gruplara böl
        for batch_start in range(0, len(image_paths), MAX_BATCH_SIZE):
            batch_paths = image_paths[batch_start:batch_start + MAX_BATCH_SIZE]
            vllm_inputs = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = resize_if_needed(img)  # Token overflow önleme
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": PROMPT_TEXT}
                            ]
                        }
                    ]
                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    vllm_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": img}
                    })
                except Exception:
                    dummy_img = Image.new("RGB", (256, 28), (255, 255, 255))
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": dummy_img},
                                {"type": "text", "text": "Hata"}
                            ]
                        }
                    ]
                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": dummy_img}})

            try:
                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
                for out in outputs:
                    raw_text = out.outputs[0].text
                    all_clean_outputs.append(clean_prediction_text(raw_text))
            except Exception as e:
                # Batch hata verirse tek tek dene
                print(f"  ⚠️ Batch hatası, tek tek işleniyor: {e}")
                for single_input in vllm_inputs:
                    try:
                        single_out = llm.generate([single_input], sampling_params=sampling_params, use_tqdm=False)
                        all_clean_outputs.append(clean_prediction_text(single_out[0].outputs[0].text))
                    except Exception:
                        all_clean_outputs.append("")  # Boş çıktı → SKIPPED_GARBAGE olacak

        return all_clean_outputs

    # Resume & Veri Hazırlığı
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔍 Önceki kayıtlar taranıyor (Resume Logic)...")
    processed_files = set()

    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.txt') and f != "failed_lines_log.txt":
                processed_files.add(f.rsplit('.', 1)[0] + ".png")

    if os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "r", encoding="utf-8") as log_f:
            for line in log_f:
                l = line.strip()
                if l:
                    processed_files.add(l)

    all_png_files = [f for f in os.listdir(LINE_BASED_DIR) if f.lower().endswith(".png")]
    images_by_doc = {}
    target_lines = 0

    for filename in all_png_files:
        if filename in processed_files:
            continue
        try:
            base_name, line_str = filename.rsplit('_', 1)
            line_num = int(line_str.split('.')[0])
            if base_name not in images_by_doc:
                images_by_doc[base_name] = []
            images_by_doc[base_name].append((line_num, filename))
            target_lines += 1
        except ValueError:
            pass

    print(f"📦 İşlenecek: {len(images_by_doc)} döküman | Toplam {target_lines} yeni satır")

    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    toplam_eslesme = 0
    toplam_hata = 0
    pending_save_tasks = []

    for doc_base, img_list in tqdm(images_by_doc.items(), desc="RTX 4070 Ti SUPER vLLM ile İşleniyor"):
        img_list.sort(key=lambda x: x[0])

        gt_file_path = os.path.join(ANNOTATIONS_DIR, f"{doc_base}.txt")
        if not os.path.exists(gt_file_path):
            continue

        try:
            with open(gt_file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_gt_text = f.read()
        except Exception:
            continue

        img_paths = [os.path.join(LINE_BASED_DIR, img_filename) for _, img_filename in img_list]
        predictions = predict_vllm_batch(img_paths)
        aligned_results = align_ocr_to_ground_truth_v5(predictions, full_gt_text, min_ratio=0.60)

        failed_this_doc = []

        for i, match in enumerate(aligned_results):
            img_filename = img_list[i][1]

            if match["status"] == "OK":
                matched_text = match['matched_gt']
                src_img_path = os.path.join(LINE_BASED_DIR, img_filename)
                dest_img_path = os.path.join(OUTPUT_DIR, img_filename)
                txt_filename = img_filename.rsplit('.', 1)[0] + ".txt"
                dest_txt_path = os.path.join(OUTPUT_DIR, txt_filename)

                pending_save_tasks.append((src_img_path, dest_img_path, dest_txt_path, matched_text))
                toplam_eslesme += 1
            else:
                toplam_hata += 1
                failed_this_doc.append(img_filename)

        if len(pending_save_tasks) >= SAVE_BATCH_SIZE:
            list(save_executor.map(save_pair, pending_save_tasks))
            pending_save_tasks = []

        if failed_this_doc:
            with open(FAILED_LOG_PATH, "a", encoding="utf-8") as log_f:
                for fname in failed_this_doc:
                    log_f.write(fname + "\n")

    if pending_save_tasks:
        list(save_executor.map(save_pair, pending_save_tasks))
        pending_save_tasks = []

    print("\n" + "=" * 80)
    print(f"🎉 İŞLEM TAMAMLANDI! (NVIDIA RTX 4070 Ti SUPER)")
    print(f"✅ Kurtarılan / Eşleşen Satır: {toplam_eslesme}")
    print(f"⚠️ Eşleşmeyen Satır: {toplam_hata}")
    print(f"📁 Çıktı Klasörü: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
