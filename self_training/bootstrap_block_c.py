#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
QWEN3.5-9B LATİN MERGED V2 (GPTQ W4A16) - VERİ SETİ HİZALAMA VE ÜRETİM SCRİPTİ
==================================================================================
Amaç        : line_based_data içerisindeki 74.417 ham satır görselini vLLM ile
              çıkarıp tam sayfa transkripsiyonları (annotations_transliteration)
              ile hizalayarak yeni satır bazlı altın standart (Ground Truth) veri seti üretmek.
Model       : <PROJECT_ROOT>/models/qwen35_9b_latin_gptq_w4a16_v2
Girdi Dizin : <PROJECT_ROOT>/data-set/line_based_data (74.417 Görsel)
Metin Dizin : <PROJECT_ROOT>/data-set/annotations_transliteration
Çıktı Dizin : <PROJECT_ROOT>/data-set/ground_truth_line_based_dataset_qwen35_latin_v2
Donanım     : NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
Teknoloji   : vLLM (v0.26.0) + compressed-tensors (GPTQ W4A16) + Fuzzy Matching
==================================================================================
"""

import os
import sys

# ============================================================
# 0. SİSTEM VE CUDA OPTİMİZASYONLARI
# ============================================================
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import re
import time
import math
import json
import difflib
import shutil
import concurrent.futures
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# 1. KLASÖR VE MODEL YAPILANDIRMASI
# ============================================================
BASE_DATASET_DIR = "<PROJECT_ROOT>/data-set"

LINE_BASED_DIR = os.path.join(BASE_DATASET_DIR, "line_based_data")
ANNOTATIONS_DIR = os.path.join(BASE_DATASET_DIR, "annotations_transliteration")

MODEL_PATH = "<PROJECT_ROOT>/models/qwen35_9b_latin_gptq_w4a16_v2"

OUTPUT_DIR = os.path.join(BASE_DATASET_DIR, "ground_truth_line_based_dataset_qwen35_latin_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAILED_LOG_PATH = os.path.join(OUTPUT_DIR, "failed_lines_log.txt")
STATS_LOG_PATH = os.path.join(OUTPUT_DIR, "alignment_summary.json")

PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)

# ============================================================
# 2. HİPERPARAMETRELER (Eğitim ve Test Scripti ile %100 Senkronize)
# ============================================================
MIN_PIXELS = 128 * 28 * 28   # 100.352 px (128 vision token)
MAX_PIXELS = 768 * 28 * 28   # 602.112 px (768 vision token)
MAX_DIM = 1568               # Maksimum kenar sınırı (VLM aşımı önleyici)

MAX_MODEL_LEN = 1280         # (768 vision + 30 prompt + 256 new tokens = ~1054 < 1280)
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.12
GPU_MEMORY_UTILIZATION = 0.85
SAVE_BATCH_SIZE = 500
MAX_BATCH_SIZE = 32          # Sayfa içi satır çıkarım alt grup boyutu
MIN_ALIGN_RATIO = 0.60       # Altın standart eşleşme güven eşiği (%60+)

# ============================================================
# 3. GÖRSEL İŞLEME VE YARDIMCI FONKSİYONLAR
# ============================================================
def resize_for_vlm(image, max_dim=MAX_DIM, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """
    Satır görselini modelin görme işlemcisi (VLM) piksel bütçesine ve 28x28
    grid katsayısına göre en boy oranını koruyarak yeniden boyutlandırır.
    """
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
    
    pixels = w * h
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        w = int(w * scale)
        h = int(h * scale)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        w = int(w * scale)
        h = int(h * scale)

    w = max(28, int(round(w / 28.0) * 28))
    h = max(28, int(round(h / 28.0) * 28))
    return image.resize((w, h), Image.Resampling.LANCZOS)

def clean_prediction_text(pred: str) -> str:
    """Model tahmininden thinking bloklarını ve özel tokenleri temizler"""
    if not pred:
        return ""
    if "</think>" in pred:
        pred = pred.split("</think>")[-1]
    pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL)
    pred = pred.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    pred = pred.replace('\n', ' ').replace('\t', ' ')
    return " ".join(pred.split()).strip()

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\t', ' ')
    return " ".join(text.split()).strip()

def save_pair(args):
    """Arka planda çok iş parçacıklı olarak görsel ve Ground Truth TXT yazar"""
    src_img, dest_img, txt_path, text = args
    try:
        shutil.copy2(src_img, dest_img)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"⚠️ Dosya yazma hatası ({src_img}): {e}")

# ============================================================
# 4. BULANIK METİN HİZALAMA ALGORİTMASI (FUZZY ALIGNMENT v5)
# ============================================================
def align_ocr_to_ground_truth_v5(model_predictions, ground_truth_text, min_ratio=MIN_ALIGN_RATIO):
    """
    Modelin satır satır yaptığı transkripsiyon tahminlerini, dökümanın
    tam sayfa Latin transkripsiyon metni ile kayan pencere (sliding window)
    ve SequenceMatcher kullanarak en yüksek benzerlikle eşleştirir.
    """
    aligned_pairs = []
    gt_clean = clean_text(ground_truth_text)
    gt_words = gt_clean.split()
    current_word_idx = 0

    for i, pred in enumerate(model_predictions):
        pred_clean = clean_text(pred)
        if not pred_clean or len(pred_clean) < 2:
            aligned_pairs.append({"line_idx": i, "matched_gt": None, "status": "SKIPPED_GARBAGE"})
            continue

        pred_words = pred_clean.split()
        pred_word_count = len(pred_words)
        best_ratio = 0
        best_match_idx = -1
        best_match_word_count = 0
        best_matched_text = ""

        # Kayan pencere arama sınırı
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
            aligned_pairs.append({
                "line_idx": i,
                "matched_gt": best_matched_text,
                "status": "OK",
                "ratio": best_ratio
            })
            current_word_idx = best_match_idx + best_match_word_count
        else:
            aligned_pairs.append({
                "line_idx": i,
                "matched_gt": None,
                "status": f"LOW_CONFIDENCE ({best_ratio:.2f})"
            })

    return aligned_pairs

# ============================================================
# 5. ANA İŞLEM MOTORU (vLLM + RESUME LOGIC)
# ============================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("🚀 QWEN3.5-9B LATİN V2 - 74.417 SATIR HİZALAMA VE VERİ SETİ ÜRETİMİ")
    print(f"📅 Başlangıç Tarihi/Saati : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📌 Model Yolu             : {MODEL_PATH}")
    print(f"📂 Ham Satır Görselleri   : {LINE_BASED_DIR}")
    print(f"📄 Sayfa Transkripsiyon   : {ANNOTATIONS_DIR}")
    print(f"💾 Çıktı Hedef Klasörü    : {OUTPUT_DIR}")
    print(f"⚙️ Çözünürlük Bütçesi     : Min {MIN_PIXELS} px | Max {MAX_PIXELS} px | MaxDim {MAX_DIM} px")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: Model klasörü bulunamadı: {MODEL_PATH}")
        sys.exit(1)

    # 1. vLLM Motorunu Yükle
    print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ vLLM Motoru ve GPTQ W4A16 Modeli yükleniyor...")
    from vllm import LLM, SamplingParams
    
    t_load = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    print(f"✅ vLLM Modeli başarıyla yüklendi! (Yükleme süresi: {time.time() - t_load:.2f} sn)")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )

    prompt_template = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{PROMPT_TEXT}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 2. Batch Çıkarım Fonksiyonu
    def predict_vllm_batch(image_paths):
        all_clean_outputs = []
        for batch_start in range(0, len(image_paths), MAX_BATCH_SIZE):
            batch_paths = image_paths[batch_start : batch_start + MAX_BATCH_SIZE]
            vllm_inputs = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = resize_for_vlm(img)
                    vllm_inputs.append({
                        "prompt": prompt_template,
                        "multi_modal_data": {"image": img}
                    })
                except Exception:
                    dummy_img = Image.new("RGB", (224, 32), (255, 255, 255))
                    vllm_inputs.append({
                        "prompt": prompt_template,
                        "multi_modal_data": {"image": dummy_img}
                    })

            try:
                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
                for out in outputs:
                    raw_text = out.outputs[0].text
                    all_clean_outputs.append(clean_prediction_text(raw_text))
            except Exception as e:
                # Toplu grupta hata olursa tek tek güvenli modda çıkar
                for single_input in vllm_inputs:
                    try:
                        single_out = llm.generate([single_input], sampling_params=sampling_params, use_tqdm=False)
                        all_clean_outputs.append(clean_prediction_text(single_out[0].outputs[0].text))
                    except Exception:
                        all_clean_outputs.append("")

        return all_clean_outputs

    # 3. Resume Logic & İşlenecek Görselleri Tara
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔍 Önceki kayıtlar taranıyor (Resume Logic)...")
    processed_files = set()

    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.txt') and f != "failed_lines_log.txt":
                processed_files.add(f.rsplit('.', 1)[0] + ".png")

    if os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "r", encoding="utf-8", errors="ignore") as log_f:
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

    print(f"📦 İşlenecek: {len(images_by_doc)} Döküman Sayfası | Toplam {target_lines} Yeni Satır Görseli")

    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    toplam_eslesme = 0
    toplam_hata = 0
    pending_save_tasks = []

    # 4. Ana Hizalama Döngüsü
    for doc_base, img_list in tqdm(images_by_doc.items(), desc="vLLM + Fuzzy Alignment"):
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
        aligned_results = align_ocr_to_ground_truth_v5(predictions, full_gt_text, min_ratio=MIN_ALIGN_RATIO)

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

    # Kalan kayıtları yaz
    if pending_save_tasks:
        list(save_executor.map(save_pair, pending_save_tasks))
        pending_save_tasks = []

    elapsed_total = time.time() - t_start
    
    # 5. Özet İstatistikleri Kaydet
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_lines_processed": target_lines,
        "matched_ground_truth_lines": toplam_eslesme,
        "failed_or_skipped_lines": toplam_hata,
        "match_success_rate_pct": round((toplam_eslesme / max(1, target_lines)) * 100, 2),
        "total_elapsed_seconds": round(elapsed_total, 2),
        "output_directory": OUTPUT_DIR
    }
    with open(STATS_LOG_PATH, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("🎉 HİZALAMA VE VERİ SETİ ÜRETİMİ TAMAMLANDI!")
    print(f"✅ Kurtarılan / Eşleşen Altın Standart Satır Sayısı : {toplam_eslesme}")
    print(f"⚠️ Eşleşmeyen / Düşük Güvenli Satır Sayısı           : {toplam_hata}")
    print(f"📈 Eşleşme Başarı Oranı                             : %{summary_data['match_success_rate_pct']}")
    print(f"📁 Üretilen Veri Seti Konumu                        : {OUTPUT_DIR}")
    print(f"⏱️ Toplam Geçen Süre                                : {elapsed_total / 60:.2f} dakika")
    print("=" * 80)

if __name__ == '__main__':
    main()
