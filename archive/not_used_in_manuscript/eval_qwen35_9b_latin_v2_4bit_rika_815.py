#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
QWEN3.5-9B LATİN MERGED V2 - 4-BIT NF4 + FLASH-ATTENTION-2 TEST VE DEĞERLENDİRME
==================================================================================
Model       : /home/fatih/Documents/ottoman/models/qwen35_9b_latin_merged_v2
Veri Kümesi : /home/fatih/Documents/ottoman/data-set/815_line_based_in_distrubition_rika_latin
Çıktı       : /home/fatih/Documents/ottoman/evaluation_results
Donanım     : NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
Teknoloji   : 4-Bit NF4 (BitsAndBytes) + Native bfloat16 + FlashAttention-2 + TF32
==================================================================================
"""

import os
import sys
import gc
import time
import datetime
import re
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import jiwer
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

# ============================================================
# 0. CUDA & ORTAM OPTİMİZASYONLARI
# ============================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. DİZİNLER VE SABİTLER
# ============================================================
MODEL_PATH = "/home/fatih/Documents/ottoman/models/qwen35_9b_latin_merged_v2"
DATASET_PATH = "/home/fatih/Documents/ottoman/data-set/815_line_based_in_distrubition_rika_latin"
OUTPUT_DIR = "/home/fatih/Documents/ottoman/evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_TXT_PATH = os.path.join(OUTPUT_DIR, "eval_qwen35_9b_latin_v2_rika_815_report.txt")
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "eval_qwen35_9b_latin_v2_rika_815_results.csv")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "eval_qwen35_9b_latin_v2_rika_815.log")

PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)

# Fine-tuning çözünürlük parametreleri ile birebir uyumlu
MIN_PIXELS = 128 * 28 * 28   # 100.352 px
MAX_PIXELS = 768 * 28 * 28   # 602.112 px
MAX_NEW_TOKENS = 256

# ============================================================
# 2. LOGGING YARDIMCISI
# ============================================================
class DualLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# ============================================================
# 3. METİN AYIKLAMA VE NORMALİZASYON FONKSİYONLARI
# ============================================================
def clean_extract(text: str) -> str:
    """
    Düşünme tokenları (<think>...</think>) ve fazla satırları temizler.
    İlk anlamlı satırı döndürür.
    """
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "").replace("<think>", "")
    text = text.replace("</reasoning>", "").replace("<reasoning>", "")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else ""

def normalize_latin(text: str) -> str:
    """
    Latin harfli Osmanlıca transkripsiyon için akıllı normalizasyon:
    - Küçük harfe çevirme
    - Şapkalı harf (â, î, û vb.) ve transkripsiyon işaretlerini sadeleştirme
    - Noktalama ve fazla boşlukları temizleme
    """
    text = clean_extract(text)
    if not text:
        return ""
    text = text.lower()

    # Özel Osmanlıca transkripsiyon işaretleri (ayn/hemze: ‘ ’ ` ' ʿ ʾ ʻ)
    text = re.sub(r"[‘’`'ʿʾʻ\^]", "", text)

    # Uzun ünlüler (şapkalı / uzatmalı harfler)
    text = text.replace("â", "a").replace("î", "i").replace("û", "u").replace("ô", "o").replace("ê", "e")
    text = text.replace("ā", "a").replace("ī", "i").replace("ū", "u").replace("ō", "o").replace("ē", "e")

    # Özel konsonant transkripsiyonları
    char_map = {
        "ḥ": "h", "ḫ": "h", "ḫ": "h", "ẖ": "h",
        "ṣ": "s", "ṣ": "s", "š": "s", "ş": "s",
        "ḍ": "d", "ḍ": "d", "ḏ": "d", "ż": "z", "ẓ": "z", "ẓ": "z", "ẕ": "z",
        "ṭ": "t", "ṭ": "t", "ṯ": "t",
        "ġ": "g", "ġ": "g", "ğ": "g",
        "ñ": "n", "ñ": "n", "ŋ": "n",
        "ḳ": "k", "ḳ": "k",
        "ç": "c", "ç": "c",
        "ı": "i", "i̇": "i",
        "ö": "o", "ö": "o",
        "ü": "u", "ü": "u"
    }
    for k, v in char_map.items():
        text = text.replace(k, v)

    # Noktalama işaretlerini boşlukla değiştir
    text = re.sub(r"[^\w\s]", " ", text)
    # Fazla boşlukları teke indir
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# 4. METRİK HESAPLAMA
# ============================================================
def calculate_metrics_summary(refs, hyps):
    """
    CER, WER (Micro & Macro) ve Tam Eşleşme (Accuracy) hesaplar.
    """
    cers, wers = [], []
    exact_matches = 0

    for r, h in zip(refs, hyps):
        if len(r) == 0:
            c = 0.0 if len(h) == 0 else 1.0
        else:
            c = jiwer.cer(r, h)
        cers.append(c)

        r_words = r.split()
        h_words = h.split()
        if len(r_words) == 0:
            w = 0.0 if len(h_words) == 0 else 1.0
        else:
            w = jiwer.wer(r, h)
        wers.append(w)

        if r == h:
            exact_matches += 1

    macro_cer = float(np.mean(cers)) * 100.0 if cers else 0.0
    macro_wer = float(np.mean(wers)) * 100.0 if wers else 0.0

    total_ref_chars = sum(len(r) for r in refs)
    total_ref_words = sum(len(r.split()) for r in refs)

    total_char_errs = sum(jiwer.cer(r, h) * len(r) for r, h in zip(refs, hyps) if len(r) > 0)
    total_word_errs = sum(jiwer.wer(r, h) * len(r.split()) for r, h in zip(refs, hyps) if len(r.split()) > 0)

    micro_cer = (total_char_errs / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0
    micro_wer = (total_word_errs / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    accuracy = (exact_matches / len(refs) * 100.0) if len(refs) > 0 else 0.0

    return {
        "macro_cer": macro_cer,
        "macro_wer": macro_wer,
        "micro_cer": micro_cer,
        "micro_wer": micro_wer,
        "accuracy": accuracy,
        "exact_matches": exact_matches,
        "total_samples": len(refs)
    }

# ============================================================
# 5. MODEL VE PROCESSOR YÜKLEME (4-BIT NF4 + FLASH-ATTN-2)
# ============================================================
def load_model_and_processor():
    print("=" * 80)
    print("🚀 QWEN3.5-9B LATİN MERGED V2 MODELİ YÜKLENİYOR (4-BIT NF4 + FLASH-ATTENTION-2)...")
    print("=" * 80)
    print(f"📌 Model Yolu           : {MODEL_PATH}")
    print(f"⚡ Cihaz                : {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"💾 GPU & VRAM           : {gpu_name} ({vram_gb:.2f} GB)")
    print(f"🖼️ Görsel Çözünürlük    : min={MIN_PIXELS} px, max={MAX_PIXELS} px")
    print("-" * 80)

    # AutoProcessor Yükle
    print("⏳ AutoProcessor yükleniyor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )
    print("✅ AutoProcessor hazır.")

    # 4-bit NF4 Quantization (Fine-Tuning ile Birebir Aynı Altyapı)
    print("⏳ Model 4-bit NF4 + bfloat16 + FlashAttention-2 ile GPU'ya yükleniyor...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    t0 = time.time()
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        print("✅ Model FlashAttention-2 desteğiyle yüklendi.")
    except Exception as e:
        print(f"⚠️ FlashAttention-2 yükleme uyarısı ({e}), SDPA'ya geçiliyor...")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        print("✅ Model SDPA dikkat mekanizmasıyla yüklendi.")

    load_time = time.time() - t0
    model.eval()
    model.config.use_cache = True

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"✅ Model 4-bit NF4 olarak hazır! (Yükleme süresi: {load_time:.2f} sn)")
    print(f"📊 Bellek Durumu: Allocated={allocated_gb:.2f} GB | Reserved={reserved_gb:.2f} GB\n")

    return model, processor

# ============================================================
# 6. VERİ SETİNİ YÜKLEME
# ============================================================
def load_dataset(dataset_dir):
    print("=" * 80)
    print(f"📂 TEST VERİ SETİ TARANIYOR: {dataset_dir}")
    print("=" * 80)
    if not os.path.exists(dataset_dir):
        print(f"❌ HATA: Veri seti dizini bulunamadı: {dataset_dir}")
        sys.exit(1)

    valid_exts = ('.png', '.jpg', '.jpeg')
    all_files = sorted(os.listdir(dataset_dir))
    samples = []

    for fname in all_files:
        if fname.lower().endswith(valid_exts):
            stem = os.path.splitext(fname)[0]
            txt_path = os.path.join(dataset_dir, stem + ".txt")
            img_path = os.path.join(dataset_dir, fname)
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    gt_text = f.read().strip()
                if gt_text:
                    samples.append({
                        "id": stem,
                        "filename": fname,
                        "image_path": img_path,
                        "gt_text": gt_text
                    })

    print(f"✅ Toplam Eşleşen Test Satırı (Görsel + Metin): {len(samples)}")
    if len(samples) == 0:
        print("❌ HATA: Test seti içinde eşleşen örnek bulunamadı!")
        sys.exit(1)
    return samples

# ============================================================
# 7. TEKİL ÇIKARIM FONKSİYONU
# ============================================================
def predict_sample(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT}
            ]
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.0,
            num_beams=1
        )
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0, input_len:]
        prediction = processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

    return clean_extract(prediction)

# ============================================================
# 8. ANA DEĞERLENDİRME VE RAPORLAMA DÖNGÜSÜ
# ============================================================
def main():
    sys.stdout = DualLogger(LOG_FILE_PATH)
    start_all = time.time()

    print("\n" + "=" * 80)
    print("🎯 OSMANLICA LATİN TRANSKRİPSİYON 815 SATIRLIK TESTİ BAŞLATILIYOR")
    print(f"📅 Tarih/Saat: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Model ve Veri Yükle
    model, processor = load_model_and_processor()
    samples = load_dataset(DATASET_PATH)

    # 2. Test Döngüsü
    print("\n" + "=" * 80)
    print(f"🔮 {len(samples)} ADET TEST SATIRI İÇİN ÇIKARIM VE METRİK HESAPLAMA BAŞLIYOR...")
    print("=" * 80)

    results = []
    latencies = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    progress_bar = tqdm(samples, desc="🚀 Değerlendiriliyor", unit="satır", ncols=100)

    for idx, item in enumerate(progress_bar, 1):
        t_sample_start = time.time()
        try:
            pred_text = predict_sample(model, processor, item["image_path"])
        except Exception as e:
            print(f"\n❌ [{item['id']}] Çıkarım hatası: {e}")
            pred_text = ""

        sample_latency = time.time() - t_sample_start
        latencies.append(sample_latency)

        gt_raw = item["gt_text"]
        pred_raw = pred_text

        gt_norm = normalize_latin(gt_raw)
        pred_norm = normalize_latin(pred_raw)

        # Ham Metrikler
        c_raw = jiwer.cer(gt_raw, pred_raw) if len(gt_raw) > 0 else (0.0 if len(pred_raw) == 0 else 1.0)
        w_raw = jiwer.wer(gt_raw, pred_raw) if len(gt_raw.split()) > 0 else (0.0 if len(pred_raw.split()) == 0 else 1.0)

        # Normalizasyonlu Metrikler
        c_norm = jiwer.cer(gt_norm, pred_norm) if len(gt_norm) > 0 else (0.0 if len(pred_norm) == 0 else 1.0)
        w_norm = jiwer.wer(gt_norm, pred_norm) if len(gt_norm.split()) > 0 else (0.0 if len(pred_norm.split()) == 0 else 1.0)

        # chrF++ Skoru
        chrf_score = 0.0
        if HAS_SACREBLEU and gt_raw and pred_raw:
            try:
                chrf_score = sacrebleu.sentence_chrf(pred_raw, [gt_raw], word_order=2).score
            except Exception:
                chrf_score = 0.0

        results.append({
            "No": idx,
            "Sample_ID": item["id"],
            "Filename": item["filename"],
            "GT_Raw": gt_raw,
            "Pred_Raw": pred_raw,
            "GT_Norm": gt_norm,
            "Pred_Norm": pred_norm,
            "CER_Raw": round(c_raw, 4),
            "WER_Raw": round(w_raw, 4),
            "CER_Norm": round(c_norm, 4),
            "WER_Norm": round(w_norm, 4),
            "chrF++": round(chrf_score, 2),
            "Latency_sec": round(sample_latency, 3),
            "Exact_Match_Raw": int(gt_raw == pred_raw),
            "Exact_Match_Norm": int(gt_norm == pred_norm)
        })

        # Canlı bar açıklaması
        current_avg_cer = np.mean([r["CER_Raw"] for r in results]) * 100
        current_norm_cer = np.mean([r["CER_Norm"] for r in results]) * 100
        progress_bar.set_postfix({"CER(Ham)": f"{current_avg_cer:.1f}%", "CER(Norm)": f"{current_norm_cer:.1f}%"})

    total_infer_time = sum(latencies)
    avg_latency = np.mean(latencies)
    speed_samples_per_sec = len(samples) / total_infer_time if total_infer_time > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print(f"\n⚡ Tüm Çıkarımlar Tamamlandı! Toplam Süre: {total_infer_time:.2f} sn ({speed_samples_per_sec:.2f} satır/sn)\n")

    # 3. İstatistik ve Metrik Özetleri
    df = pd.DataFrame(results)

    gt_raws = df["GT_Raw"].tolist()
    pred_raws = df["Pred_Raw"].tolist()
    gt_norms = df["GT_Norm"].tolist()
    pred_norms = df["Pred_Norm"].tolist()

    raw_summary = calculate_metrics_summary(gt_raws, pred_raws)
    norm_summary = calculate_metrics_summary(gt_norms, pred_norms)
    avg_chrf = df["chrF++"].mean() if HAS_SACREBLEU else 0.0

    # 4. Rapor Metni Oluşturma
    report_lines = []
    def rlog(msg=""):
        report_lines.append(msg)

    rlog("=" * 85)
    rlog("  QWEN3.5-9B LATİN MERGED V2 (4-BIT NF4 + FLASH-ATTENTION-2) DEĞERLENDİRME RAPORU")
    rlog("=" * 85)
    rlog(f"📅 Test Tarihi            : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rlog(f"📌 Model Yolu            : {MODEL_PATH}")
    rlog(f"📂 Test Seti             : {DATASET_PATH}")
    rlog(f"📊 Toplam Satır Sayısı    : {len(samples)} satır")
    rlog(f"⚡ Çıkarım Motoru         : 4-Bit NF4 (BitsAndBytes) + FlashAttention-2 (bfloat16)")
    rlog(f"⏱️ Toplam Çıkarım Süresi  : {total_infer_time:.2f} saniye")
    rlog(f"🚀 Ortalama Hız           : {speed_samples_per_sec:.2f} satır / saniye ({avg_latency*1000:.1f} ms/satır)")
    rlog(f"💾 Peak GPU VRAM          : {peak_vram_gb:.2f} GB / 16.0 GB")
    rlog("-" * 85)
    rlog("📈 PERFORMANS METRİKLERİ TABLOSU")
    rlog("-" * 85)

    metrics_table = [
        ["Metrik", "Ham (Raw) Değer", "Akıllı Normalizasyonlu (Norm) Değer"],
        ["Micro CER (Karakter Hata Oranı)", f"%{raw_summary['micro_cer']:.2f}", f"%{norm_summary['micro_cer']:.2f}"],
        ["Macro CER (Ortalama CER)", f"%{raw_summary['macro_cer']:.2f}", f"%{norm_summary['macro_cer']:.2f}"],
        ["Micro WER (Kelime Hata Oranı)", f"%{raw_summary['micro_wer']:.2f}", f"%{norm_summary['micro_wer']:.2f}"],
        ["Macro WER (Ortalama WER)", f"%{raw_summary['macro_wer']:.2f}", f"%{norm_summary['macro_wer']:.2f}"],
        ["Tam Satır Doğruluğu (Exact Match)", f"%{raw_summary['accuracy']:.2f} ({raw_summary['exact_matches']}/{len(samples)})", f"%{norm_summary['accuracy']:.2f} ({norm_summary['exact_matches']}/{len(samples)})"],
        ["chrF++ Skoru (SacreBLEU)", f"{avg_chrf:.2f}" if HAS_SACREBLEU else "N/A", "-"]
    ]

    # Format Tablo
    col_w = [36, 22, 24]
    for row in metrics_table:
        rlog(f"| {row[0]:<{col_w[0]}} | {row[1]:<{col_w[1]}} | {row[2]:<{col_w[2]}} |")
    rlog("-" * 85)

    # Örnek Çıktılar (İlk 15 ve En Zor 5 Örnek)
    rlog("\n🔍 DETAYLI ÖRNEK ÇIKTILAR VE KARŞILAŞTIRMALAR")
    rlog("=" * 85)

    rlog("\n--- [A] İLK 10 TEST SATIRI ÇIKTILARI ---")
    for i in range(min(10, len(results))):
        r = results[i]
        rlog(f"[{r['No']:>3}] Dosya: {r['Filename']} | Raw CER: %{r['CER_Raw']*100:.1f} | Norm CER: %{r['CER_Norm']*100:.1f} | Süre: {r['Latency_sec']}s")
        rlog(f"      📌 HEDEF (GT) : {r['GT_Raw']}")
        rlog(f"      🤖 TAHMİN     : {r['Pred_Raw']}")
        rlog(f"      ✨ NORM (GT)  : {r['GT_Norm']}")
        rlog(f"      ✨ NORM (PR)  : {r['Pred_Norm']}")
        rlog("-" * 85)

    worst_samples = df.sort_values(by="CER_Raw", ascending=False).head(5).to_dict("records")
    rlog("\n--- [B] EN YÜKSEK HATA ALINAN (EN ZOR) 5 ÖRNEK ---")
    for r in worst_samples:
        rlog(f"[{r['No']:>3}] Dosya: {r['Filename']} | Raw CER: %{r['CER_Raw']*100:.1f} | Norm CER: %{r['CER_Norm']*100:.1f}")
        rlog(f"      📌 HEDEF (GT) : {r['GT_Raw']}")
        rlog(f"      🤖 TAHMİN     : {r['Pred_Raw']}")
        rlog("-" * 85)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 5. Dosyaları Kaydet
    print("\n" + "=" * 80)
    print("💾 SONUÇ DOSYALARI KAYDEDİLİYOR...")
    print("=" * 80)

    # TXT Rapor
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅ Rapor Metin Dosyası: {REPORT_TXT_PATH}")

    # CSV Sonuçlar
    df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Satır Bazlı CSV     : {RESULTS_CSV_PATH}")
    print(f"✅ Canlı Log Dosyası   : {LOG_FILE_PATH}")

    total_wall_time = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"🎉 TÜM DEĞERLENDİRME BAŞARIYLA TAMAMLANDI! Toplam Süre: {total_wall_time/60:.2f} dakika.")
    print("=" * 80)

if __name__ == "__main__":
    main()
