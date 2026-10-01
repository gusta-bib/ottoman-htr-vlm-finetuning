#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
TÜBİTAK 1001 - OSMANLICA HTR PROJESİ
TEKİL MODEL GPTQ W4A16 KUANTİZASYON ÇALIŞANI (İZOLE SUBPROCESS)
==================================================================================
Bu script her model için bağımsız bir Python sürecinde çalışır.
İşlem bittiğinde süreç sonlanır ve Linux / NVIDIA sürücüsü GPU VRAM'ini 
%100 kernel seviyesinde temizler.
==================================================================================
"""

import os
import sys
import gc
import json
import time
import argparse
import random
import torch
from PIL import Image, ImageFile
from datasets import Dataset as HFDataset
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 576
MIN_PIXELS = 128 * 28 * 28   # 100.352 px
MAX_PIXELS = 768 * 28 * 28   # 602.112 px

def load_split(split_dir):
    samples = []
    valid_exts = ('.png', '.jpg', '.jpeg')
    if not os.path.exists(split_dir):
        return samples
    for fname in sorted(os.listdir(split_dir)):
        if not fname.lower().endswith(valid_exts):
            continue
        stem = os.path.splitext(fname)[0]
        txt_path = os.path.join(split_dir, stem + ".txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            gt_text = f.read().strip()
        if gt_text:
            samples.append({"image": os.path.join(split_dir, fname), "text": gt_text})
    return samples

def data_collator(batch):
    assert len(batch) == 1, "VLM kalibrasyonu batch_size=1 gerektirir"
    example = batch[0]
    collated = {}
    for k, v in example.items():
        if k == "pixel_values":
            collated[k] = torch.tensor(v, dtype=torch.bfloat16)
        elif k == "image_grid_thw":
            collated[k] = torch.tensor(v, dtype=torch.long)
        elif k in ("input_ids", "attention_mask", "mm_token_type_ids"):
            t = torch.tensor(v)
            collated[k] = t.unsqueeze(0)
        else:
            collated[k] = torch.tensor(v)
    return collated

def main():
    parser = argparse.ArgumentParser(description="Tekil Model GPTQ Kuantizasyon Çalışanı")
    parser.add_argument("--input_dir", type=str, required=True, help="Girdi model dizini")
    parser.add_argument("--output_dir", type=str, required=True, help="Çıktı GPTQ model dizini")
    parser.add_argument("--dataset_root", type=str, default="<PROJECT_ROOT>/data-set/latin_pixel_cleaned", help="Veri seti kök dizini")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset_root = args.dataset_root

    print("=" * 80)
    print(f"🚀 İZOLE KUANTİZASYON ÇALIŞANI BAŞLATILDI")
    print(f"📁 Girdi  : {input_dir}")
    print(f"💾 Çıktı  : {output_dir}")
    print("=" * 80)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Girdi modeli bulunamadı: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 1. Model ve Processor Yükle
    print("⏳ AutoProcessor ve Model yükleniyor (BF16)...")
    processor = AutoProcessor.from_pretrained(
        input_dir,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    max_memory = {0: "9GiB", "cpu": "32GiB"}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        input_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    print(f"✅ Model ve Processor başarıyla yüklendi. (VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB)")

    # 2. Kalibrasyon Verisini Yükle
    train_dir = os.path.join(dataset_root, "train")
    train_samples = load_split(train_dir)
    if len(train_samples) == 0:
        raise RuntimeError(f"Train veri setinde örnek bulunamadı: {train_dir}")

    random.seed(42)
    random.shuffle(train_samples)
    calib_samples = train_samples[:NUM_CALIBRATION_SAMPLES]
    print(f"📋 Kalibrasyon için {len(calib_samples)} örnek hazırlandı.")

    raw_dataset = HFDataset.from_list(calib_samples)

    def preprocess(example):
        image = Image.open(example["image"]).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT_TEXT}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": example["text"]}]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        inputs = processor(
            text=[text], images=[image],
            padding=False, truncation=True, max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            if image_grid_thw.ndim == 3:
                image_grid_thw = image_grid_thw.squeeze(0)
            elif image_grid_thw.ndim == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)

        out = {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "pixel_values": pixel_values.tolist(),
        }
        if image_grid_thw is not None:
            out["image_grid_thw"] = image_grid_thw.tolist()
        if mm_token_type_ids is not None:
            out["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return out

    print("⚙️ Kalibrasyon tensörleri haritalanıyor...")
    calib_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

    # 3. GPTQ Modifier
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            "re:.*visual.*",
            "re:.*norm.*",
            "re:.*embed.*",
            "re:.*linear_attn\\.in_proj_qkv.*",
            "re:.*linear_attn\\.in_proj_z.*",
            "re:.*linear_attn\\.in_proj_b.*",
            "re:.*linear_attn\\.in_proj_a.*",
            "re:.*linear_attn\\.conv1d.*",
            "re:.*linear_attn\\.out_proj.*",
        ],
    )

    print("⚙️ GPTQ W4A16 oneshot kalibrasyonu ve Hessian hesaplanıyor...")
    torch.cuda.empty_cache()
    gc.collect()

    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        data_collator=data_collator,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=len(calib_samples),
    )

    # 4. Modeli Kaydet
    print(f"💾 Quantize model kaydediliyor: {output_dir}")
    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)

    total_time = (time.time() - t_start) / 60.0
    print(f"✅ İZOLE KUANTİZASYON BAŞARIYLA TAMAMLANDI! (Süre: {total_time:.2f} dakika)")

if __name__ == "__main__":
    main()
