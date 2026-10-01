#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSMANLICA OCR - QWEN3.5-4B LATİN V2 (22.782 Örnek) ABLASYON 2 FINE-TUNING SCRIPT
Ablasyon Amacı : LoRA KAPASİTESİ VE ADAPTÖR BÜYÜKLÜĞÜNÜN ETKİSİNİ İZOLE ETMEK
Tasarım        : Low-Res (200K px) + Yüksek LoRA Kapasitesi (r=64, alpha=128)
Hedef Donanım  : NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
Teknik         : 4-bit NF4 QLoRA + FlashAttention-2 + Gradient Checkpointing KAPALI
Veri Seti      : 15.042 (latin_pixel_cleaned/train) + 7.740 (ground_truth v1) = 22.782 Örnek
"""

import os
import sys

# ============================================================
# 0. CUDA & ALLOCATOR OPTİMİZASYONLARI (import torch öncesi!)
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import time
import glob
import math
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import jiwer
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# TorchAO Uyumsuzluk Yaması
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

# ============================================================
# 1. HİPERPARAMETRELER VE DİZİNLER
# ============================================================
MODEL_NAME      = "<PROJECT_ROOT>/models/Qwen3.5-4B"

# Eğitim veri seti kökü (split: train / val / test)
DATASET_ROOT    = "<PROJECT_ROOT>/data-set/latin_pixel_cleaned"

# Ek düz veri seti (7.740 eşleşme, ayrı train/val/test bölmesi yok)
EXTRA_DATASET   = "<PROJECT_ROOT>/data-set/ground_truth_line_based_dataset_qwen35_latin_v1"

# Çıktı dizinleri — ablasion_lowres_lora64 son ekiyle
OUTPUT_LORA_DIR   = "<PROJECT_ROOT>/models/qwen35_4b_latin_lora_v2_15042_7740_ablasion_lowres_lora64"
CHECKPOINTS_DIR   = "<PROJECT_ROOT>/models/qwen35_4b_latin_checkpoints_v2_15042_7740_ablasion_lowres_lora64"
MERGED_OUTPUT_DIR = "<PROJECT_ROOT>/models/qwen35_4b_latin_merged_v2_15042_7740_ablasion_lowres_lora64"

os.makedirs(OUTPUT_LORA_DIR,  exist_ok=True)
os.makedirs(CHECKPOINTS_DIR,  exist_ok=True)

PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)
MAX_SEQ_LENGTH = 384  # [ABLASYON 2]: Low-Res standardı (384)

# [ABLASYON 2]: Low-Res ayarları
MIN_PIXELS = 3136           # 3.136 px
MAX_PIXELS = 256 * 28 * 28  # 200.704 px (~200K)

# ============================================================
# 2. VERİ SETİ YÜKLEME
# ============================================================
def load_split(split_dir):
    """Bir dizindeki .png/.jpg + .txt eşleşmelerini yükler."""
    samples = []
    if not os.path.exists(split_dir):
        print(f"⚠️ Dizin bulunamadı: {split_dir}")
        return samples
    valid_exts = ('.png', '.jpg', '.jpeg')
    for fname in sorted(os.listdir(split_dir)):
        if fname.lower().endswith(valid_exts):
            stem = os.path.splitext(fname)[0]
            txt_path = os.path.join(split_dir, stem + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    gt_text = f.read().strip()
                if gt_text:
                    samples.append({
                        "image": os.path.join(split_dir, fname),
                        "text":  gt_text
                    })
    return samples


def load_extra_dataset(extra_dir, val_ratio=0.05, test_ratio=0.02, seed=42):
    """
    EXTRA_DATASET dizinini yükleyip rastgele train/val/test'e böler.
    Bölme oranları: %93 train, %5 val, %2 test
    """
    import random
    random.seed(seed)
    all_samples = load_split(extra_dir)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_test - n_val
    return (
        all_samples[:n_train],
        all_samples[n_train:n_train + n_val],
        all_samples[n_train + n_val:]
    )

# ============================================================
# 3. PYTORCH DATASET & DATA COLLATOR
# ============================================================
class OttomanHTRDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples   = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 32), (255, 255, 255))

        ground_truth = sample["text"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": PROMPT_TEXT}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ground_truth}]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

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

        return {
            "input_ids":         inputs["input_ids"].squeeze(0),
            "attention_mask":    inputs["attention_mask"].squeeze(0),
            "pixel_values":      pixel_values,
            "image_grid_thw":    image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }

class OttomanDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batch = {}
        pad_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        input_ids      = [f["input_ids"]      for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        batch["input_ids"]      = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        if features[0].get("mm_token_type_ids") is not None:
            mm_types = [f["mm_token_type_ids"] for f in features]
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm_types, batch_first=True, padding_value=0
            )

        pvs = [
            f["pixel_values"].squeeze(0) if f["pixel_values"].ndim == 3
            else f["pixel_values"]
            for f in features
        ]
        batch["pixel_values"] = torch.cat(pvs, dim=0)

        if features[0]["image_grid_thw"] is not None:
            thws = [
                f["image_grid_thw"].unsqueeze(0) if f["image_grid_thw"].ndim == 1
                else (
                    f["image_grid_thw"].squeeze(0) if f["image_grid_thw"].ndim == 3
                    else f["image_grid_thw"]
                )
                for f in features
            ]
            batch["image_grid_thw"] = torch.cat(thws, dim=0)

        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100

        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        if assistant_token_id is not None and assistant_token_id != self.processor.tokenizer.unk_token_id:
            for i, feature in enumerate(features):
                seq = feature["input_ids"]
                im_start_positions = (seq == assistant_token_id).nonzero(as_tuple=True)[0]
                if len(im_start_positions) > 0:
                    prompt_end_idx    = im_start_positions[-1].item()
                    actual_mask_limit = min(len(seq), prompt_end_idx + 3)
                    labels[i, :actual_mask_limit] = -100

        batch["labels"] = labels
        return batch

# ============================================================
# 4. DEĞERLENDİRME METRİKLERİ VE ÖZEL TRAINER
# ============================================================
# ============================================================
# 4. ÖZEL VRAM OPTİMİZE TRAINER
# ============================================================
class OttomanTrainer(Trainer):
    """
    Qwen3.5-4B için VRAM Optimize Edilmiş Trainer.
    compute_loss ve prediction_step içinde logits_to_keep argümanı kullanılarak
    devasa logit tensörünün (576 x 152.064) VRAM'de patlaması engellenir.
    Sadece etiketli (target) token'ların logit'leri tutulur.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        valid_mask        = (labels != -100)
        num_target_tokens = valid_mask.sum().item()
        num_target_tokens = max(1, num_target_tokens)

        logits_to_keep = min(inputs["input_ids"].shape[1], num_target_tokens + 1)

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        model_inputs["logits_to_keep"] = logits_to_keep

        outputs = model(**model_inputs)
        logits  = outputs.logits

        shift_logits  = logits[..., :-1, :].contiguous().float()
        target_labels = labels[..., -logits_to_keep:]
        shift_labels  = target_labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        with torch.no_grad():
            valid_mask        = (labels != -100)
            num_target_tokens = valid_mask.sum().item()
            num_target_tokens = max(1, num_target_tokens)

            logits_to_keep = min(inputs["input_ids"].shape[1], num_target_tokens + 1)
            model_inputs   = {k: v for k, v in inputs.items() if k != "labels"}
            model_inputs["logits_to_keep"] = logits_to_keep

            outputs = model(**model_inputs)
            logits  = outputs.logits

            shift_logits  = logits[..., :-1, :].contiguous().float()
            target_labels = labels[..., -logits_to_keep:]
            shift_labels  = target_labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            if prediction_loss_only:
                return (loss, None, None)

            preds = shift_logits.argmax(dim=-1)
            return (loss, preds, shift_labels)

# ============================================================
# 5. METRİK HESAPLAMA (CER / WER)
# ============================================================
def get_compute_metrics_fn(processor):
    def compute_metrics(eval_pred):
        pred_ids, labels = eval_pred
        pad_id = (
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        )
        all_cer, all_wer = [], []
        for pred_row, label_row in zip(pred_ids, labels):
            label_mask = label_row != -100
            if not label_mask.any():
                continue
            label_ids = label_row[label_mask]
            pred_ids_ = pred_row[label_mask]
            label_ids = label_ids[label_ids != pad_id]
            pred_ids_ = pred_ids_[pred_ids_ != pad_id]

            ref_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
            hyp_text = processor.tokenizer.decode(pred_ids_, skip_special_tokens=True).strip()

            if ref_text:
                all_cer.append(jiwer.cer(ref_text, hyp_text))
                all_wer.append(jiwer.wer(ref_text, hyp_text))

        return {
            "cer": round(float(np.mean(all_cer)), 4) if all_cer else 1.0,
            "wer": round(float(np.mean(all_wer)), 4) if all_wer else 1.0,
        }
    return compute_metrics

# ============================================================
# 5. ANA EĞİTİM VE MERGE FONKSİYONU
# ============================================================
def main():
    print("=" * 80)
    print("🎯 OSMANLICA LATİN TRANSKRİPSİYON ABLASYON 2 (LOW-RES + LoRA r=64)")
    print(f"📌 Model Tabanı         : {MODEL_NAME}")
    print(f"📌 Ana Veri Seti        : {DATASET_ROOT}")
    print(f"📌 Ek Veri Seti         : {EXTRA_DATASET}")
    print(f"📌 Görsel Çözünürlük    : [ABLASYON] LOW-RES min={MIN_PIXELS} px, max={MAX_PIXELS} px")
    print(f"📌 LoRA Parametreleri   : r=64, alpha=128 (all-linear)")
    print(f"📌 Donanım Optimizasyonu: 4-bit NF4 QLoRA + FlashAttention-2 + adamw_8bit (GC KAPALI)")
    print("-" * 80)

    # 1. Veri Setlerini Yükle
    train_samples = load_split(os.path.join(DATASET_ROOT, "train"))
    val_samples   = load_split(os.path.join(DATASET_ROOT, "val"))
    test_samples  = load_split(os.path.join(DATASET_ROOT, "test"))

    extra_train, extra_val, extra_test = load_extra_dataset(EXTRA_DATASET, seed=42)

    train_samples = train_samples + extra_train
    val_samples   = val_samples   + extra_val
    test_samples  = test_samples  + extra_test

    print(f"📊 Eğitim Örneği (Train) : {len(train_samples):,} satır "
          f"(15.042 orijinal + {len(extra_train):,} ekstra)")
    print(f"📊 Doğrulama Örneği (Val): {len(val_samples):,} satır")
    print(f"📊 Test Örneği (Test)    : {len(test_samples):,} satır")
    print(f"📊 TOPLAM EĞİTİM         : {len(train_samples):,} örnek")

    if len(train_samples) == 0:
        print("❌ HATA: Train veri setinde örnek bulunamadı!")
        sys.exit(1)

    # 2. AutoProcessor Yapılandırması
    print("\n⏳ AutoProcessor Yükleniyor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    # 3. 4-bit QLoRA Model Yükleme (FlashAttention-2 ile)
    print("⏳ Qwen3.5-4B 4-bit NF4 + FlashAttention-2 ile GPU'ya yükleniyor...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    # Model Hazırlığı (VRAM & Hız Optimizasyonu - Gradient Checkpointing KAPALI)
    model.enable_input_require_grads()
    model.config.use_cache = False
    print("🚀 Gradient Checkpointing KAPALI (Maksimum GPU Hızı), use_cache=False & bfloat16 korundu.")

    if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "attn_implementation"):
        print(f"👁️ Vision Tower Attention: {model.config.vision_config.attn_implementation}")

    # 4. LoRA Yapılandırması (r=64, alpha=128)
    print("⏳ LoRA Adaptörleri Ekleniyor (r=64, alpha=128, all-linear)...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        if ("lm_head" in name or "embed_tokens" in name) and "lora_" not in name:
            param.requires_grad = False

    print("✅ lm_head & embed_tokens donduruldu.")
    model.print_trainable_parameters()

    # 5. Dataset ve Collator
    train_dataset      = OttomanHTRDataset(train_samples, processor)
    val_dataset        = OttomanHTRDataset(val_samples,   processor)
    collator           = OttomanDataCollator(processor)
    compute_metrics_fn = get_compute_metrics_fn(processor)

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=3,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        tf32=True,
        optim="adamw_8bit",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none"
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=8)]

    # 7. Trainer Başlat
    trainer = OttomanTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks
    )

    # 8. Checkpoint Kontrolü
    resume_ckpt = None
    existing = sorted(
        glob.glob(os.path.join(CHECKPOINTS_DIR, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else 0
    )
    if existing:
        resume_ckpt = existing[-1]
        print(f"♻️ Mevcut checkpoint bulundu, devam ediliyor: {resume_ckpt}")
    else:
        print("🆕 Checkpoint bulunamadı, eğitim sıfırdan başlıyor.")

    # 9. Eğitimi Başlat
    print("\n" + "=" * 80)
    print("🔥 ABLASYON 2 EĞİTİMİ BAŞLADI!")
    print("=" * 80)
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    total_train_time = time.time() - start_time
    print(f"\n✅ Eğitim Tamamlandı! Toplam Süre: {total_train_time / 3600:.2f} saat.")

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\n🏆 En İyi Checkpoint: {best_ckpt}")

    # 10. LoRA Adaptörünü Kaydet
    print(f"\n💾 En İyi LoRA Adaptörü Kaydediliyor: {OUTPUT_LORA_DIR}")
    os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)
    import shutil
    if best_ckpt and os.path.exists(os.path.join(best_ckpt, "adapter_model.safetensors")):
        shutil.copy2(os.path.join(best_ckpt, "adapter_model.safetensors"), os.path.join(OUTPUT_LORA_DIR, "adapter_model.safetensors"))
        shutil.copy2(os.path.join(best_ckpt, "adapter_config.json"), os.path.join(OUTPUT_LORA_DIR, "adapter_config.json"))
    else:
        trainer.save_model(OUTPUT_LORA_DIR)
    processor.save_pretrained(OUTPUT_LORA_DIR)

    # 11. Otomatik Model Merge
    print("\n" + "=" * 80)
    print("🔄 NİHAİ MODEL MERGE İŞLEMİ BAŞLATILIYOR (LoRA + Base Model -> BF16)...")
    print("=" * 80)
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("⏳ Temel model bfloat16 olarak RAM/GPU'ya yükleniyor...")
    base_model_reload = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"⏳ LoRA adaptörü yükleniyor ve ağırlıklar birleştiriliyor...")
    merged_model = PeftModel.from_pretrained(base_model_reload, OUTPUT_LORA_DIR)
    merged_model = merged_model.merge_and_unload()

    print(f"💾 Birleştirilmiş Standalone Model Kaydediliyor: {MERGED_OUTPUT_DIR}")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    processor.save_pretrained(MERGED_OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"🎉 TÜM SÜREÇ BAŞARIYLA TAMAMLANDI!")
    print(f"   ├─ LoRA Adaptörü      : {OUTPUT_LORA_DIR}")
    print(f"   └─ Birleştirilmiş Model: {MERGED_OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
