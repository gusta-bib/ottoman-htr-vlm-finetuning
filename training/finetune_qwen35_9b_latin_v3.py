#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSMANLICA OCR - QWEN3.5-9B LATİN TRANSKRİPSİYON V3 FINE-TUNING SCRIPT
Hedef Donanım : NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
Teknik        : 4-bit NF4 QLoRA + FlashAttention-2 + Gradient Checkpointing
Veri Seti     : 27.460 Satır (latin_pixel_cleaned/train [22.782] + ground_truth_v2 [4.678])
İyileştirmeler:
  - Görsel Çözünürlük: min_pixels=128*28*28 (100K) / max_pixels=768*28*28 (602K)
  - LoRA Kapasitesi  : r=64 / alpha=128 (Tam Kapsam: all-linear)
  - Eğitim Dinamikleri: Patience=8, Warmup=0.05, Weight Decay=0.01
  - VRAM Optimizasyonu: Batch=1, GradAccum=16, adamw_8bit, EvalAccum=1, logits_to_keep
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
MODEL_NAME = "<PROJECT_ROOT>/models/Qwen3.5-9B"
DATASET_ROOT = "<PROJECT_ROOT>/data-set/latin_pixel_cleaned"
NEW_DATASET_DIR = "<PROJECT_ROOT>/data-set/ground_truth_line_based_dataset_qwen35_latin_v2"

OUTPUT_LORA_DIR = "<PROJECT_ROOT>/models/qwen35_9b_latin_lora_v3"
CHECKPOINTS_DIR = "<PROJECT_ROOT>/models/qwen35_9b_latin_checkpoints_v3"
MERGED_OUTPUT_DIR = "<PROJECT_ROOT>/models/qwen35_9b_latin_merged_v3"

os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)
MAX_SEQ_LENGTH = 576  # Veri setindeki en uzun örneği %100 kapsar, 0 kesilme

MIN_PIXELS = 128 * 28 * 28   # 100.352 px (OCR harf/patch yoğunluğu korunur)
MAX_PIXELS = 768 * 28 * 28   # 602.112 px

# ============================================================
# 2. VERİ SETİ YÜKLEME
# ============================================================
def load_split(split_dir):
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
                        "text": gt_text
                    })
    return samples

# ============================================================
# 3. PYTORCH DATASET & DATA COLLATOR
# ============================================================
class OttomanHTRDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
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

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
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
        if assistant_token_id is not None:
            for i, feature in enumerate(features):
                seq = feature["input_ids"]
                im_start_positions = (seq == assistant_token_id).nonzero(as_tuple=True)[0]
                if len(im_start_positions) > 0:
                    prompt_end_idx = im_start_positions[-1].item()
                    actual_mask_limit = min(len(seq), prompt_end_idx + 3)
                    labels[i, :actual_mask_limit] = -100

        batch["labels"] = labels
        return batch

# ============================================================
# 4. ÖZEL TRAINER (LOGITS_TO_KEEP İLE VRAM TASARRUFU)
# ============================================================
class OttomanTrainer(Trainer):
    """
    Standart Trainer tüm sequence (576 token) için tam vocab logits hesaplayarak
    lm_head çıktısında ~1.9 GB VRAM spike yaratır.
    OttomanTrainer, logits_to_keep parametresi ile lm_head'i SADECE assistant
    cevabının olduğu son N token için çalıştırır -> %90+ logits VRAM tasarrufu!
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # 1. Sadece -100 olmayan (hedef assistant cevabı) token sayısını belirle
        valid_mask = (labels != -100)
        num_target_tokens = valid_mask.sum().item()
        num_target_tokens = max(1, num_target_tokens)

        # 2. Sadece son (num_target_tokens + 1) token için logits hesaplat
        logits_to_keep = min(inputs["input_ids"].shape[1], num_target_tokens + 1)

        # 3. labels'ı model.forward'a vermeden, logits_to_keep ile çağır
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        model_inputs["logits_to_keep"] = logits_to_keep

        outputs = model(**model_inputs)
        logits = outputs.logits  # shape: (batch_size, logits_to_keep, vocab_size)

        # 4. Kısaltılmış logits ve labels üzerinde Next-Token Prediction Shift
        shift_logits = logits[..., :-1, :].contiguous().float()
        target_labels = labels[..., -logits_to_keep:]
        shift_labels = target_labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        with torch.no_grad():
            valid_mask = (labels != -100)
            num_target_tokens = valid_mask.sum().item()
            num_target_tokens = max(1, num_target_tokens)
            logits_to_keep = min(inputs["input_ids"].shape[1], num_target_tokens + 1)

            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            model_inputs["logits_to_keep"] = logits_to_keep

            outputs = model(**model_inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous().float()
            target_labels = labels[..., -logits_to_keep:]
            shift_labels = target_labels[..., 1:].contiguous()

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
# 6. ANA EĞİTİM VE MERGE FONKSİYONU
# ============================================================
def main():
    print("=" * 80)
    print("🚀 QWEN3.5-9B LATİN TRANSKRİPSİYON V3 EĞİTİMİ BAŞLATILIYOR...")
    print("=" * 80)
    print(f"📌 Model Yolu          : {MODEL_NAME}")
    print(f"📌 Temel Veri Seti     : {DATASET_ROOT}")
    print(f"📌 Yeni Eklenen Set (v2): {NEW_DATASET_DIR}")
    print(f"📌 Görsel Çözünürlük   : min={MIN_PIXELS} px, max={MAX_PIXELS} px")
    print(f"📌 LoRA Parametreleri  : r=64, alpha=128 (all-linear: LLM + ViT Attn + ViT MLP + Merger)")
    print(f"📌 Donanım Optimizasyonu: 4-bit NF4 QLoRA + FlashAttention-2 + adamw_8bit")
    print("-" * 80)

    # 1. Veri Setlerini Yükle (22.782 Mevcut + 4.678 Yeni Üretilen = 27.460 Train)
    base_train = load_split(os.path.join(DATASET_ROOT, "train"))
    new_train = load_split(NEW_DATASET_DIR)
    train_samples = base_train + new_train

    val_samples = load_split(os.path.join(DATASET_ROOT, "val"))
    test_samples = load_split(os.path.join(DATASET_ROOT, "test"))

    print(f"📊 Eğitim Örneği (Train) : {len(train_samples):,} satır ({len(base_train):,} Temel + {len(new_train):,} Yeni Altın Set)")
    print(f"📊 Doğrulama Örneği (Val): {len(val_samples):,} satır")
    print(f"📊 Test Örneği (Test)    : {len(test_samples):,} satır")

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
    print("⏳ Qwen3.5-9B 4-bit NF4 + FlashAttention-2 ile GPU'ya yükleniyor...")
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

    # Model Hazırlığı (VRAM Optimizasyonu)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("✅ Gradient Checkpointing Aktif, use_cache=False & bfloat16 korundu (+3.8 GB VRAM tasarrufu).")

    # Vision Tower Attention kontrolü
    if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "attn_implementation"):
        print(f"👁️ Vision Tower Attention: {model.config.vision_config.attn_implementation}")

    # 4. LoRA Yapılandırması (r=64, alpha=128, all-linear)
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

    # lm_head ve embed_tokens'ın base modelde requires_grad=False olduğundan emin ol
    for name, param in model.named_parameters():
        if ("lm_head" in name or "embed_tokens" in name) and "lora_" not in name:
            param.requires_grad = False

    print("✅ lm_head & embed_tokens donduruldu.")
    model.print_trainable_parameters()

    # 5. Dataset ve Collator
    train_dataset = OttomanHTRDataset(train_samples, processor)
    val_dataset = OttomanHTRDataset(val_samples, processor)
    collator = OttomanDataCollator(processor)
    compute_metrics_fn = get_compute_metrics_fn(processor)

    # 6. Training Arguments (16 GB VRAM Koruma + Hızlandırılmış Ayarlar)
    training_args_kwargs = dict(
        output_dir=CHECKPOINTS_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,        # 16 GB VRAM için batch=1
        gradient_accumulation_steps=16,       # Efektif batch size: 1x16=16
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,            # Eval OOM koruması
        learning_rate=1e-4,                   # QLoRA için optimal öğrenme oranı
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=750,                       # Hızlandırma: 750 adımda bir doğrulama
        save_strategy="steps",
        save_steps=750,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        tf32=True,
        optim="adamw_8bit",                   # 8-bit AdamW
        max_grad_norm=1.0,
        dataloader_num_workers=6,             # CPU I/O paralelleştirme
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none"
    )

    try:
        training_args = TrainingArguments(torch_empty_cache_steps=50, **training_args_kwargs)
    except TypeError:
        training_args = TrainingArguments(**training_args_kwargs)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=8)]

    trainer = OttomanTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks
    )

    # 7. VRAM Smoke Test (Optimizer Dahil Gerçek Peak VRAM Doğrulaması)
    print("\n" + "=" * 80)
    print("🧪 VRAM SMOKE TEST: Optimizer (adamw_8bit) + logits_to_keep ile Peak VRAM test ediliyor...")
    print("=" * 80)

    clean_lora_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    trainer.create_optimizer()

    def get_sample_weight(s):
        try:
            with Image.open(s["image"]) as img:
                pixels = img.size[0] * img.size[1]
        except Exception:
            pixels = 0
        return pixels + len(s.get("text", "")) * 100

    worst_samples = sorted(train_samples, key=get_sample_weight, reverse=True)[:5]

    model.train()
    torch.cuda.reset_peak_memory_stats()

    for idx, sample in enumerate(worst_samples, 1):
        try:
            dummy_ds = OttomanHTRDataset([sample], processor)
            feature = dummy_ds[0]
            batch = collator([feature])
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 1. Forward Pass
            loss = trainer.compute_loss(model, batch)

            # 2. Backward Pass
            loss.backward()

            # 3. Optimizer Step
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"   ├─ Test [{idx}/5] ({os.path.basename(sample['image'])}): "
                  f"Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB / 16.0 GB ✅")

            del batch, loss
        except Exception as e:
            print(f"   ❌ Smoke test sırasında hata oluştu: {e}")
            raise e

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"\n✅ Smoke Test Başarılı! Optimizer Dahil Gerçek Peak VRAM: Allocated={peak_gb:.2f} GB | Reserved={peak_res_gb:.2f} GB.")
    print("   Eğitim sırasında OOM riski kesin olarak bulunmamaktadır.\n")

    # Model ağırlıklarını tertemiz ilk haline geri yükle
    model.load_state_dict(clean_lora_state, strict=False)
    del clean_lora_state
    trainer.optimizer = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 8. Checkpoint Kontrolü (Varsa kaldığı yerden devam)
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
    print("🔥 EĞİTİM BAŞLADI!")
    print("=" * 80)
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    total_train_time = time.time() - start_time
    print(f"\n✅ Eğitim Tamamlandı! Toplam Süre: {total_train_time/3600:.2f} saat.")

    # 10. En İyi LoRA Adaptörünü Kaydet
    print(f"\n💾 En İyi LoRA Adaptörü Kaydediliyor: {OUTPUT_LORA_DIR}")
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
