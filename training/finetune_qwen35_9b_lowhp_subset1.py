#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
QWEN3.5-9B LATİN FINE-TUNING (MODEL V1 - GOOGLE COLAB / L4 GPU)
==================================================================================
Model       : Qwen/Qwen3.5-9B
Veri Kümesi : 15.166 Eğitim | 843 Doğrulama | 820 Test (Toplam ~16.829 satır)
Metodoloji  : 4-bit NF4 QLoRA (r=32, alpha=64, all-linear)
Eğitim Logu : 2844 Adım | 3 Epoch | Eval CER: %21.42 | Eval WER: %41.88
==================================================================================
"""

import os
import sys
import glob
import torch
import numpy as np
import shutil
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from google.colab import drive

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Google Drive Bağlama ---
drive.mount('/content/drive', force_remount=False)

# ============================================================
# TORCHAO UYUMSUZLUK HOTFIX
# ============================================================
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    print("✅ PEFT - torchao uyumsuzluk yaması uygulandı.")
except ImportError:
    pass

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)

try:
    from jiwer import cer, wer
    JIWER_AVAILABLE = True
    print("✅ jiwer yüklü — CER/WER metrikleri aktif.")
except ImportError:
    JIWER_AVAILABLE = False
    print("⚠️ jiwer bulunamadı → eval_loss kullanılacak.")

# ============================================================
# 1. YAPILANDIRMA VE VERİ SETİ YOLLARI
# ============================================================
MODEL_NAME   = "Qwen/Qwen3.5-9B"
DATASET_ROOT = "/content/work/paired_dataset"  # Doğrudan Colab diskindeki eşleşmiş veri

# --- YENİ: Checkpoint'ler artık DOĞRUDAN Drive'a yazılıyor ---
# Böylece oturum kopsa bile en son kaydedilen checkpoint kaybolmaz.
OUTPUT_DIR = "<DRIVE_PATH> 3 - Ottoman Turkish/Modeller/Qwen3.5-9B fine-tunning - latin"

MAX_SEQ_LENGTH = 384

PROMPT_TEXT = (
    "Aşağıdaki görselde yer alan arap alfabeli Osmanlı Türkçesi el yazısı "
    "metnin birebir Latin harfli transkripsiyonunu yazınız:"
)

# ============================================================
# 2. VERİ SETİ YÜKLEME
# ============================================================
def load_split(split_dir):
    samples = []
    if not os.path.exists(split_dir):
        return samples
    for fname in os.listdir(split_dir):
        if not fname.lower().endswith(".png"):
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

print("🔍 Veri seti bölümleri `/content/work/paired_dataset` altından yükleniyor...")
train_samples = load_split(os.path.join(DATASET_ROOT, "train"))
val_samples   = load_split(os.path.join(DATASET_ROOT, "val"))
test_samples  = load_split(os.path.join(DATASET_ROOT, "test"))

print(f"   ├─ Eğitim   (Train): {len(train_samples)} satır")
print(f"   ├─ Doğrulama (Val) : {len(val_samples)} satır")
print(f"   └─ Test            : {len(test_samples)} satır")

if len(train_samples) == 0:
    raise RuntimeError(f"❌ Train setinde örnek bulunamadı! Yol kontrolü yapın: {DATASET_ROOT}/train")

# ============================================================
# 3. PyTorch Dataset
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
            messages, tokenize=False, add_generation_prompt=False,
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

# ============================================================
# 4. DATA COLLATOR
# ============================================================
def data_collator(features):
    batch  = {}
    pad_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id
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
        mm_token_type_ids = [f["mm_token_type_ids"] for f in features]
        batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
            mm_token_type_ids, batch_first=True, padding_value=0
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

    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    if assistant_token_id is not None and assistant_token_id != processor.tokenizer.unk_token_id:
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
# 5. METRİK İŞLEMLERİ (CER / WER)
# ============================================================
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred
    shifted_preds  = pred_ids[:, 1:]
    shifted_labels = labels[:, 1:]

    pad_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id
    )

    all_cer, all_wer = [], []
    for pred_row, label_row in zip(shifted_preds, shifted_labels):
        label_mask = label_row != -100
        if not label_mask.any():
            continue
        label_ids = label_row[label_mask]
        pred_ids_ = pred_row[label_mask]
        label_ids = label_ids[label_ids != pad_id]
        pred_ids_ = pred_ids_[pred_ids_ != pad_id]

        ref_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        hyp_text = processor.tokenizer.decode(pred_ids_, skip_special_tokens=True).strip()
        if not ref_text:
            continue

        if JIWER_AVAILABLE:
            try:
                all_cer.append(cer(ref_text, hyp_text))
                all_wer.append(wer(ref_text, hyp_text))
            except Exception:
                pass
        else:
            import difflib
            sim = difflib.SequenceMatcher(None, ref_text, hyp_text).ratio()
            all_cer.append(1.0 - sim)

    metrics = {}
    if all_cer:
        metrics["cer"] = float(np.mean(all_cer))
    if all_wer:
        metrics["wer"] = float(np.mean(all_wer))
    return metrics

# ============================================================
# 6. MODEL VE İŞLEMCİ YÜKLEME (4-bit QLoRA)
# ============================================================
print("\n🚀 Model ve İşlemci Yükleniyor (Qwen3.5-9B)...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

try:
    min_pixels = 3136
    max_pixels = 256 * 28 * 28
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    print(f"✅ Görsel İşlemci limitleri: min={min_pixels}, max={max_pixels}")
except Exception as e:
    print(f"⚠️ Görsel işlemci limitleri ayarlanamadı ({e}), varsayılanlar kullanılacak.")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa"  # flash-attn kurulumu gerektirmez, PyTorch'ta hazır gelir
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print("✅ L4 (24GB VRAM) için Gradient Checkpointing AKTİF.")

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print("✅ LoRA eğitime hazır.")
model.print_trainable_parameters()

# ============================================================
# 7. EĞİTİM VERİ SETLERİ VE YAPILANDIRMA
# ============================================================
train_dataset = OttomanHTRDataset(train_samples, processor)
val_dataset   = OttomanHTRDataset(val_samples,   processor)

os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_METRIC = "eval_cer" if JIWER_AVAILABLE else "eval_loss"
print(f"📊 Erken durdurma metriği: {BEST_METRIC}")
print(f"💾 Checkpoint kayıt klasörü (Drive): {OUTPUT_DIR}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,        # efektif batch: 2x8=16

    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,                    # NOT: dokunulmadı (istek üzerine)

    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,                       # DEĞİŞTİ: 250 -> 500
    save_steps=500,                       # DEĞİŞTİ: 250 -> 500 (Drive'a her 500 adımda kayıt)
    num_train_epochs=3,

    fp16=False,
    bf16=True,
    tf32=True,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,

    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model=BEST_METRIC,
    greater_is_better=False,

    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    report_to="none"
)

callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=callbacks
)

# ============================================================
# 8. VAR OLAN CHECKPOINT KONTROLÜ (kaldığı yerden devam)
# ============================================================
resume_checkpoint = None
existing_checkpoints = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
    key=lambda p: int(p.split("-")[-1])
)
if existing_checkpoints:
    resume_checkpoint = existing_checkpoints[-1]
    print(f"♻️  Mevcut checkpoint bulundu, buradan devam edilecek: {resume_checkpoint}")
else:
    print("🆕 Mevcut checkpoint bulunamadı, eğitim sıfırdan başlıyor.")

# ============================================================
# 9. EĞİTİMİ BAŞLAT
# ============================================================
print("\n🎯 Osmanlıca OCR Fine-Tuning (Qwen3.5-9B) Başlatılıyor...")
print(f"   ├─ Toplam Eğitim Örneği: {len(train_samples)}")
print(f"   ├─ Batch size: 2 | Grad accum: 8 | Efektif BS: 16")
print(f"   ├─ Attention Implementation: SDPA (PyTorch Native)")
print(f"   ├─ Checkpoint: her 500 adımda Drive'a kaydediliyor")
print(f"   └─ Gradient checkpointing: AKTİF | Quantization: 4-bit NF4 QLoRA\n")

trainer.train(resume_from_checkpoint=resume_checkpoint)

# ============================================================
# 10. NİHAİ ADAPTÖRÜ KAYDET
# ============================================================
# OUTPUT_DIR zaten Drive üzerinde olduğu için ayrıca kopyalamaya gerek yok.
final_path = os.path.join(OUTPUT_DIR, "final_ottoman_qwen35_adapter")
print(f"💾 Nihai adaptör kaydediliyor (doğrudan Drive'a): {final_path}")
trainer.save_model(final_path)
processor.save_pretrained(final_path)

print("\n🎉 Tüm eğitim adımları başarıyla tamamlandı! Tüm checkpoint'ler ve nihai model Drive'da.")


# ==============================================================================
# EĞİTİM ÇIKTI VE SONUÇ GÜNLÜĞÜ (GOOGLE COLAB L4 ÇIKTISI)
# ==============================================================================
"""
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
✅ PEFT - torchao uyumsuzluk yaması uygulandı.
✅ jiwer yüklü — CER/WER metrikleri aktif.
🔍 Veri seti bölümleri `/content/work/paired_dataset` altından yükleniyor...
   ├─ Eğitim   (Train): 15166 satır
   ├─ Doğrulama (Val) : 843 satır
   └─ Test            : 820 satır

🚀 Model ve İşlemci Yükleniyor (Qwen3.5-9B)...
[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in /usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py.
[ERROR] `max_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in /usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py.
✅ Görsel İşlemci limitleri: min=3136, max=200704
Download complete: : 
  0.00B            
Reconstruction complete: 
  0.00B /  0.00B            
Fetching 4 files: 100%
 4/4 [00:00<00:00, 298.40it/s]
Loading weights: 100%
 760/760 [00:05<00:00, 617.95it/s]
✅ L4 (24GB VRAM) için Gradient Checkpointing AKTİF.
✅ LoRA eğitime hazır.
trainable params: 102,530,048 || all params: 9,512,343,792 || trainable%: 1.0779
📊 Erken durdurma metriği: eval_cer
💾 Checkpoint kayıt klasörü (Drive): <DRIVE_PATH> 3 - Ottoman Turkish/Modeller/Qwen3.5-9B fine-tunning - latin
🆕 Mevcut checkpoint bulunamadı, eğitim sıfırdan başlıyor.

🎯 Osmanlıca OCR Fine-Tuning (Qwen3.5-9B) Başlatılıyor...
   ├─ Toplam Eğitim Örneği: 15166
   ├─ Batch size: 2 | Grad accum: 8 | Efektif BS: 16
   ├─ Attention Implementation: SDPA (PyTorch Native)
   ├─ Checkpoint: her 500 adımda Drive'a kaydediliyor
   └─ Gradient checkpointing: AKTİF | Quantization: 4-bit NF4 QLoRA

/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:944: UserWarning: inner dimension (4304) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
  warn(
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
 [1787/2844 11:52:49 < 7:02:06, 0.04 it/s, Epoch 1.88/3]
Step	Training Loss	Validation Loss	Cer	Wer
500	1.110035	1.069532	0.283573	0.568444
1000	0.768191	0.853066	0.245953	0.493541
1500	0.682146	0.746075	0.230032	0.456292
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:944: UserWarning: inner dimension (4304) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
  warn(
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:944: UserWarning: inner dimension (4304) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
  warn(
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:944: UserWarning: inner dimension (4304) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
  warn(
 [2844/2844 18:59:53, Epoch 3/3]
Step	Training Loss	Validation Loss	Cer	Wer
500	1.110035	1.069532	0.283573	0.568444
1000	0.768191	0.853066	0.245953	0.493541
1500	0.682146	0.746075	0.230032	0.456292
2000	0.500185	0.705239	0.219548	0.431993
2500	0.501025	0.682410	0.214211	0.418849
2844	0.463303	0.680545	0.215580	0.421917
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:944: UserWarning: inner dimension (4304) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
  warn(
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:944: UserWarning: inner dimension (4304) is not aligned for fast kernel with blocksize=64, falling back to slower implementation.
  warn(
💾 Nihai adaptör kaydediliyor (doğrudan Drive'a): <DRIVE_PATH> 3 - Ottoman Turkish/Modeller/Qwen3.5-9B fine-tunning - latin/final_ottoman_qwen35_adapter

🎉 Tüm eğitim adımları başarıyla tamamlandı! Tüm checkpoint'ler ve nihai model Drive'da.
"""
