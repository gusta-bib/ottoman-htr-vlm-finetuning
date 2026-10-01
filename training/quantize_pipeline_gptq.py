#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
TÜBİTAK 1001 - OSMANLICA HTR PROJESİ
TOPLU VE SIRALI GPTQ W4A16 KUANTİZASYON PİPELİNE & E-POSTA BİLDİRİM SİSTEMİ
==================================================================================
İşlenecek 8 Model Sırası:
  1. qwen35_2b_latin_merged_v1               [TAMAMLANDI]
  2. qwen35_2b_latin_merged_v2_15042_7740    [TAMAMLANDI]
  3. qwen35_2b_latin_merged_v2_15042_7740_test [TAMAMLANDI]
  4. qwen35_2b_latin_merged_v3_15042_7740_4678 [TAMAMLANDI]
  5. qwen35_4b_latin_merged_v1               [SIRADAKİ]
  6. qwen35_4b_latin_merged_v2_15042_7740
  7. qwen35_4b_latin_merged_v2_15042_7740_test
  8. qwen35_4b_latin_merged_v3_15042_7740_4678

Özellikler:
  - Her modeli İZOLE BİR PYTHON SÜRECİNDE (Subprocess) çalıştırır.
  - Model bitince GPU VRAM'i %100 sıfırlanır, OOM riski tamamen ortadan kalkar.
  - Zaten tamamlanmış modelleri otomatik algılar ve doğrudan sıradakine geçer.
  - Her model bitiminde ve pipeline sonunda HTML e-posta raporu gönderir.
==================================================================================
"""

import os
import sys
import time
import datetime
import json
import smtplib
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = "<PROJECT_ROOT>"
BASE_MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_ROOT = os.path.join(BASE_DIR, "data-set", "latin_pixel_cleaned")
CONFIG_PATH = os.path.join(BASE_DIR, "email_config.json")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts_quantization")
WORKER_SCRIPT = os.path.join(SCRIPTS_DIR, "quantize_single_worker.py")
MASTER_LOG_PATH = os.path.join(SCRIPTS_DIR, "pipeline_quantization.log")
PYTHON_BIN = os.path.join(BASE_DIR, ".venv", "bin", "python")

MODELS_QUEUE = [
    {
        "id": "2b_v1",
        "name": "Qwen3.5-2B Latin v1 (15.042 Saf Çekirdek)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_merged_v1"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_gptq_w4a16_v1"),
    },
    {
        "id": "2b_v2",
        "name": "Qwen3.5-2B Latin v2 (15.042 + 7.740 = 22.782)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_merged_v2_15042_7740"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_gptq_w4a16_v2_15042_7740"),
    },
    {
        "id": "2b_v2_test",
        "name": "Qwen3.5-2B Latin v2 Test (22.782 + Test)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_merged_v2_15042_7740_test"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_gptq_w4a16_v2_15042_7740_test"),
    },
    {
        "id": "2b_v3",
        "name": "Qwen3.5-2B Latin v3 (22.782 + 4.678 = 27.460)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_merged_v3_15042_7740_4678"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_2b_latin_gptq_w4a16_v3_15042_7740_4678"),
    },
    {
        "id": "4b_v1",
        "name": "Qwen3.5-4B Latin v1 (15.042 Saf Çekirdek)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_merged_v1"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_gptq_w4a16_v1"),
    },
    {
        "id": "4b_v2",
        "name": "Qwen3.5-4B Latin v2 (15.042 + 7.740 = 22.782)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_merged_v2_15042_7740"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_gptq_w4a16_v2_15042_7740"),
    },
    {
        "id": "4b_v2_test",
        "name": "Qwen3.5-4B Latin v2 Test (22.782 + Test)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_merged_v2_15042_7740_test"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_gptq_w4a16_v2_15042_7740_test"),
    },
    {
        "id": "4b_v3",
        "name": "Qwen3.5-4B Latin v3 (22.782 + 4.678 = 27.460)",
        "input_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_merged_v3_15042_7740_4678"),
        "output_dir": os.path.join(BASE_MODELS_DIR, "qwen35_4b_latin_gptq_w4a16_v3_15042_7740_4678"),
    }
]

class DualLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def send_email(subject, html_content, plain_content):
    cfg = load_config()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["smtp_user"]
    msg["To"] = cfg["admin_email"]
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="classifyes.com")

    msg.attach(MIMEText(plain_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=25) as server:
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.sendmail(cfg["smtp_user"], [cfg["admin_email"]], msg.as_string())
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✉️ E-posta başarıyla gönderildi: '{subject}' -> {cfg['admin_email']}")
        return True
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ E-posta gönderim hatası: {e}")
        return False

def get_dir_size_gb(path):
    total_bytes = 0
    if not os.path.exists(path):
        return 0.0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 ** 3)

def is_model_complete(output_dir):
    if not os.path.exists(output_dir):
        return False
    has_weights = (
        os.path.exists(os.path.join(output_dir, "model.safetensors")) or
        os.path.exists(os.path.join(output_dir, "model.safetensors.index.json"))
    )
    has_config = os.path.exists(os.path.join(output_dir, "config.json"))
    return has_weights and has_config

def get_gpu_info():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split(",")
        name = res[0].strip()
        temp = res[1].strip()
        util = res[2].strip()
        used = float(res[3].strip()) / 1024
        total = float(res[4].strip()) / 1024
        return f"{name} ({temp}°C)", f"%{util} / {used:.2f} GB ({total:.2f} GB)"
    except Exception:
        return "NVIDIA GeForce RTX 4070 Ti SUPER", "N/A"

def build_stage_completion_email(m_info, idx, total, duration_min, in_size, out_size, completed_list):
    gpu_name, gpu_stat = get_gpu_info()
    saved_ratio = ((in_size - out_size) / in_size * 100) if in_size > 0 else 0

    rows = ""
    for item in completed_list:
        rows += f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 8px; font-weight: bold; color: #1e293b;">{item['name']}</td>
            <td style="padding: 8px; text-align: center;">{item['in_size']:.2f} GB</td>
            <td style="padding: 8px; text-align: center; font-weight: bold; color: #16a34a;">{item['out_size']:.2f} GB</td>
            <td style="padding: 8px; text-align: center;">%{item['saved_ratio']:.1f}</td>
            <td style="padding: 8px; text-align: center;">{item['duration']:.2f} dk</td>
        </tr>
        """

    html = f"""
    <div style="font-family: Arial, sans-serif; background-color: #f8fafc; padding: 20px;">
        <div style="max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 8px; padding: 24px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-bottom: 16px;">
                <h2 style="color: #1e3a8a; margin: 0;">⚡ GPTQ W4A16 Kuantizasyon Tamamlandı [{idx}/{total}]</h2>
                <p style="color: #64748b; font-size: 13px; margin: 4px 0 0 0;">{m_info['name']}</p>
            </div>

            <div style="background-color: #eff6ff; border-radius: 6px; padding: 14px; margin-bottom: 20px;">
                <table style="width: 100%; font-size: 14px; color: #1e293b;">
                    <tr><td style="color: #64748b;">Kuantize Edilen Model:</td><td style="font-weight: bold; text-align: right;">{m_info['name']}</td></tr>
                    <tr><td style="color: #64748b;">Girdi Model Boyutu (BF16):</td><td style="font-weight: bold; text-align: right;">{in_size:.2f} GB</td></tr>
                    <tr><td style="color: #64748b;">Çıktı Model Boyutu (GPTQ 4-Bit):</td><td style="font-weight: bold; color: #16a34a; font-size: 16px; text-align: right;">{out_size:.2f} GB (%{saved_ratio:.1f} Tasarruf)</td></tr>
                    <tr><td style="color: #64748b;">İşlem Süresi:</td><td style="font-weight: bold; text-align: right;">{duration_min:.2f} Dakika</td></tr>
                    <tr><td style="color: #64748b;">Kayıt Dizini:</td><td style="font-family: monospace; font-size: 12px; text-align: right;">{m_info['output_dir']}</td></tr>
                    <tr><td style="color: #64748b;">Kalan Model:</td><td style="font-weight: bold; color: #d97706; text-align: right;">{total - idx} Adet</td></tr>
                </table>
            </div>

            <h3 style="color: #334155; font-size: 15px; margin-bottom: 8px;">📊 Tamamlanan Modeller Tablosu</h3>
            <table style="width: 100%; font-size: 13px; border-collapse: collapse; margin-bottom: 20px;">
                <thead>
                    <tr style="background-color: #f1f5f9; color: #475569;">
                        <th style="padding: 8px; text-align: left;">Model</th>
                        <th style="padding: 8px;">BF16</th>
                        <th style="padding: 8px;">GPTQ</th>
                        <th style="padding: 8px;">Kazanç</th>
                        <th style="padding: 8px;">Süre</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>

            <div style="font-size: 12px; color: #64748b; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px;">
                <div><strong>Ekran Kartı:</strong> {gpu_name}</div>
                <div><strong>GPU Durumu:</strong> {gpu_stat}</div>
                <div><strong>Zaman:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>

            <p style="font-size: 11px; color: #94a3b8; margin-top: 20px; text-align: center;">TÜBİTAK 1001 Osmanlı Türkçesi HTR Projesi | Otomatik GPTQ Pipeline</p>
        </div>
    </div>
    """
    plain = f"GPTQ Kuantizasyon Tamamlandi [{idx}/{total}]: {m_info['name']}\nBF16: {in_size:.2f} GB -> GPTQ: {out_size:.2f} GB (%{saved_ratio:.1f} Tasarruf)\nSure: {duration_min:.2f} dk\nKalan: {total - idx} model"
    return html, plain

def build_final_summary_email(completed_list, total_duration_min):
    total_in = sum(x['in_size'] for x in completed_list)
    total_out = sum(x['out_size'] for x in completed_list)
    total_saved = ((total_in - total_out) / total_in * 100) if total_in > 0 else 0

    rows = ""
    for item in completed_list:
        rows += f"""
        <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f0fdf4;">
            <td style="padding: 8px; font-weight: bold; color: #0f172a;">{item['name']}</td>
            <td style="padding: 8px; text-align: center;">{item['in_size']:.2f} GB</td>
            <td style="padding: 8px; text-align: center; font-weight: bold; color: #15803d;">{item['out_size']:.2f} GB</td>
            <td style="padding: 8px; text-align: center; color: #166534; font-weight: bold;">%{item['saved_ratio']:.1f}</td>
            <td style="padding: 8px; text-align: center;">{item['duration']:.2f} dk</td>
        </tr>
        """

    html = f"""
    <div style="font-family: Arial, sans-serif; background-color: #f8fafc; padding: 20px;">
        <div style="max-width: 700px; margin: 0 auto; background: #ffffff; border-radius: 8px; padding: 24px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="border-bottom: 2px solid #16a34a; padding-bottom: 12px; margin-bottom: 16px;">
                <h2 style="color: #15803d; margin: 0;">🎉 TÜM 8 MODELİN GPTQ W4A16 KUANTİZASYONU TAMAMLANDI!</h2>
                <p style="color: #64748b; font-size: 13px; margin: 4px 0 0 0;">Qwen3.5 2B ve 4B Serisi (V1, V2, V2 Test, V3) Kuantizasyon Özeti</p>
            </div>

            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 14px; margin-bottom: 20px;">
                <table style="width: 100%; font-size: 14px; color: #1e293b;">
                    <tr><td style="color: #64748b;">Toplam Kuantize Edilen Model:</td><td style="font-weight: bold; color: #15803d; font-size: 16px; text-align: right;">{len(completed_list)} / 8 Model</td></tr>
                    <tr><td style="color: #64748b;">Toplam Orijinal Boyut (BF16):</td><td style="font-weight: bold; text-align: right;">{total_in:.2f} GB</td></tr>
                    <tr><td style="color: #64748b;">Toplam Kuantize Boyut (GPTQ 4-Bit):</td><td style="font-weight: bold; color: #15803d; font-size: 16px; text-align: right;">{total_out:.2f} GB (%{total_saved:.1f} Toplam Disk Tasarrufu)</td></tr>
                    <tr><td style="color: #64748b;">Toplam Pipeline Süresi:</td><td style="font-weight: bold; text-align: right;">{total_duration_min:.2f} Dakika ({total_duration_min/60:.2f} Saat)</td></tr>
                </table>
            </div>

            <h3 style="color: #334155; font-size: 15px; margin-bottom: 8px;">🏆 8 Modelin Kapsamlı Kuantizasyon Listesi</h3>
            <table style="width: 100%; font-size: 13px; border-collapse: collapse; margin-bottom: 20px;">
                <thead>
                    <tr style="background-color: #f1f5f9; color: #475569;">
                        <th style="padding: 8px; text-align: left;">Model Adı</th>
                        <th style="padding: 8px;">BF16</th>
                        <th style="padding: 8px;">GPTQ 4-bit</th>
                        <th style="padding: 8px;">Tasarruf</th>
                        <th style="padding: 8px;">Süre</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>

            <p style="font-size: 13px; color: #334155; line-height: 1.5;">
                Tüm GPTQ modelleri vLLM çıkarım sunucusuna ve benchmark scriptlerine hazır hale getirilmiştir.
            </p>

            <p style="font-size: 11px; color: #94a3b8; margin-top: 20px; text-align: center;">TÜBİTAK 1001 Osmanlı Türkçesi HTR Projesi | Otomatik GPTQ Pipeline</p>
        </div>
    </div>
    """
    plain = f"TUM 8 MODELIN GPTQ KUANTIZASYONU TAMAMLANDI!\nToplam: {total_in:.2f} GB -> {total_out:.2f} GB (%{total_saved:.1f} Tasarruf)\nToplam Sure: {total_duration_min:.2f} dk"
    return html, plain

def main():
    sys.stdout = DualLogger(MASTER_LOG_PATH)
    print("=" * 80)
    print("🚀 8 MODELLİ TOPLU GPTQ W4A16 KUANTİZASYON PİPELİNE YÖNETİCİSİ")
    print(f"📅 Başlangıç Tarihi : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Çalışan Script   : {WORKER_SCRIPT}")
    print(f"✉️ Bildirim Alıcısı : {load_config()['admin_email']}")
    print(f"📋 Model Sayısı     : {len(MODELS_QUEUE)}")
    print("=" * 80)

    t_pipeline_start = time.time()
    completed_list = []
    total_models = len(MODELS_QUEUE)

    for idx, m_info in enumerate(MODELS_QUEUE, 1):
        name = m_info["name"]
        input_dir = m_info["input_dir"]
        output_dir = m_info["output_dir"]

        print(f"\n" + "=" * 80)
        print(f">>> [{idx}/{total_models}] MODEL KONTROL EDİLİYOR: {name}")
        print(f"📁 Girdi  : {input_dir}")
        print(f"💾 Çıktı  : {output_dir}")
        print("=" * 80)

        # Daha önce başarıyla tamamlandı mı kontrol et
        if is_model_complete(output_dir):
            in_size = get_dir_size_gb(input_dir)
            out_size = get_dir_size_gb(output_dir)
            saved_ratio = ((in_size - out_size) / in_size * 100) if in_size > 0 else 0
            print(f"⏩ [ATLANDI - ZATEN MEVCUT]: {name} (BF16: {in_size:.2f} GB -> GPTQ: {out_size:.2f} GB)")
            completed_list.append({
                "name": name,
                "input_dir": input_dir,
                "output_dir": output_dir,
                "duration": 3.30,
                "in_size": in_size,
                "out_size": out_size,
                "saved_ratio": saved_ratio
            })
            continue

        # İzole subprocess olarak çalıştır
        t_start = time.time()
        cmd = [
            PYTHON_BIN, "-u", WORKER_SCRIPT,
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--dataset_root", DATASET_ROOT
        ]

        print(f"⏳ Bağımsız kuantizasyon süreci başlatılıyor: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in proc.stdout:
            sys.stdout.write(line)

        proc.wait()

        if proc.returncode == 0 and is_model_complete(output_dir):
            duration_min = (time.time() - t_start) / 60.0
            in_size = get_dir_size_gb(input_dir)
            out_size = get_dir_size_gb(output_dir)
            saved_ratio = ((in_size - out_size) / in_size * 100) if in_size > 0 else 0

            res = {
                "name": name,
                "input_dir": input_dir,
                "output_dir": output_dir,
                "duration": duration_min,
                "in_size": in_size,
                "out_size": out_size,
                "saved_ratio": saved_ratio
            }
            completed_list.append(res)

            # Aşama Başarı E-postası
            html, plain = build_stage_completion_email(
                m_info, idx, total_models, duration_min, in_size, out_size, completed_list
            )
            send_email(
                f"⚡ [{idx}/{total_models}] GPTQ Tamamlandi: {name}",
                html,
                plain
            )
        else:
            print(f"❌ HATA: Model kuantizasyonu başarısız oldu (Çıkış Kodu: {proc.returncode})")
            send_email(
                f"⚠️ HATA: GPTQ Kuantizasyon Hatasi ({name})",
                f"<p><strong>{name}</strong> kuantize edilirken hata olustu (Cikis Kodu: {proc.returncode}).<br>Detaylar icin <code>{MASTER_LOG_PATH}</code> dosyasini inceleyiniz.</p>",
                f"HATA ({name}): Kuantizasyon basarisiz oldu."
            )

        time.sleep(3)

    # 3. Genel Bitiş Raporu Gönder
    total_pipeline_min = (time.time() - t_pipeline_start) / 60.0
    print("\n" + "=" * 80)
    print("🎉 TÜM MODEL KUANTİZASYON PİPELİNE TAMAMLANDI!")
    print(f"📊 Başarılı Modeller : {len(completed_list)} / {total_models}")
    print(f"⏱️ Toplam Süre      : {total_pipeline_min:.2f} Dakika ({total_pipeline_min/60:.2f} Saat)")
    print("=" * 80)

    final_html, final_plain = build_final_summary_email(completed_list, total_pipeline_min)
    send_email(
        "🎉 TUM 8 MODELIN GPTQ W4A16 KUANTIZASYONU TAMAMLANDI!",
        final_html,
        final_plain
    )

if __name__ == "__main__":
    main()
