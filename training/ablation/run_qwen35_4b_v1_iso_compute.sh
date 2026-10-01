#!/usr/bin/env bash
# ==============================================================================
# QWEN3.5-4B BLOCK A ISO-COMPUTE (PARITY VS BLOCK A+B) ÇALIŞTIRICI
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="python3"

echo "=============================================================================="
echo "🚀 QWEN3.5-4B BLOCK A ISO-COMPUTE (PARITY VS BLOCK A+B) BAŞLATILIYOR"
echo "Tarih: $(date)"
echo "Hedef Adım: 4.272 (Iso-FLOP compute parity vs Block A+B 22.782 satır)"
echo "=============================================================================="

# Takip / Monitor servisini arka planda başlat (eğer çalışmıyorsa)
if ! pgrep -f "monitor_finetune_4b_v1_iso_compute.py" > /dev/null; then
    echo "📡 E-posta ve Eğitim Takip Servisi arka planda başlatılıyor..."
    nohup $PYTHON_EXEC -u "$DIR/monitor_finetune_4b_v1_iso_compute.py" > "$DIR/monitor_finetune_4b_v1_iso_compute.log" 2>&1 &
    MONITOR_PID=$!
    echo "✅ Takip Servisi Başlatıldı (PID: $MONITOR_PID)"
else
    echo "ℹ️ Takip Servisi zaten çalışıyor."
fi

# Eğitimi başlat
$PYTHON_EXEC -u "$DIR/finetune_qwen35_4b_latin_v1_iso_compute_4272.py" 2>&1 | tee "$DIR/finetune_qwen35_4b_latin_v1_iso_compute_4272.log"
