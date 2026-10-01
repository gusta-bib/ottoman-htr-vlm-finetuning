import os
import re
import shutil
import time
import subprocess
from collections import Counter
from tqdm import tqdm
from rapidfuzz import fuzz

# ============================================================
# 1. KLASÖR YOLLARI
# ============================================================
DRIVE_ARABIC_LINE_GT_DIR = "<DRIVE_PATH> 3 - Ottoman Turkish/TB3/data-set/ground_truth_line_based_dataset"
DRIVE_ARABIC_PAGE_DIR    = "<DRIVE_PATH> 3 - Ottoman Turkish/TB3/data-set/clean_dataset/annotations_nesih"
DRIVE_LATIN_PAGE_DIR     = "<DRIVE_PATH> 3 - Ottoman Turkish/TB3/data-set/clean_dataset/annotations_transliteration"
DRIVE_OUTPUT_DIR         = "<DRIVE_PATH> 3 - Ottoman Turkish/TB3/data-set/ground_truth_line_based_dataset_latin1"

LOCAL_BASE_DIR          = "/tmp/dataset_work"
LOCAL_ARABIC_LINE_GT    = os.path.join(LOCAL_BASE_DIR, "line_gt")
LOCAL_ARABIC_PAGE       = os.path.join(LOCAL_BASE_DIR, "arabic_page")
LOCAL_LATIN_PAGE        = os.path.join(LOCAL_BASE_DIR, "latin_page")
LOCAL_OUTPUT_DIR        = os.path.join(LOCAL_BASE_DIR, "output_latin")

PAGE_CACHE = {}          # ham (temiz) Latin sayfa kelime listeleri
NORM_CACHE = {}          # normalize edilmiş Latin sayfa kelime listeleri (karşılaştırma için)

# ============================================================
# 2. EŞLEŞTİRME AYARLARI (gerekirse buradan ayarla)
# ============================================================
MIN_SIMILARITY_SCORE = 70     # 0-100 arası, altında kalan pencereler NEEDS_REVIEW'a düşer
WINDOW_TOLERANCE      = 3      # Arapça satırdaki kelime sayısına göre pencere uzunluğu +/- kaç kelime denenecek
SEARCH_LOOKAHEAD      = 400    # son_bulunan_yer'den itibaren en fazla kaç kelime ileri taranacak (performans için sınır)

# ============================================================
# 3. ARAPÇA -> KABA LATİN HARF HARİTASI
#    Amaç mükemmel transkripsiyon değil, sadece rapidfuzz'un
#    karşılaştırabileceği ortak bir "iskelet" üretmek.
#    İhtiyaca göre genişletilebilir/düzeltilebilir.
# ============================================================
ARABIC_TO_LATIN_MAP = {
    # ---------------- TEMEL HARFLER ----------------
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ء": "",
    "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "c", "چ": "ç", "ح": "h", "خ": "h",
    "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "j",
    "س": "s", "ش": "ş", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "", "غ": "g",
    "ف": "f", "ق": "k", "ك": "k", "گ": "g", "ڭ": "n",
    "ل": "l", "م": "m", "ن": "n",
    "و": "v", "ؤ": "v", "ه": "h", "ة": "e",
    "ی": "i", "ي": "i", "ى": "i", "ئ": "i",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    # ---------------- ÖZEL / GİZLİ KARAKTERLER ----------------
    " ": " ",       # Standart boşluk
    "\u200c": "",   # Zero-Width Non-Joiner (Bitişmeyen kelimeler için gizli karakter: "علی‌یه" gibi)
    "ـ": "",        # Tatweel / Kashida (Harf uzatma çizgisi)
    "ً": "an",      # Tenvin / İki üstün ("تحريراً")
    "ک": "k",       # Keheh (Farsça/Arapça Kaf varyantı)
    # ---------------- NOKTALAMA ----------------
    "(": "(", ")": ")",     # Parantez işaretleri ("( تحريراً في 11 رجب 304")
    ",": ",", "،": ",",     # Standart ve Arapça virgül işareti ("عثمان،")
    # ---------------- HAREKELER / TENVİNLER ----------------
    "ٍ": "in",              # Kesre Tenvin / İki esre ("علاوةً" veya tamlamalarda)
    "ٌ": "un",              # Zamme Tenvin / İki ötre
    "ّ": "",                # Şedde (Normalizasyonda harfi çiftlemek yerine gürültüyü önlemek için kaldırılır)
    "ْ": "",                # Cezm / Sükun
    "َ": "a",               # Üstün
    "ِ": "i",               # Esre
    "ُ": "u",               # Ötre
    # ---------------- RAKAMLAR VE SİMBOLOJİ ----------------
    # Doğu Arap Rakamları
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Farsça/Osmanlıca Rakam Varyantları ("11 رجب 304" gibi tarih metinleri için)
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

_NORM_STRIP_PATTERN = re.compile(r"[^a-z0-9]+")


def naive_transliterate(arabic_text: str) -> str:
    """Arapça metni kaba bir Latin 'iskelete' çevirir (fuzzy karşılaştırma için)."""
    out_chars = []
    for ch in arabic_text:
        if ch in ARABIC_TO_LATIN_MAP:
            out_chars.append(ARABIC_TO_LATIN_MAP[ch])
        elif ch.isspace():
            out_chars.append(" ")
        # tanımadığı karakteri (noktalama, harekeler vs.) sessizce atla
    return " ".join("".join(out_chars).split())


def normalize_latin(text: str) -> str:
    """Gerçek Latin transkripsiyonu, kaba çeviriyle adil kıyaslanabilsin diye sadeleştirir:
    küçük harfe çevirir, aksan/noktalama/özel karakterleri (ʻ, ‘, ', â, î, û -> a,i,u vs.) sadeleştirir."""
    text = text.lower()
    accent_map = str.maketrans({
        "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o",
        "ı": "i", "ğ": "g", "ş": "s", "ç": "c", "ö": "o", "ü": "u",
        "‘": "", "’": "", "'": "", "ʻ": "", "ʼ": "", "`": "",
        "-": " ", "–": " ", "—": " ",
    })
    text = text.translate(accent_map)
    text = _NORM_STRIP_PATTERN.sub(" ", text)
    return " ".join(text.split())


def fast_copy(src, dst):
    """Google Drive kilitlenmesini aşmak için rsync tabanlı ultra hızlı kopyalama."""
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{src}/", dst], check=True)


def setup_local_environment():
    print("🚀 Yerel çalışma alanı `rsync` ile Colab diskine taşınıyor...")
    start_setup = time.time()

    if os.path.exists(LOCAL_BASE_DIR):
        shutil.rmtree(LOCAL_BASE_DIR)

    print(" ├─ Satır bazlı Arapça veriler kopyalanıyor...")
    fast_copy(DRIVE_ARABIC_LINE_GT_DIR, LOCAL_ARABIC_LINE_GT)

    print(" ├─ Sayfa bazlı Arapça metinler kopyalanıyor...")
    fast_copy(DRIVE_ARABIC_PAGE_DIR, LOCAL_ARABIC_PAGE)

    print(" └─ Sayfa bazlı Latin metinler kopyalanıyor...")
    fast_copy(DRIVE_LATIN_PAGE_DIR, LOCAL_LATIN_PAGE)

    setup_duration = time.time() - start_setup
    print(f"✅ Bütün veriler {setup_duration:.2f} saniyede Colab diskine alındı!\n")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def get_latin_page_words(doc_base: str):
    """Bir sayfanın hem ham (yazdırılacak) hem normalize edilmiş (karşılaştırılacak)
    Latin kelime listelerini döner. İkisi de aynı index sırasına sahiptir."""
    if doc_base not in PAGE_CACHE:
        latin_page_path = os.path.join(LOCAL_LATIN_PAGE, f"{doc_base}.txt")
        if not os.path.exists(latin_page_path):
            PAGE_CACHE[doc_base] = None
            NORM_CACHE[doc_base] = None
            return None, None

        with open(latin_page_path, "r", encoding="utf-8") as f:
            raw_words = clean_text(f.read()).split()

        norm_words = [normalize_latin(w) for w in raw_words]

        PAGE_CACHE[doc_base] = raw_words
        NORM_CACHE[doc_base] = norm_words

    return PAGE_CACHE[doc_base], NORM_CACHE[doc_base]


def find_best_window(norm_line_words, norm_page_words, start_from: int):
    """norm_page_words içinde, start_from'dan itibaren (SEARCH_LOOKAHEAD ile sınırlı),
    norm_line_words'e en çok benzeyen kelime penceresini bulur.
    Pencere uzunluğu, satır kelime sayısı etrafında +/- WINDOW_TOLERANCE olarak denenir.
    Döner: (best_start, best_length, best_score) ya da None
    """
    n = len(norm_line_words)
    if n == 0:
        return None

    line_joined = " ".join(norm_line_words)
    m = len(norm_page_words)

    search_end = min(m, start_from + SEARCH_LOOKAHEAD)

    best_score = -1
    best_start = None
    best_length = None

    for length in range(max(1, n - WINDOW_TOLERANCE), n + WINDOW_TOLERANCE + 1):
        for start in range(start_from, max(start_from, search_end - length + 1)):
            candidate = " ".join(norm_page_words[start:start + length])
            score = fuzz.ratio(line_joined, candidate)
            if score > best_score:
                best_score = score
                best_start = start
                best_length = length

    if best_start is None:
        return None

    return best_start, best_length, best_score


def extract_line_number(filename: str):
    """'632321_382_21.txt' -> 21. Sıralama için kullanılır. Bulunamazsa çok büyük bir sayı döner
    (böyle dosyalar sıralamanın sonuna düşer, hata olarak zaten INVALID_FILENAME_STRUCTURE'a gider)."""
    stem = filename[:-4] if filename.endswith(".txt") else filename
    tail = stem.rsplit("_", 1)[-1]
    match = re.search(r"\d+", tail)
    return int(match.group()) if match else float("inf")


def process_all():
    setup_local_environment()

    print("📋 Dosya listesi indeksleniyor ve sayfa bazında gruplanıyor...")
    # doc_base -> [(rel_path, filename), ...] şeklinde grupluyoruz ki
    # aynı sayfaya ait satırları sırayla, tek bir "son_bulunan_yer" hafızasıyla işleyebilelim.
    pages = {}
    invalid_files = []

    for root, dirs, files in os.walk(LOCAL_ARABIC_LINE_GT):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            if "_" not in filename:
                invalid_files.append((root, filename))
                continue
            doc_base, _line_str = filename.rsplit("_", 1)
            pages.setdefault(doc_base, []).append((root, filename))

    # Her sayfa içindeki satırları dosya adındaki satır numarasına göre sırala
    for doc_base in pages:
        pages[doc_base].sort(key=lambda rf: extract_line_number(rf[1]))

    review_entries = []
    error_counter = Counter()
    ok_count = 0

    total_lines = sum(len(v) for v in pages.values()) + len(invalid_files)
    print(f"\n⚡ {total_lines} satır, {len(pages)} sayfa içinde gruplanmış olarak eşleştiriliyor...\n")

    # Geçersiz dosya adları
    for root, filename in invalid_files:
        rel_path = os.path.relpath(root, LOCAL_ARABIC_LINE_GT)
        status = "INVALID_FILENAME_STRUCTURE"
        review_entries.append(f"{os.path.join(rel_path, filename)}\t{status}")
        error_counter[status] += 1

    progress = tqdm(total=total_lines - len(invalid_files), desc="Eşleştirme İlerlemesi", unit="satır")

    for doc_base, line_files in pages.items():
        raw_latin_words, norm_latin_words = get_latin_page_words(doc_base)

        # Sayfa başına TEK hafıza: "şu ana kadar nereye kadar geldik"
        last_found_end = 0

        for root, filename in line_files:
            rel_path = os.path.relpath(root, LOCAL_ARABIC_LINE_GT)
            target_dir = os.path.join(LOCAL_OUTPUT_DIR, rel_path) if rel_path != "." else LOCAL_OUTPUT_DIR
            os.makedirs(target_dir, exist_ok=True)

            file_path = os.path.join(root, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                arabic_line_text = f.read()

            if raw_latin_words is None:
                status = "MISSING_PAGE_FILE"
                review_entries.append(f"{os.path.join(rel_path, filename)}\t{status}")
                error_counter[status] += 1
                progress.update(1)
                continue

            pseudo_latin = naive_transliterate(arabic_line_text)
            norm_line_words = pseudo_latin.split()

            if not norm_line_words:
                status = "EMPTY_LINE_TEXT"
                review_entries.append(f"{os.path.join(rel_path, filename)}\t{status}")
                error_counter[status] += 1
                progress.update(1)
                continue

            result = find_best_window(norm_line_words, norm_latin_words, last_found_end)

            if result is None:
                status = "NO_WINDOW_FOUND"
                review_entries.append(f"{os.path.join(rel_path, filename)}\t{status}")
                error_counter[status] += 1
                progress.update(1)
                continue

            best_start, best_length, best_score = result

            if best_score < MIN_SIMILARITY_SCORE:
                status = f"LOW_SIMILARITY({best_score:.0f})"
                review_entries.append(f"{os.path.join(rel_path, filename)}\t{status}")
                error_counter["LOW_SIMILARITY"] += 1
                # last_found_end'i İLERİ ALMIYORUZ: bir sonraki satır yine aynı yerden aramaya devam etsin
                progress.update(1)
                continue

            latin_candidate = " ".join(raw_latin_words[best_start: best_start + best_length])

            out_file_path = os.path.join(target_dir, filename)
            with open(out_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(latin_candidate)

            ok_count += 1
            last_found_end = best_start + best_length  # hafızayı ilerlet
            progress.update(1)

    progress.close()

    # İnceleme dosyasını yaz
    local_review_log = os.path.join(LOCAL_OUTPUT_DIR, "needs_review.txt")
    with open(local_review_log, "w", encoding="utf-8") as log_f:
        log_f.write("\n".join(review_entries))

    print(f"\n🎯 Başarılı eşleşme: {ok_count} | İncelenmesi gereken: {len(review_entries)}")
    for status, count in error_counter.most_common():
        print(f"   ├─ {status}: {count}")

    # Drive'a geri kopyalama (rsync)
    print("\n📤 Çıktılar Google Drive'a kopyalanıyor...")
    fast_copy(LOCAL_OUTPUT_DIR, DRIVE_OUTPUT_DIR)
    print("✅ Tüm işlemler başarıyla tamamlandı!")


🚀 Yerel çalışma alanı `rsync` ile Colab diskine taşınıyor...
 ├─ Satır bazlı Arapça veriler kopyalanıyor...
 ├─ Sayfa bazlı Arapça metinler kopyalanıyor...
 └─ Sayfa bazlı Latin metinler kopyalanıyor...
✅ Bütün veriler 246.58 saniyede Colab diskine alındı!

📋 Dosya listesi indeksleniyor ve sayfa bazında gruplanıyor...

⚡ 25635 satır, 1893 sayfa içinde gruplanmış olarak eşleştiriliyor...

Eşleştirme İlerlemesi: 100%|██████████| 25635/25635 [01:54<00:00, 223.50satır/s]

🎯 Başarılı eşleşme: 16829 | İncelenmesi gereken: 8806
   ├─ LOW_SIMILARITY: 8704
   ├─ NO_WINDOW_FOUND: 100
   ├─ MISSING_PAGE_FILE: 2

📤 Çıktılar Google Drive'a kopyalanıyor...
✅ Tüm işlemler başarıyla tamamlandı!

if __name__ == "__main__":
    process_all()