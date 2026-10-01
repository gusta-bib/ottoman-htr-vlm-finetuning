#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consonant skeleton extraction: Naive Transliteration and Consonantal Skeleton Extraction
Paper Reference: Section 1.2 (Methodology)

Converts raw Ottoman Arabic HTR draft predictions or Arabic ground truths
into a stripped consonantal Latin skeleton for robust fuzzy alignment against
Latin transcription ground truths.
"""

import re

ARABIC_TO_LATIN_MAP = {
    # ---------------- Consonants & Core Letters ----------------
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
    # ---------------- Special / Invisible Characters ----------------
    " ": " ",
    "\u200c": "",   # Zero-Width Non-Joiner (ZWNJ)
    "ـ": "",        # Tatweel / Kashida
    "ً": "an",      # Tanween fath
    "ٍ": "in",      # Tanween kasr
    "ٌ": "un",      # Tanween damm
    "ّ": "",        # Shaddah
    "ْ": "",        # Sukun
    "َ": "a",       # Fatha
    "ِ": "i",       # Kasra
    "ُ": "u",       # Damma
    "ک": "k",       # Keheh (Perso-Arabic Kaf)
    # ---------------- Punctuation ----------------
    "(": "(", ")": ")",
    ",": ",", "،": ",",
    # ---------------- Eastern Arabic & Persian Numerals ----------------
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

_NORM_STRIP_PATTERN = re.compile(r"[^a-z0-9]+")

def naive_transliterate(arabic_text: str) -> str:
    """Converts Ottoman Arabic text to a coarse Latin skeletal representation."""
    if not arabic_text:
        return ""
    out_chars = []
    for ch in arabic_text:
        if ch in ARABIC_TO_LATIN_MAP:
            out_chars.append(ARABIC_TO_LATIN_MAP[ch])
        elif ch.isspace():
            out_chars.append(" ")
    return " ".join("".join(out_chars).split())

def normalize_latin_skeleton(text: str) -> str:
    """
    Normalizes Latin reference text by stripping diacritics, accents, and punctuation
    to match the consonantal skeletal phonetic space.
    """
    if not text:
        return ""
    text = text.lower()
    accent_map = str.maketrans({
        "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o",
        "ā": "a", "ī": "i", "ū": "u", "ō": "o", "ē": "e",
        "ı": "i", "ğ": "g", "ş": "s", "ç": "c", "ö": "o", "ü": "u",
        "‘": "", "’": "", "'": "", "ʻ": "", "ʼ": "", "`": "",
        "-": " ", "–": " ", "—": " ",
    })
    text = text.translate(accent_map)
    text = _NORM_STRIP_PATTERN.sub(" ", text)
    return " ".join(text.split())
