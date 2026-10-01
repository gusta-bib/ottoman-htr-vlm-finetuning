#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monotonic Sliding Window Matcher with RapidFuzz
Paper Reference: Section 1.2 (Methodology, 70% similarity threshold)

Aligns line-level OCR draft predictions to full-page transcription texts
using a monotonically progressing sliding window.
"""

from rapidfuzz import fuzz

def find_best_sliding_window(
    norm_line_words: list,
    norm_page_words: list,
    start_from: int = 0,
    window_tolerance: int = 3,
    search_lookahead: int = 400,
    min_similarity: float = 70.0
):
    """
    Finds the optimal word window matching norm_line_words within norm_page_words
    starting strictly from start_from to preserve vertical document monotonicity.
    
    Returns:
        (best_start, best_length, best_score, is_match)
    """
    n = len(norm_line_words)
    if n == 0 or not norm_page_words:
        return None

    line_joined = " ".join(norm_line_words)
    m = len(norm_page_words)
    search_end = min(m, start_from + search_lookahead)

    best_score = -1.0
    best_start = None
    best_length = None

    for length in range(max(1, n - window_tolerance), n + window_tolerance + 1):
        for start in range(start_from, max(start_from, search_end - length + 1)):
            candidate = " ".join(norm_page_words[start : start + length])
            score = fuzz.ratio(line_joined, candidate)
            if score > best_score:
                best_score = score
                best_start = start
                best_length = length

    if best_start is None:
        return None

    is_match = (best_score >= min_similarity)
    return {
        "best_start": best_start,
        "best_length": best_length,
        "score": best_score,
        "is_match": is_match
    }
