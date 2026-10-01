#!/usr/bin/env python3
"""
ICDAR Handwriting Segmentation Contest Metric Evaluation Script
Based on Demokritos Institute / ICDAR 2007, 2009, 2013 competition methodology.

Formulas:
- MatchScore(i, j) = |G_j \cap R_i| / |G_j \cup R_i|
- One-to-one (o2o): MatchScore(i, j) >= T_a
- Detection Rate (DR / Recall) = o2o / N
- Recognition Accuracy (RA / Precision) = o2o / M
- F-Measure (FM) = 2 * DR * RA / (DR + RA)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from shapely.geometry import Polygon

def evaluate_icdar(gt_file, preds_file, pages_dir=None, output_csv=None):
    script_dir = Path(__file__).resolve().parent
    gt_path = Path(gt_file) if gt_file else script_dir / "ground_truth_segmentation.json"
    preds_path = Path(preds_file) if preds_file else script_dir / "predictions_20_pages.json"

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    with open(preds_path, encoding='utf-8') as f:
        data = json.load(f)

    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    total_N = sum(len(p['gt']) for p in data.values())
    total_M = sum(len(p['preds']) for p in data.values())

    o2o_geom = {t: 0 for t in thresholds}
    o2o_pixel = {t: 0 for t in thresholds}

    pdir = Path(pages_dir) if pages_dir else None
    has_images = pdir and pdir.exists()

    for pid, pdata in data.items():
        w = pdata['width']
        h = pdata['height']
        gt_boxes = pdata['gt']
        pred_boxes = pdata['preds']

        # Ink mask extraction if image dir is provided
        ink_mask = None
        if has_images:
            cands = list(pdir.glob(f"{pid}.*"))
            if cands:
                with open(cands[0], 'rb') as f_img:
                    arr = np.asarray(bytearray(f_img.read()), dtype=np.uint8)
                    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    ink_mask = (binary > 0)

        gt_polys = []
        for b in gt_boxes:
            bx, by, bw, bh = b['x'], b['y'], b['width'], b['height']
            poly = Polygon([(bx, by), (bx+bw, by), (bx+bw, by+bh), (bx, by+bh)])
            if not poly.is_valid:
                poly = poly.buffer(0)
            gt_polys.append(poly)

        pred_polys = []
        for b in pred_boxes:
            poly = Polygon(b)
            if not poly.is_valid:
                poly = poly.buffer(0)
            pred_polys.append(poly)

        n_pr = len(pred_polys)
        n_gt = len(gt_polys)
        match_geom = np.zeros((n_pr, n_gt))
        match_pix = np.zeros((n_pr, n_gt))

        # Precompute ink pixel counts if ink_mask is present
        gt_ink_counts = []
        pred_ink_counts = []
        if ink_mask is not None:
            for p in gt_polys:
                minx, miny, maxx, maxy = map(int, p.bounds)
                sub_ink = ink_mask[max(0, miny):min(h, maxy), max(0, minx):min(w, maxx)]
                gt_ink_counts.append(np.count_nonzero(sub_ink) if sub_ink.size > 0 else 0)
            for p in pred_polys:
                minx, miny, maxx, maxy = map(int, p.bounds)
                sub_ink = ink_mask[max(0, miny):min(h, maxy), max(0, minx):min(w, maxx)]
                pred_ink_counts.append(np.count_nonzero(sub_ink) if sub_ink.size > 0 else 0)

        for i, pr in enumerate(pred_polys):
            for j, gt in enumerate(gt_polys):
                if pr.intersects(gt):
                    try:
                        inter_poly = pr.intersection(gt)
                        inter_area = inter_poly.area
                        union_area = pr.area + gt.area - inter_area
                        match_geom[i, j] = inter_area / union_area if union_area > 0 else 0.0

                        if ink_mask is not None and inter_area > 0:
                            minx, miny, maxx, maxy = map(int, inter_poly.bounds)
                            sub_ink = ink_mask[max(0, miny):min(h, maxy), max(0, minx):min(w, maxx)]
                            inter_ink = np.count_nonzero(sub_ink) if sub_ink.size > 0 else 0
                            p_ink = pred_ink_counts[i]
                            g_ink = gt_ink_counts[j]
                            union_ink = p_ink + g_ink - inter_ink
                            match_pix[i, j] = inter_ink / union_ink if union_ink > 0 else 0.0
                    except Exception:
                        pass

        for t in thresholds:
            o2o_geom[t] += len(np.argwhere(match_geom >= t))
            if ink_mask is not None:
                o2o_pixel[t] += len(np.argwhere(match_pix >= t))

    # Output formatted report
    print("=" * 80)
    print("ICDAR HANDWRITING SEGMENTATION CONTEST METRIC EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Pages Evaluated : {len(data)}")
    print(f"Ground Truth Lines (N): {total_N}")
    print(f"Predicted Lines (M)   : {total_M}")
    print("-" * 80)

    rows = []

    if has_images:
        print("\n--- MODALITY 1: PIXEL-LEVEL FOREGROUND INK (TRUE ICDAR COMPETITION STANDARD) ---")
        print(f"{'Ta Threshold':<16} | {'o2o':<6} | {'DR (Recall)':<12} | {'RA (Prec)':<12} | {'FM (F-Score)':<12}")
        print("-" * 75)
        for t in thresholds:
            cnt = o2o_pixel[t]
            dr = cnt / total_N * 100
            ra = cnt / total_M * 100
            fm = 2 * dr * ra / (dr + ra) if (dr + ra) > 0 else 0.0
            print(f"Ta = {t:.2f}           | {cnt:<6} | %{dr:<11.2f} | %{ra:<11.2f} | %{fm:<11.2f}")
            rows.append(["Pixel Foreground Ink", f"{t:.2f}", cnt, total_N, total_M, f"{dr:.2f}", f"{ra:.2f}", f"{fm:.2f}"])

    print("\n--- MODALITY 2: GEOMETRIC BOUNDING BOX / POLYGON AREA ---")
    print(f"{'Ta Threshold':<16} | {'o2o':<6} | {'DR (Recall)':<12} | {'RA (Prec)':<12} | {'FM (F-Score)':<12}")
    print("-" * 75)
    for t in thresholds:
        cnt = o2o_geom[t]
        dr = cnt / total_N * 100
        ra = cnt / total_M * 100
        fm = 2 * dr * ra / (dr + ra) if (dr + ra) > 0 else 0.0
        print(f"Ta = {t:.2f}           | {cnt:<6} | %{dr:<11.2f} | %{ra:<11.2f} | %{fm:<11.2f}")
        rows.append(["Geometric Bounding Box", f"{t:.2f}", cnt, total_N, total_M, f"{dr:.2f}", f"{ra:.2f}", f"{fm:.2f}"])

    if output_csv:
        import csv
        with open(output_csv, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Modality", "Threshold_Ta", "One_to_One_Matches", "GT_Lines_N", "Pred_Lines_M", "Detection_Rate_Recall_Pct", "Recognition_Accuracy_Precision_Pct", "F_Measure_Pct"])
            writer.writerows(rows)
        print(f"\nSaved CSV results to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ICDAR Line Segmentation Metrics")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground_truth_segmentation.json")
    parser.add_argument("--preds", type=str, default=None, help="Path to predictions JSON")
    parser.add_argument("--pages_dir", type=str, default=None, help="Path to page images directory (for pixel ink analysis)")
    parser.add_argument("--output_csv", type=str, default=None, help="Export output CSV file path")
    args = parser.parse_args()

    evaluate_icdar(args.gt, args.preds, args.pages_dir, args.output_csv)
