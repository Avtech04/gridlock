"""
Ground-truth evaluation utilities for violation detection.

Expected ground-truth format:
{
  "images": [
    {
      "file_name": "image_001.jpg",
      "violations": [
        {"type": "Helmet Non-compliance", "bbox_percent": {"x": 10, "y": 20, "w": 15, "h": 12}}
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1 = float(a.get("x", 0)), float(a.get("y", 0))
    ax2, ay2 = ax1 + float(a.get("w", 0)), ay1 + float(a.get("h", 0))
    bx1, by1 = float(b.get("x", 0)), float(b.get("y", 0))
    bx2, by2 = bx1 + float(b.get("w", 0)), by1 + float(b.get("h", 0))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        images = data
    else:
        images = data.get("images", [])

    by_image: dict[str, list[dict[str, Any]]] = {}
    for item in images:
        file_name = item.get("file_name") or item.get("image") or item.get("path")
        if not file_name:
            continue
        by_image[Path(file_name).name] = item.get("violations", [])
    return by_image


def _match_class(
    gt_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    matched_gt: set[int] = set()
    rows: list[dict[str, Any]] = []
    true_positive = 0
    false_positive = 0

    sorted_preds = sorted(pred_items, key=lambda item: item.get("confidence", 0.0), reverse=True)
    for pred in sorted_preds:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_items):
            if idx in matched_gt:
                continue
            iou = bbox_iou(gt.get("bbox_percent", {}), pred.get("bbox_percent", {}))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            true_positive += 1
            rows.append({"confidence": pred.get("confidence", 0.0), "tp": 1, "fp": 0})
        else:
            false_positive += 1
            rows.append({"confidence": pred.get("confidence", 0.0), "tp": 0, "fp": 1})

    false_negative = len(gt_items) - len(matched_gt)
    return true_positive, false_positive, false_negative, rows


def _average_precision(rows: list[dict[str, Any]], total_gt: int) -> float:
    if total_gt <= 0 or not rows:
        return 0.0
    ordered = sorted(rows, key=lambda item: item.get("confidence", 0.0), reverse=True)
    tp_cum = 0
    fp_cum = 0
    points: list[tuple[float, float]] = []
    for row in ordered:
        tp_cum += int(row["tp"])
        fp_cum += int(row["fp"])
        precision = tp_cum / max(tp_cum + fp_cum, 1)
        recall = tp_cum / total_gt
        points.append((recall, precision))

    ap = 0.0
    prev_recall = 0.0
    for recall_level in [x / 100 for x in range(0, 101)]:
        precision_at_recall = max((p for r, p in points if r >= recall_level), default=0.0)
        ap += precision_at_recall
        prev_recall = recall_level
    return ap / 101 if prev_recall >= 0 else 0.0


def evaluate_predictions(
    ground_truth_by_image: dict[str, list[dict[str, Any]]],
    predictions_by_image: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    classes = sorted(
        {
            item.get("type", "Unknown")
            for values in list(ground_truth_by_image.values()) + list(predictions_by_image.values())
            for item in values
        }
    )

    per_class: list[dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0}

    for class_name in classes:
        gt_by_image = defaultdict(list)
        pred_by_image = defaultdict(list)
        for image_name, items in ground_truth_by_image.items():
            gt_by_image[image_name] = [item for item in items if item.get("type") == class_name]
        for image_name, items in predictions_by_image.items():
            pred_by_image[image_name] = [item for item in items if item.get("type") == class_name]

        class_rows: list[dict[str, Any]] = []
        class_tp = class_fp = class_fn = 0
        image_names = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))
        for image_name in image_names:
            tp, fp, fn, rows = _match_class(gt_by_image[image_name], pred_by_image[image_name], iou_threshold)
            class_tp += tp
            class_fp += fp
            class_fn += fn
            class_rows.extend(rows)

        total_gt = class_tp + class_fn
        precision = class_tp / max(class_tp + class_fp, 1)
        recall = class_tp / max(class_tp + class_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        ap50 = _average_precision(class_rows, total_gt)
        totals["tp"] += class_tp
        totals["fp"] += class_fp
        totals["fn"] += class_fn
        per_class.append(
            {
                "type": class_name,
                "ground_truth": total_gt,
                "detections": class_tp + class_fp,
                "true_positives": class_tp,
                "false_positives": class_fp,
                "false_negatives": class_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "ap50": round(ap50, 4),
            }
        )

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    map50 = sum(item["ap50"] for item in per_class) / max(len(per_class), 1)

    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "accuracy": round(precision, 4),
        },
        "confusion_matrix": {
            "true_positives": totals["tp"],
            "false_positives": totals["fp"],
            "false_negatives": totals["fn"],
            "true_negatives": 0,
        },
        "per_class_metrics": per_class,
        "iou_threshold": iou_threshold,
    }
