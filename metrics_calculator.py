"""
Module: metrics_calculator.py
Menghitung metrik kuantitatif dan analisis error khusus Single-Class (Binary: TARGET vs BUKAN_TARGET).
"""

from typing import List, Dict, Any
import numpy as np


def determine_error_type(gt_label: str, pred_label: str, target_class: str = "TINGGI") -> str:
    """Menentukan kategori jenis kesalahan untuk Single-Class Classification."""
    if gt_label == pred_label:
        return "Tepat (Tidak Ada Kesalahan)"
    if pred_label in ["UNKNOWN", "ERROR"]:
        return "Format / Parse Error"

    if gt_label == target_class and pred_label != target_class:
        if target_class == "TINGGI":
            return "False Negative (KRITIS - Krisis Terabaikan)"
        return f"False Negative (Gagal Deteksi {target_class})"
    elif gt_label != target_class and pred_label == target_class:
        return f"False Positive (Over-detection {target_class})"

    return "Label Mismatch"


def compute_detailed_evaluation(records: List[Dict[str, Any]], target_class: str = "TINGGI") -> Dict[str, Any]:
    """
    Menghitung metrik evaluasi Single-Class (Biner: target_class vs BUKAN_target_class).
    """
    total = len(records)
    if total == 0:
        return {}

    target = target_class.strip().upper()
    neg_target = f"BUKAN_{target}"

    # Confusion Matrix Biner untuk Target Class
    tp = sum(1 for r in records if r["gt_label"] == target and r["pred_label"] == target)
    tn = sum(1 for r in records if r["gt_label"] == neg_target and r["pred_label"] == neg_target)
    fp = sum(1 for r in records if r["gt_label"] == neg_target and r["pred_label"] == target)
    fn = sum(1 for r in records if r["gt_label"] == target and r["pred_label"] != target)

    correct_all = tp + tn
    accuracy_overall = (correct_all / total) * 100.0

    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    fp_rate = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fn_rate = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    # Confidence rata-rata
    confidences = [r.get("confidence", 0.0) for r in records]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    # Latencies
    latencies = [r.get("latency_ms", 0) for r in records if r.get("latency_ms", 0) > 0]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "target_class": target,
        "total_sampel": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_keseluruhan": accuracy_overall,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "avg_confidence": avg_confidence,
        "latency_avg_ms": avg_latency,
        "latency_p95_ms": p95_latency,
    }


def print_single_class_summary(metrics: Dict[str, Any]):
    """Mencetak ringkasan evaluasi Single-Class ke terminal (tanpa multi-class)."""
    if not metrics:
        return
    t = metrics["target_class"]
    print("\n" + "=" * 60)
    print(f"REKAPITULASI HASIL METRIK EVALUASI (SINGLE CLASS: {t})")
    print("=" * 60)
    print(f"Total Sampel Evaluasi              : {metrics['total_sampel']}")
    print(f"Target Kelas Deteksi               : {t} vs BUKAN_{t}")
    print(f"Confusion Matrix (TP / TN / FP / FN): {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    print("-" * 60)
    print(f"Accuracy Binary ({t:<6})          : {metrics['accuracy_keseluruhan']:.2f}%")
    print(f"Precision ({t:<6})                : {metrics['precision']:.2f}%")
    print(f"Recall / Sensitivitas ({t:<6})    : {metrics['recall']:.2f}%")
    print(f"Specificity (BUKAN_{t:<6})        : {metrics['specificity']:.2f}%")
    print(f"F1-Score ({t:<6})                 : {metrics['f1_score']:.2f}%")
    print(f"False Positive Rate                : {metrics['fp_rate']:.2f}%")
    print(f"False Negative Rate                : {metrics['fn_rate']:.2f}%")
    print(f"Rata-rata Confidence Score         : {metrics['avg_confidence']:.4f}")
    print(f"Latency Rata-rata                  : {metrics['latency_avg_ms']:.1f} ms")
    print(f"Latency P95                        : {metrics['latency_p95_ms']:.1f} ms")
    print("=" * 60 + "\n")
