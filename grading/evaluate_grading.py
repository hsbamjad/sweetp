"""
evaluate_grading.py — Sample-Level Sweet Potato Grading Evaluation

After running detection on a video / batch of images, this script
computes the final sample-level grading accuracy using a temporal
voting / confidence-weighted strategy.

Workflow:
  1. Detection outputs are aggregated per potato (using session / tracking ID)
  2. Each potato gets a final grade via confidence-weighted voting
  3. Final grade is compared to ground truth
  4. Metrics: overall accuracy, per-class P/R/F1, confusion matrix

Input CSV format (one row per detection):
  potato_id, frame_id, predicted_class, confidence, [actual_class]

Usage:
    # Evaluate with ground truth (actual_class column present)
    python grading/evaluate_grading.py --csv results/detections.csv --output results/grading_eval

    # Generate predictions.csv template
    python grading/evaluate_grading.py --template --output results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)

CLASS_NAMES  = ["Normal", "Moderate defect", "Severe defect"]
GRADE_COLORS = {"Normal": "#55A868", "Moderate defect": "#DD8452",
                "Severe defect": "#C44E52"}


# ─── Voting strategies ────────────────────────────────────────────────────────

def majority_vote(classes, confidences):
    """Simple majority vote (ignores confidence)."""
    from collections import Counter
    return Counter(classes).most_common(1)[0][0]


def confidence_weighted_vote(classes, confidences):
    """Confidence-weighted vote — same logic as 2024 grading code.

    Special rule: if any detection is 'Severe defect' with conf >= 0.978,
    it immediately wins regardless of other votes.
    """
    weighted = {c: 0.0 for c in CLASS_NAMES}
    total = sum(confidences)
    for cls, conf in zip(classes, confidences):
        if cls == "Severe defect" and conf >= 0.978:
            return "Severe defect"
        if cls in weighted:
            weighted[cls] += conf
    if total == 0:
        return classes[0] if classes else "Normal"
    return max(weighted, key=weighted.get)


def aggregate_detections(df, strategy="confidence"):
    """Aggregate per-frame detections into per-potato final grades."""
    fn = confidence_weighted_vote if strategy == "confidence" else majority_vote

    results = []
    for potato_id, group in df.groupby("potato_id"):
        classes     = group["predicted_class"].tolist()
        confidences = group["confidence"].tolist()
        final_grade = fn(classes, confidences)
        n_frames    = len(group)
        actual      = group["actual_class"].iloc[0] if "actual_class" in group.columns else None
        results.append({
            "potato_id":   potato_id,
            "final_grade": final_grade,
            "n_frames":    n_frames,
            "actual_class": actual,
        })
    return pd.DataFrame(results)


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_grading(csv_path, output_dir=None, strategy="confidence"):
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"  SAMPLE-LEVEL GRADING EVALUATION")
    print(f"{'='*60}\n")
    print(f"  Input file : {csv_path}")
    print(f"  Rows       : {len(df)}")
    print(f"  Strategy   : {strategy}")

    # Required columns
    for col in ["potato_id", "predicted_class", "confidence"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Aggregate to potato level
    results_df = aggregate_detections(df, strategy)
    print(f"\n  Total unique potatoes: {len(results_df)}")

    # Grade distribution
    grade_dist = results_df["final_grade"].value_counts()
    print("\n  Predicted grade distribution:")
    for grade in CLASS_NAMES:
        n = grade_dist.get(grade, 0)
        print(f"    {grade:<22}: {n}")

    # Evaluate if actual_class available
    has_gt = "actual_class" in results_df.columns and results_df["actual_class"].notna().any()
    metrics = {"n_potatoes": len(results_df), "grade_distribution": grade_dist.to_dict()}

    if not has_gt:
        print("\n  ℹ  No ground truth ('actual_class') — printing predictions only.")
        if output_dir:
            _save(results_df, metrics, output_dir)
        return metrics

    # Filter rows with ground truth
    verified = results_df.dropna(subset=["actual_class"])
    y_true = verified["actual_class"].tolist()
    y_pred = verified["final_grade"].tolist()

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=CLASS_NAMES,
                                   output_dict=True, zero_division=0)

    print(f"\n{'='*60}")
    print("  ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"\n  Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Per-class performance:")
    print(f"  {'Class':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*60}")
    for name in CLASS_NAMES:
        r = report.get(name, {})
        print(f"  {name:<22} {r.get('precision',0):>10.3f} "
              f"{r.get('recall',0):>8.3f} "
              f"{r.get('f1-score',0):>8.3f} "
              f"{int(r.get('support',0)):>8}")

    metrics.update({
        "overall_accuracy": acc,
        "classification_report": report,
    })

    if output_dir:
        _save(verified, metrics, output_dir, y_true, y_pred)

    # Status message
    if acc >= 0.90:
        print(f"\n  🎯 Target ≥90% achieved!  ({acc*100:.2f}%)")
    elif acc >= 0.85:
        print(f"\n  📈 Good ({acc*100:.2f}%). Close to 90% target.")
    else:
        print(f"\n  ⚠  Accuracy {acc*100:.2f}% — needs improvement.")

    return metrics


# ─── Save outputs ─────────────────────────────────────────────────────────────

def _save(results_df, metrics, output_dir, y_true=None, y_pred=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Predictions CSV
    results_df.to_csv(out / "grading_results.csv", index=False)
    print(f"\n  ✓ Results → {out / 'grading_results.csv'}")

    # Metrics JSON
    def _serialize(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_serialize(i) for i in obj]
        return obj

    with open(out / "grading_metrics.json", "w") as f:
        json.dump(_serialize(metrics), f, indent=2)
    print(f"  ✓ Metrics → {out / 'grading_metrics.json'}")

    if y_true is None or y_pred is None:
        return

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix (Sample Level)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Per-class accuracy bar
    acc_by_class = []
    for name in CLASS_NAMES:
        mask = [t == name for t in y_true]
        n_correct = sum(p == t for p, t in zip(y_pred, y_true) if t == name)
        n_total   = sum(mask)
        acc_by_class.append(n_correct / n_total if n_total > 0 else 0.0)

    bars = axes[1].bar(CLASS_NAMES, acc_by_class,
                       color=[GRADE_COLORS[n] for n in CLASS_NAMES])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Per-Class Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].axhline(0.90, color="red", linestyle="--", linewidth=1.5,
                    label="90% target")
    axes[1].legend()
    for bar, val in zip(bars, acc_by_class):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val*100:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out / "grading_results.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Plot → {out / 'grading_results.png'}")
    plt.close()


# ─── Template generator ───────────────────────────────────────────────────────

def generate_template(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame({
        "potato_id":     [1, 1, 1, 2, 2, 3],
        "frame_id":      [10, 15, 20, 10, 15, 10],
        "predicted_class": ["Normal", "Normal", "Moderate defect",
                            "Severe defect", "Severe defect", "Normal"],
        "confidence":    [0.92, 0.88, 0.71, 0.98, 0.95, 0.85],
        "actual_class":  ["Normal", "Normal", "Normal",
                          "Severe defect", "Severe defect", "Normal"],
    })
    path = out / "detections_template.csv"
    template.to_csv(path, index=False)
    print(f"✓ Template CSV → {path}")
    print("\n  Fill 'actual_class' column with ground truth, then run:")
    print("  python grading/evaluate_grading.py --csv detections_template.csv "
          "--output results/grading_eval")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample-level sweet potato grading evaluation")
    parser.add_argument("--csv",      type=str, default=None)
    parser.add_argument("--output",   type=str, default="results/grading_eval")
    parser.add_argument("--strategy", type=str, default="confidence",
                        choices=["confidence", "majority"])
    parser.add_argument("--template", action="store_true",
                        help="Generate a template CSV and exit")
    args = parser.parse_args()

    if args.template:
        generate_template(args.output)
        return

    if not args.csv:
        print("Error: provide --csv or --template")
        return

    evaluate_grading(args.csv, args.output, args.strategy)


if __name__ == "__main__":
    main()
