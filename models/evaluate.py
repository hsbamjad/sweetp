"""
evaluate.py — Sweet Potato Model Evaluation (Instance Level)

Computes:
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95  (overall + per-class)
  - Precision, Recall, F1  (per class)
  - Confusion matrix
  - Speed benchmarks (ms/frame, FPS)

Supports single model or multi-model comparison.

Usage:
    # Single model
    python models/evaluate.py --checkpoint runs/model1_rgb/weights/best.pt \
        --data processed_data/RGB/data.yaml --output results/model1

    # Compare all models
    python models/evaluate.py \
        --checkpoints runs/model1*/weights/best.pt \
        --data processed_data/RGB/data.yaml \
        --output results/comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ultralytics import YOLO

CLASS_NAMES = ["Normal", "Moderate defect", "Severe defect"]


# ─── Single model evaluation ─────────────────────────────────────────────────

def evaluate_model(model_path, data_yaml, save_dir=None, split="val"):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}\n")

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        split=split,
        save_json=True,
        plots=True,
        verbose=True,
    )

    # ── aggregate metrics ────────────────────────────────────────────────────
    metrics = {
        "model": str(model_path),
        "mAP_50":     float(results.box.map50),
        "mAP_75":     float(results.box.map75),
        "mAP_50_95":  float(results.box.map),
        "precision":  float(results.box.mp),
        "recall":     float(results.box.mr),
        "f1": float(
            2 * results.box.mp * results.box.mr
            / (results.box.mp + results.box.mr + 1e-10)
        ),
    }

    # ── per-class ────────────────────────────────────────────────────────────
    per_class = {}
    if hasattr(results.box, "p") and len(results.box.p) >= 3:
        for i, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "precision": float(results.box.p[i]),
                "recall":    float(results.box.r[i]),
                "ap50":      float(results.box.ap50[i]),
            }
    metrics["per_class"] = per_class

    # ── speed ────────────────────────────────────────────────────────────────
    speed = results.speed
    total_ms = float(sum(speed.values()))
    metrics["speed_ms"] = {k: float(v) for k, v in speed.items()}
    metrics["speed_ms"]["total"] = total_ms
    metrics["fps"] = 1000.0 / total_ms if total_ms > 0 else 0.0

    # ── print ────────────────────────────────────────────────────────────────
    print(f"\n📊 Results Summary:")
    print(f"  mAP@0.5:       {metrics['mAP_50']:.4f}")
    print(f"  mAP@0.75:      {metrics['mAP_75']:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics['mAP_50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print(f"  FPS:           {metrics['fps']:.2f}")
    print(f"\n  Per-class:")
    for name, m in per_class.items():
        print(f"    {name:<20} P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  AP50={m['ap50']:.3f}")

    # ── save ─────────────────────────────────────────────────────────────────
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved → {save_dir / 'metrics.json'}")

    return metrics


# ─── Multi-model comparison ───────────────────────────────────────────────────

def compare_models(model_paths, data_yaml, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for mp in model_paths:
        m = evaluate_model(mp, data_yaml,
                           save_dir=output_dir / Path(mp).parent.parent.name)
        m["name"] = Path(mp).parent.parent.name
        all_metrics.append(m)

    # ── comparison table ────────────────────────────────────────────────────
    rows = []
    for m in all_metrics:
        row = {
            "Model":        m["name"],
            "mAP@0.5":      m["mAP_50"],
            "mAP@0.75":     m["mAP_75"],
            "mAP@0.5:0.95": m["mAP_50_95"],
            "Precision":    m["precision"],
            "Recall":       m["recall"],
            "F1":           m["f1"],
            "FPS":          m["fps"],
        }
        # Per-class AP50
        for name in CLASS_NAMES:
            short = name.split()[0]
            row[f"AP50_{short}"] = m["per_class"].get(name, {}).get("ap50", 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table → {csv_path}")
    print("\n" + df.to_string(index=False))

    # ── plots ────────────────────────────────────────────────────────────────
    _plot_comparison(df, output_dir)
    return df, all_metrics


def _plot_comparison(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sweet Potato YOLO — Model Comparison", fontsize=14, fontweight="bold")

    x    = np.arange(len(df))
    w    = 0.25
    models = df["Model"].tolist()

    # mAP comparison
    ax = axes[0, 0]
    ax.bar(x - w, df["mAP@0.5"],      w, label="mAP@0.5",      color="#4C72B0")
    ax.bar(x,     df["mAP@0.75"],     w, label="mAP@0.75",     color="#DD8452")
    ax.bar(x + w, df["mAP@0.5:0.95"],w, label="mAP@0.5:0.95", color="#55A868")
    ax.set_title("Mean Average Precision")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # P / R / F1
    ax = axes[0, 1]
    ax.bar(x - w, df["Precision"], w, label="Precision", color="#4C72B0")
    ax.bar(x,     df["Recall"],    w, label="Recall",    color="#DD8452")
    ax.bar(x + w, df["F1"],        w, label="F1",        color="#55A868")
    ax.set_title("Precision / Recall / F1")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # Per-class AP50
    ax = axes[1, 0]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    for i, name in enumerate(CLASS_NAMES):
        short = name.split()[0]
        col = f"AP50_{short}"
        if col in df.columns:
            ax.bar(x + (i - 1) * w, df[col], w, label=name, color=colors[i])
    ax.set_title("Per-Class AP@0.5")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    # FPS
    ax = axes[1, 1]
    ax.barh(models, df["FPS"], color="#4C72B0")
    ax.set_xlabel("FPS"); ax.set_title("Inference Speed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "model_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Comparison plot → {out}")
    plt.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate sweet potato YOLO models")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--data",   required=True, help="Path to data.yaml")
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--split",  default="val", choices=["val", "test"])
    args = parser.parse_args()

    if args.checkpoints:
        compare_models(args.checkpoints, args.data, args.output)
    elif args.checkpoint:
        evaluate_model(args.checkpoint, args.data, args.output, args.split)
    else:
        print("Error: provide --checkpoint or --checkpoints")


if __name__ == "__main__":
    main()
