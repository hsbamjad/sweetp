"""
evaluate_all.py — Evaluate ALL trained models on the locked test set

Edit RUNS_DIR and MODELS at the top, then run:
    python scripts/evaluate_all.py

For each model:
  - Loads weights/best.pt from the run folder
  - Evaluates on test split of the specified data.yaml
  - Collects overall + per-class Precision / Recall / F1 / mAP50 / mAP50-95

At the end: prints a ranked comparison table and declares the best model.

Results are saved to:
  runs/eval_all_<timestamp>/summary.txt
  runs/eval_all_<timestamp>/<model_name>/   (plots per model)
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit this section
# ═══════════════════════════════════════════════════════════════════════════════

# Root folder that contains all run sub-folders
RUNS_DIR = r"runs"

# List of (run_folder_name, data_yaml_path) pairs.
# run_folder_name is relative to RUNS_DIR.
# data_yaml_path is relative to the project root (where you run the script).
# Add / remove / comment out entries as needed.

MODELS = [
    # ── 2024-only models ──────────────────────────────────────────────────────
    ("model3_2024only",      "processed_data_2024only/R_NIR1_NIR2/data.yaml"),
    ("model5_2024only",      "processed_data_2024only/R_NIR1_NIR2/data.yaml"),

    # ── Full data models (original processed_data) ───────────────────────────
    ("model3_full",          "processed_data/R_NIR1_NIR2/data.yaml"),
    ("model5_full",          "processed_data/R_NIR1_NIR2/data.yaml"),

    # ── R+G+NIR1 models (updated data) ───────────────────────────────────────
    ("model6_rg_nir1",       "updated_processed_data/R_G_NIR1/data.yaml"),
    ("model7_large_finetune","updated_processed_data/R_G_NIR1/data.yaml"),

    # ── R+G+NIR2-norm models (updated data) ──────────────────────────────────
    ("model8_finetune",      "updated_processed_data/R_G_NIR2/data.yaml"),
    ("model8b_scratch",      "updated_processed_data/R_G_NIR2/data.yaml"),
]

# Confidence threshold used during evaluation
CONF = 0.25
IOU  = 0.5

# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed — run: pip install ultralytics")

CLASS_NAMES = ["Normal", "Moderate defect", "Severe defect"]
NC = len(CLASS_NAMES)


# ─── helpers ──────────────────────────────────────────────────────────────────

def fmt(v, w=7):
    if v is None:
        return " " * (w - 1) + "—"
    try:
        if np.isnan(float(v)):
            return " " * (w - 1) + "—"
        return f"{float(v):.4f}"
    except Exception:
        return " " * (w - 1) + "—"

def f1(p, r):
    if p is None or r is None:
        return None
    p, r = float(p), float(r)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def per_class(metric_arr, idx_arr, n=NC):
    out = [None] * n
    if metric_arr is None or idx_arr is None:
        return out
    for i, cls in enumerate(idx_arr):
        if int(cls) < n:
            out[int(cls)] = float(metric_arr[i])
    return out

def bar(v, width=16):
    if v is None:
        return " " * width
    return ("█" * int(float(v) * width)).ljust(width)


# ─── evaluate one model ───────────────────────────────────────────────────────

def evaluate_model(run_name, data_yaml, out_root):
    """Returns a result dict or None on failure."""

    weights = Path(RUNS_DIR) / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"  [SKIP] {run_name} — weights not found: {weights}")
        return None

    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"  [SKIP] {run_name} — data.yaml not found: {data_path}")
        return None

    print(f"\n  Evaluating: {run_name}")
    print(f"    weights  : {weights}")
    print(f"    data     : {data_path}")

    try:
        model = YOLO(str(weights))
        results = model.val(
            data      = str(data_path),
            split     = "test",
            project   = str(out_root),
            name      = run_name,
            plots     = True,
            verbose   = False,
            conf      = CONF,
            iou       = IOU,
            save_json = False,
        )
    except Exception as e:
        print(f"  [ERROR] {run_name}: {e}")
        return None

    box = results.box
    seg = results.seg if hasattr(results, "seg") and results.seg is not None else None

    ap50_b = per_class(box.ap50, box.ap_class_index)
    ap_b   = per_class(box.ap,   box.ap_class_index)
    p_b    = per_class(box.p,    box.ap_class_index)
    r_b    = per_class(box.r,    box.ap_class_index)

    return {
        "name":       run_name,
        "data":       data_yaml,
        # Overall
        "map50":      box.map50,
        "map":        box.map,
        "prec":       box.mp,
        "rec":        box.mr,
        "map50_seg":  seg.map50 if seg else None,
        "map_seg":    seg.map   if seg else None,
        # Per-class box
        "p_cls":      p_b,
        "r_cls":      r_b,
        "ap50_cls":   ap50_b,
        "ap_cls":     ap_b,
        "f1_cls":     [f1(p_b[i], r_b[i]) for i in range(NC)],
        "f1_mean":    f1(box.mp, box.mr),
    }


# ─── print per-model detail ───────────────────────────────────────────────────

def print_model_detail(r, lines):
    def out(s=""):
        print(s); lines.append(s)

    out()
    out("═" * 66)
    out(f"  MODEL: {r['name']}")
    out(f"  DATA : {r['data']}")
    out("═" * 66)
    out(f"  {'Metric':<28}  {'Box':>8}  {'Mask':>8}")
    out(f"  {'-'*28}  {'-'*8}  {'-'*8}")
    out(f"  {'mAP @ 0.50':<28}  {fmt(r['map50']):>8}  {fmt(r['map50_seg']):>8}")
    out(f"  {'mAP @ 0.50:0.95':<28}  {fmt(r['map']):>8}  {fmt(r['map_seg']):>8}")
    out(f"  {'Precision (mean)':<28}  {fmt(r['prec']):>8}")
    out(f"  {'Recall    (mean)':<28}  {fmt(r['rec']):>8}")
    out(f"  {'F1        (mean)':<28}  {fmt(r['f1_mean']):>8}")
    out()
    out(f"  {'Class':<22}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}")
    out(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for i, name in enumerate(CLASS_NAMES):
        out(f"  {name:<22}  {fmt(r['p_cls'][i]):>7}  {fmt(r['r_cls'][i]):>7}  "
            f"{fmt(r['f1_cls'][i]):>7}  {fmt(r['ap50_cls'][i]):>7}")


# ─── print final comparison ───────────────────────────────────────────────────

def print_comparison(results, lines):
    def out(s=""):
        print(s); lines.append(s)

    ranked = sorted(results, key=lambda x: x["map50"] or 0, reverse=True)

    out()
    out("═" * 90)
    out("  FINAL COMPARISON — ranked by mAP50 (Box, test set)")
    out("═" * 90)
    header = (f"  {'#':<3}  {'Model':<30}  {'mAP50':>7}  {'mAP50-95':>9}  "
              f"{'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    out(header)
    out(f"  {'-'*3}  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}")
    for rank, r in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f" {rank} "))
        out(f"  {medal}  {r['name']:<30}  {fmt(r['map50']):>7}  {fmt(r['map']):>9}  "
            f"{fmt(r['prec']):>7}  {fmt(r['rec']):>7}  {fmt(r['f1_mean']):>7}")

    out()
    out("─" * 90)
    out("  PER-CLASS F1  (test set)")
    out("─" * 90)
    out(f"  {'Model':<30}  {'Normal':^22}  {'Moderate':^22}  {'Severe':^22}")
    out(f"  {'-'*30}  {'-'*22}  {'-'*22}  {'-'*22}")
    for r in ranked:
        row = f"  {r['name']:<30}"
        for i in range(NC):
            v = r['f1_cls'][i]
            cell = f"{fmt(v)}  |{bar(v, 14)}|"
            row += f"  {cell}"
        out(row)

    out()
    out("═" * 90)
    best = ranked[0]
    out(f"  🏆  BEST MODEL : {best['name']}")
    out(f"       mAP50     : {fmt(best['map50'])}")
    out(f"       mAP50-95  : {fmt(best['map'])}")
    out(f"       F1 (mean) : {fmt(best['f1_mean'])}")
    out(f"       Precision : {fmt(best['prec'])}")
    out(f"       Recall    : {fmt(best['rec'])}")
    out("═" * 90)


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("runs") / f"eval_all_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    print("═" * 66)
    print("  SWEET POTATO — BATCH TEST SET EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("═" * 66)
    print(f"\n  Models to evaluate : {len(MODELS)}")
    print(f"  Split              : test  ⚠  LOCKED")
    print(f"  Output             : {out_root}\n")

    lines = []
    results = []

    for run_name, data_yaml in MODELS:
        r = evaluate_model(run_name, data_yaml, out_root)
        if r:
            results.append(r)
            print_model_detail(r, lines)

    if not results:
        print("\nNo models evaluated successfully.")
        return

    print_comparison(results, lines)

    # Save report
    report_path = out_root / "summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Full report saved → {report_path.resolve()}")

if __name__ == "__main__":
    main()
