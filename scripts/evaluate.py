"""
evaluate.py — Sweet Potato Model Evaluation on LOCKED Test Set

Edit the two variables in CONFIG, then run:
    python scripts/evaluate.py

Outputs
-------
  - Console: full metrics table + per-class breakdown
  - runs/<RUN_NAME>/  : saved plots (confusion matrix, PR curve, etc.)
  - runs/<RUN_NAME>/test_results.txt : plain-text report
"""

# ═══════════════════════════════════════════════════════════════
#  CONFIG — edit these two lines only
# ═══════════════════════════════════════════════════════════════

MODEL_PATH = r"runs/model6_rg_nir1/weights/best.pt"

TEST_DATA  = r"updated_processed_data/R_G_NIR1/data.yaml"

SPLIT      = "test"   # "test" | "val" | "train"  (NOT "valid" — use "val")

RUN_NAME   = "eval_model6_test"     # output folder name under runs/

# ═══════════════════════════════════════════════════════════════

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install ultralytics")


CLASS_NAMES = ["Normal", "Moderate defect", "Severe defect"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def sep(char="═", width=62):
    print(char * width)

def fmt(v):
    """Format float to 4 dp, or '—' if None/nan."""
    if v is None:
        return "   —  "
    try:
        if np.isnan(float(v)):
            return "   —  "
        return f"{float(v):.4f}"
    except Exception:
        return "   —  "


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    sep()
    print("  SWEET POTATO — TEST SET EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    sep()
    print(f"\n  Model : {MODEL_PATH}")
    print(f"  Data  : {TEST_DATA}")
    # ── validate + normalise split name ─────────────────────────────────────
    split = SPLIT.strip().lower()
    if split == "valid":
        split = "val"   # YOLO uses "val" not "valid"
    if split not in ("train", "val", "test"):
        sys.exit(f'ERROR: SPLIT must be "train", "val", or "test" — got "{split}"')

    note = "  ⚠  LOCKED — final paper numbers only" if split == "test" else ""
    print(f"  Split : {split}{note}")
    print(f"  Output: runs/{RUN_NAME}\n")

    # ── load model ────────────────────────────────────────────────────────────
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        sys.exit(f"ERROR: model not found → {model_path.resolve()}")

    data_path = Path(TEST_DATA)
    if not data_path.exists():
        sys.exit(f"ERROR: data.yaml not found → {data_path.resolve()}")

    model = YOLO(str(model_path))

    # ── run validation on test split ──────────────────────────────────────────
    print(f"Running evaluation on {split} split …\n")
    results = model.val(
        data      = str(data_path),
        split     = split,
        project   = "runs",
        name      = RUN_NAME,
        plots     = True,
        verbose   = False,
        conf      = 0.25,
        iou       = 0.5,
        save_json = False,
    )

    # ── extract metrics ───────────────────────────────────────────────────────
    box = results.box
    seg = results.seg if hasattr(results, "seg") and results.seg is not None else None

    nc = len(CLASS_NAMES)

    def per_class(metric_array, class_idx_array, n=nc):
        """Map per-class metric array back to fixed nc slots."""
        out = [None] * n
        if metric_array is None or class_idx_array is None:
            return out
        for i, cls in enumerate(class_idx_array):
            if int(cls) < n:
                out[int(cls)] = float(metric_array[i])
        return out

    ap50_box = per_class(box.ap50, box.ap_class_index)
    ap_box   = per_class(box.ap,   box.ap_class_index)
    prec_box = per_class(box.p,    box.ap_class_index)
    rec_box  = per_class(box.r,    box.ap_class_index)

    if seg is not None:
        ap50_seg = per_class(seg.ap50, seg.ap_class_index)
        ap_seg   = per_class(seg.ap,   seg.ap_class_index)
        prec_seg = per_class(seg.p,    seg.ap_class_index)
        rec_seg  = per_class(seg.r,    seg.ap_class_index)
    else:
        ap50_seg = ap_seg = prec_seg = rec_seg = [None] * nc

    # ── print + collect report ────────────────────────────────────────────────
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out()
    out("═" * 62)
    out("  OVERALL METRICS")
    out("═" * 62)
    out(f"  {'Metric':<28}  {'Box':>8}  {'Mask':>8}")
    out(f"  {'-'*28}  {'-'*8}  {'-'*8}")
    out(f"  {'mAP @ 0.50':<28}  {fmt(box.map50):>8}  {fmt(seg.map50 if seg else None):>8}")
    out(f"  {'mAP @ 0.50:0.95':<28}  {fmt(box.map):>8}  {fmt(seg.map if seg else None):>8}")
    out(f"  {'Precision (mean)':<28}  {fmt(box.mp):>8}  {fmt(seg.mp if seg else None):>8}")
    out(f"  {'Recall    (mean)':<28}  {fmt(box.mr):>8}  {fmt(seg.mr if seg else None):>8}")
    # Mean F1
    def mean_f1(p, r):
        return fmt(2*p*r/(p+r)) if p and r and (p+r) > 0 else "   —  "
    out(f"  {'F1        (mean)':<28}  {mean_f1(box.mp, box.mr):>8}  {mean_f1(seg.mp, seg.mr) if seg else '   —  ':>8}")
    out()

    out("═" * 62)
    out("  PER-CLASS — BOUNDING BOX")
    out("═" * 62)
    out(f"  {'Class':<22}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}  {'mAP50-95':>9}")
    out(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
    f1_box = []
    for i, name in enumerate(CLASS_NAMES):
        p, r = prec_box[i], rec_box[i]
        f1v = 2*p*r/(p+r) if p and r and (p+r) > 0 else None
        f1_box.append(f1v)
        out(f"  {name:<22}  {fmt(p):>7}  {fmt(r):>7}  {fmt(f1v):>7}  "
            f"{fmt(ap50_box[i]):>7}  {fmt(ap_box[i]):>9}")
    out(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")
    mean_f1_box = sum(v for v in f1_box if v) / max(1, sum(1 for v in f1_box if v))
    out(f"  {'ALL (mean)':<22}  {fmt(box.mp):>7}  {fmt(box.mr):>7}  {fmt(mean_f1_box):>7}  "
        f"{fmt(box.map50):>7}  {fmt(box.map):>9}")
    out()

    if seg is not None:
        out("═" * 62)
        out("  PER-CLASS — SEGMENTATION MASK")
        out("═" * 62)
        out(f"  {'Class':<22}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}  {'mAP50-95':>9}")
        out(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
        f1_seg = []
        for i, name in enumerate(CLASS_NAMES):
            p, r = prec_seg[i], rec_seg[i]
            f1v = 2*p*r/(p+r) if p and r and (p+r) > 0 else None
            f1_seg.append(f1v)
            out(f"  {name:<22}  {fmt(p):>7}  {fmt(r):>7}  {fmt(f1v):>7}  "
                f"{fmt(ap50_seg[i]):>7}  {fmt(ap_seg[i]):>9}")
        out(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")
        mean_f1_seg = sum(v for v in f1_seg if v) / max(1, sum(1 for v in f1_seg if v))
        out(f"  {'ALL (mean)':<22}  {fmt(seg.mp):>7}  {fmt(seg.mr):>7}  {fmt(mean_f1_seg):>7}  "
            f"{fmt(seg.map50):>7}  {fmt(seg.map):>9}")
        out()

    # ── F1 per class ──────────────────────────────────────────────────────────
    out("═" * 62)
    out("  F1 SCORE PER CLASS  (Box)   F1 = 2·P·R / (P+R)")
    out("═" * 62)
    for i, name in enumerate(CLASS_NAMES):
        p = prec_box[i]; r = rec_box[i]
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2 * p * r / (p + r)
            bar = "█" * int(f1 * 20)
            out(f"  {name:<22}  F1={f1:.4f}  |{bar:<20}|")
        else:
            out(f"  {name:<22}  F1=   —")
    out()

    out("═" * 62)
    out("  PATHS")
    out("═" * 62)
    out(f"  Model  : {Path(MODEL_PATH).resolve()}")
    out(f"  Data   : {Path(TEST_DATA).resolve()}")
    out(f"  Plots  : runs/{RUN_NAME}/")
    out()
    out(f"  NOTE: Confusion matrix + PR curves saved to runs/{RUN_NAME}/")
    out()

    # ── save text report ──────────────────────────────────────────────────────
    out_dir = Path("runs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "test_results.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved → {report_path.resolve()}")
    print("═" * 62)


if __name__ == "__main__":
    main()
