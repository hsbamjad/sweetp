"""
evaluate_weighted.py — Sweet Potato Model Evaluation with Post-hoc Class Weighting

Runs inference on every image in the chosen split, applies a per-class confidence
multiplier BEFORE thresholding, then computes P / R / mAP@50 / F1 from scratch.
This lets you boost a weaker class (e.g. Severe defect) without retraining.

Edit the CONFIG section, then run:
    python scripts/evaluate_weighted.py

Outputs
-------
  - Console: full metrics table + per-class breakdown
  - runs/<RUN_NAME>/weighted_results.txt : plain-text report
"""

# ===============================================================
#  CONFIG -- edit these lines
# ===============================================================

MODEL_PATH = r"runs/model6_rg_nir1/weights/best.pt"

TEST_DATA  = r"updated_processed_data/R_G_NIR1/data.yaml"

SPLIT      = "test"   # "test" | "val" | "train"

RUN_NAME   = "eval_model6_weighted"   # output folder name under runs/

# -- Per-class confidence multipliers ----------------------------------------
# Index order matches CLASS_NAMES:
#   0 = Normal          -> 1.0  (unchanged)
#   1 = Moderate defect -> 1.0  (unchanged)
#   2 = Severe defect   -> 1.5  (boost: easier to fire above CONF_THRESH)
CLASS_WEIGHTS = [1.0, 1.0, 1.5]

CONF_THRESH  = 0.25   # base confidence threshold (applied AFTER weighting)
IOU_THRESH   = 0.5    # IOU threshold for NMS and for matching pred->GT

# ===============================================================

import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install ultralytics")


CLASS_NAMES = ["Normal", "Moderate defect", "Severe defect"]
NC          = len(CLASS_NAMES)


# --- formatting helpers -------------------------------------------------------

def sep(char="=", width=62):
    print(char * width)

def fmt(v):
    """Format float to 4 dp, or '---' if None/nan."""
    if v is None:
        return "   ---"
    try:
        if np.isnan(float(v)):
            return "   ---"
        return f"{float(v):.4f}"
    except Exception:
        return "   ---"


# --- dataset helpers ----------------------------------------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(data_yaml_path, split):
    """
    Return a list of (image_path, label_path) tuples for the given split.
    Supports both single-string and list values in data.yaml.
    """
    cfg       = load_yaml(data_yaml_path)
    base      = Path(cfg.get("path", Path(data_yaml_path).parent)).resolve()
    split_val = cfg.get(split)
    if split_val is None:
        sys.exit(f"ERROR: split '{split}' not found in {data_yaml_path}")

    img_dirs = split_val if isinstance(split_val, list) else [split_val]

    pairs = []
    for img_dir in img_dirs:
        img_dir_path = base / img_dir
        if not img_dir_path.exists():
            print(f"  WARNING: image dir not found -> {img_dir_path}")
            continue
        # Labels expected at the sibling 'labels' directory
        lbl_dir_path = Path(str(img_dir_path).replace("images", "labels"))

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            for img_p in sorted(img_dir_path.glob(ext)):
                lbl_p = lbl_dir_path / (img_p.stem + ".txt")
                pairs.append((img_p, lbl_p))

    return pairs


def parse_label_file(lbl_path, img_w, img_h):
    """
    Parse a YOLO-format label file — handles BOTH formats:

    Bounding box  (5 values):  cls  cx  cy  w  h   (all normalised)
    Segmentation  (>5 values): cls  x1 y1 x2 y2 x3 y3 ...  (polygon, normalised)

    For segmentation lines the axis-aligned bounding box of the polygon is used,
    which is what model.predict() returns as boxes.xyxy for comparison.
    """
    gts = []
    if not lbl_path.exists():
        return gts
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls    = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) == 4:
            # Standard bbox: cx cy w h  (normalised)
            cx, cy, w, h = coords
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
        else:
            # Segmentation polygon: x1 y1 x2 y2 x3 y3 ...  (normalised)
            # Compute the axis-aligned bounding box of all vertices
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        gts.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return gts


# --- IOU + matching -----------------------------------------------------------

def box_iou_single(a, b):
    """IOU between two boxes given as dicts with x1 y1 x2 y2."""
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_thresh):
    """
    Greedy matching of predictions (sorted by conf desc) to ground-truth boxes.

    Returns list of dicts:
        {conf, pred_cls, gt_cls, tp}
        tp=True  -> correct class AND IOU >= iou_thresh with an unmatched GT
        tp=False -> false positive
    """
    matched    = []
    gt_matched = [False] * len(gts)

    for p in sorted(preds, key=lambda x: -x["conf"]):
        best_iou = 0.0
        best_j   = -1
        for j, g in enumerate(gts):
            if gt_matched[j]:
                continue
            iou = box_iou_single(p, g)
            if iou > best_iou:
                best_iou = iou
                best_j   = j

        if best_j >= 0 and best_iou >= iou_thresh:
            gt_cls = gts[best_j]["cls"]
            gt_matched[best_j] = True
            tp = (p["cls"] == gt_cls)
            matched.append({"conf": p["conf"], "pred_cls": p["cls"],
                            "gt_cls": gt_cls, "tp": tp})
        else:
            matched.append({"conf": p["conf"], "pred_cls": p["cls"],
                            "gt_cls": -1, "tp": False})
    return matched


# --- mAP computation ---------------------------------------------------------

def compute_ap(recalls, precisions):
    """Area under the P-R curve, 101-point interpolation (COCO style)."""
    recalls    = np.concatenate(([0.0], recalls,    [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in thresholds:
        p_at_t = precisions[recalls >= t]
        ap += (p_at_t[0] if len(p_at_t) > 0 else 0.0)
    return ap / 101


def compute_metrics_for_class(all_matches, n_gt, cls_id):
    """
    Compute P, R, AP@50 for one class from the global match list.
    """
    cls_preds = [m for m in all_matches if m["pred_cls"] == cls_id]
    if n_gt == 0 and len(cls_preds) == 0:
        return None, None, None

    cls_preds = sorted(cls_preds, key=lambda x: -x["conf"])

    tps = np.array([1 if m["tp"] else 0 for m in cls_preds], dtype=float)
    fps = 1 - tps

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    recalls    = cum_tp / (n_gt + 1e-9)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    ap50 = compute_ap(recalls, precisions)

    p_final = float(precisions[-1]) if len(precisions) else None
    r_final = float(recalls[-1])    if len(recalls)    else None

    return p_final, r_final, ap50


# --- main --------------------------------------------------------------------

def main():
    sep()
    print("  SWEET POTATO -- WEIGHTED CONFIDENCE EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    sep()
    print(f"\n  Model         : {MODEL_PATH}")
    print(f"  Data          : {TEST_DATA}")
    print(f"  Split         : {SPLIT}")
    print(f"  Class weights : {dict(zip(CLASS_NAMES, CLASS_WEIGHTS))}")
    print(f"  Conf thresh   : {CONF_THRESH}  (applied AFTER weighting)")
    print(f"  IOU thresh    : {IOU_THRESH}")
    print(f"  Output        : runs/{RUN_NAME}\n")

    # -- load model -----------------------------------------------------------
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        sys.exit(f"ERROR: model not found -> {model_path.resolve()}")

    data_path = Path(TEST_DATA)
    if not data_path.exists():
        sys.exit(f"ERROR: data.yaml not found -> {data_path.resolve()}")

    model = YOLO(str(model_path))

    # -- collect image/label pairs --------------------------------------------
    pairs = collect_images(str(data_path), SPLIT)
    if not pairs:
        sys.exit(f"ERROR: no images found for split '{SPLIT}' in {TEST_DATA}")
    print(f"  Images found  : {len(pairs)}\n")

    # -- predict loop with confidence reweighting -----------------------------
    print("  Running inference + applying class weights ...")
    all_matches  = []
    total_gt_cls = [0] * NC
    weights      = np.array(CLASS_WEIGHTS, dtype=float)

    for img_idx, (img_path, lbl_path) in enumerate(pairs):
        # Use a lower raw conf so the net is wide before we apply the weight
        raw_conf_thresh = CONF_THRESH / max(CLASS_WEIGHTS)
        results = model.predict(
            source  = str(img_path),
            conf    = raw_conf_thresh,
            iou     = IOU_THRESH,
            verbose = False,
            save    = False,
        )
        result = results[0]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: cannot read {img_path}")
            continue
        img_h, img_w = img.shape[:2]

        # parse ground truth
        gts = parse_label_file(lbl_path, img_w, img_h)
        for g in gts:
            if g["cls"] < NC:
                total_gt_cls[g["cls"]] += 1

        # apply class weight to raw predictions
        preds = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id   = int(boxes.cls[i].item())
                raw_conf = float(boxes.conf[i].item())

                w        = weights[cls_id] if cls_id < NC else 1.0
                adj_conf = min(raw_conf * w, 1.0)

                if adj_conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                preds.append({"conf": adj_conf, "cls": cls_id,
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        img_matches = match_predictions(preds, gts, IOU_THRESH)
        all_matches.extend(img_matches)

        if (img_idx + 1) % 50 == 0:
            print(f"    processed {img_idx + 1} / {len(pairs)} images ...")

    print(f"  Done. Total predictions (after weighting): {len(all_matches)}")
    print(f"  Total GT boxes: {total_gt_cls}  (sum={sum(total_gt_cls)})\n")

    # -- compute per-class metrics --------------------------------------------
    prec  = []
    rec   = []
    ap50s = []

    for cls_id in range(NC):
        p, r, ap50 = compute_metrics_for_class(
            all_matches, total_gt_cls[cls_id], cls_id)
        prec.append(p)
        rec.append(r)
        ap50s.append(ap50)

    # -- build and print report -----------------------------------------------
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out()
    out("=" * 62)
    out("  OVERALL METRICS  (box, weighted confidence)")
    out("=" * 62)

    valid_ap50 = [v for v in ap50s if v is not None]
    mean_ap50  = float(np.mean(valid_ap50)) if valid_ap50 else None
    valid_p    = [v for v in prec if v is not None]
    valid_r    = [v for v in rec  if v is not None]
    mean_p     = float(np.mean(valid_p)) if valid_p else None
    mean_r     = float(np.mean(valid_r)) if valid_r else None
    mean_f1    = (2 * mean_p * mean_r / (mean_p + mean_r)
                  if mean_p and mean_r and (mean_p + mean_r) > 0 else None)

    out(f"  {'Metric':<28}  {'Box':>8}")
    out(f"  {'-'*28}  {'-'*8}")
    out(f"  {'mAP @ 0.50':<28}  {fmt(mean_ap50):>8}")
    out(f"  {'Precision (mean)':<28}  {fmt(mean_p):>8}")
    out(f"  {'Recall    (mean)':<28}  {fmt(mean_r):>8}")
    out(f"  {'F1        (mean)':<28}  {fmt(mean_f1):>8}")
    out()

    out("=" * 62)
    out("  PER-CLASS -- BOUNDING BOX  (weighted confidence)")
    out("=" * 62)
    out(f"  {'Class':<22}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}  {'n_gt':>6}")
    out(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

    for i, name in enumerate(CLASS_NAMES):
        p, r, ap50 = prec[i], rec[i], ap50s[i]
        f1v = (2 * p * r / (p + r) if p and r and (p + r) > 0 else None)
        weight_tag = f"  x{CLASS_WEIGHTS[i]}" if CLASS_WEIGHTS[i] != 1.0 else ""
        out(f"  {name:<22}  {fmt(p):>7}  {fmt(r):>7}  {fmt(f1v):>7}  "
            f"{fmt(ap50):>7}  {total_gt_cls[i]:>6}{weight_tag}")

    out(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
    out(f"  {'ALL (mean)':<22}  {fmt(mean_p):>7}  {fmt(mean_r):>7}  "
        f"{fmt(mean_f1):>7}  {fmt(mean_ap50):>7}  {sum(total_gt_cls):>6}")
    out()

    out("=" * 62)
    out("  F1 SCORE PER CLASS  (Box)   F1 = 2*P*R / (P+R)")
    out("=" * 62)
    for i, name in enumerate(CLASS_NAMES):
        p, r = prec[i], rec[i]
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2 * p * r / (p + r)
            bar = "#" * int(f1 * 20)
            out(f"  {name:<22}  F1={f1:.4f}  |{bar:<20}|")
        else:
            out(f"  {name:<22}  F1=   ---")
    out()

    out("=" * 62)
    out("  CONFIG USED")
    out("=" * 62)
    out(f"  Model   : {Path(MODEL_PATH).resolve()}")
    out(f"  Data    : {Path(TEST_DATA).resolve()}")
    out(f"  Split   : {SPLIT}")
    out(f"  Weights : {dict(zip(CLASS_NAMES, CLASS_WEIGHTS))}")
    out(f"  Conf    : {CONF_THRESH}  (post-weight)")
    out(f"  IOU     : {IOU_THRESH}")
    out()

    # -- save text report -----------------------------------------------------
    out_dir = Path("runs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "weighted_results.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved -> {report_path.resolve()}")
    sep()


if __name__ == "__main__":
    main()
