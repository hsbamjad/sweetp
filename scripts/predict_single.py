"""
predict_single.py — Run inference on a single multispectral sweet potato image

How to use
----------
1. Edit the CONFIG section below (model, image paths, band combination).
2. Run:
       python scripts/predict_single.py

What it does
------------
- Composes the correct 3-channel input from separate RGB / NIR1 / NIR2 images
  (exactly the same stacking logic used in prepare_dataset.py)
- Runs YOLOv8-seg / yolo26-seg inference
- Draws segmentation masks + bounding boxes + class labels + confidence scores
- Saves annotated image to OUTPUT_PATH
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit paths here
# ═══════════════════════════════════════════════════════════════════════════════

# Trained model weights
MODEL_PATH = r"runs/model7_large_finetune/weights/best.pt"

# Source images (set NIR1_PATH / NIR2_PATH to None if not needed for your combo)
RGB_PATH   = r"path/to/your/rgb_image.jpg"
NIR1_PATH  = r"path/to/your/nir1_image.jpg"
NIR2_PATH  = r"path/to/your/nir2_image.jpg"   # set None if not used

# Band combination — must match what the model was trained on:
#   "R_G_NIR1"    → [R, G, NIR1]          (Model 6 / 7)
#   "R_G_NIR2"    → [R, G, NIR2_norm]     (Model 8)
#   "R_NIR1_NIR2" → [R, NIR1, NIR2_norm]  (Model 3 / 5)
#   "RGB"         → [R, G, B]
BAND_COMBO = "R_G_NIR1"

# Output
OUTPUT_PATH = r"runs/prediction_output.jpg"

# Inference settings
CONF_THRESH = 0.25    # minimum confidence to show a detection
IOU_THRESH  = 0.5     # NMS IOU — see NMS_MODE below

# NMS mode — controls how aggressively overlapping detections are suppressed.
# Duplicate boxes on the same potato = lower NMS_MODE value.
#
#   "strict"  → IOU=0.30  aggressive NMS, keeps only the best box per instance ✓ use this to fix doubles
#   "lenient" → IOU=0.70  permissive NMS, allows more overlapping boxes
#   "custom"  → uses NMS_IOU_CUSTOM below
NMS_MODE       = "strict"   # "strict" | "lenient" | "custom"
NMS_IOU_CUSTOM = 0.50       # only used when NMS_MODE = "custom"

# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed — run: pip install ultralytics")


# ─── class config ─────────────────────────────────────────────────────────────

CLASS_NAMES  = ["Normal", "Moderate defect", "Severe defect"]

# BGR colors for each class
CLASS_COLORS = {
    0: (50,  205, 50),    # Normal        → green
    1: (0,   165, 255),   # Moderate      → orange
    2: (0,   0,   220),   # Severe        → red
}

MASK_ALPHA = 0.4   # transparency of filled mask overlay


# ─── compose multispectral image ──────────────────────────────────────────────

def load_gray(path, normalize=False):
    """Load a grayscale image; optionally apply min-max normalization.
    Handles images stored as (H, W, 1) by squeezing to (H, W).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"ERROR: cannot read image → {path}")
    if img.ndim == 3:
        img = img[:, :, 0]   # (H, W, 1) → (H, W)
    if normalize:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def match_size(img, target_hw):
    """Resize img (H,W) to target_hw=(H,W) if needed."""
    th, tw = target_hw
    if img.shape[0] != th or img.shape[1] != tw:
        print(f"    Resizing NIR {img.shape[1]}×{img.shape[0]} → {tw}×{th}")
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    return img



def compose_input(rgb_path, nir1_path, nir2_path, band_combo):
    """
    Build a (H, W, 3) uint8 array matching the band combo used during training.

    Channel order in output:
        R_G_NIR1    → [ R,    G,    NIR1         ]
        R_G_NIR2    → [ R,    G,    NIR2_norm     ]
        R_NIR1_NIR2 → [ R,    NIR1, NIR2_norm     ]
        RGB         → [ R,    G,    B             ]
    """
    combo = band_combo.strip().upper().replace("-", "_")

    # Load RGB
    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        sys.exit(f"ERROR: cannot read RGB image → {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    H, W = R.shape[0], R.shape[1]
    print(f"    RGB array shape : {R.shape}  (H={H}, W={W})")

    def force_resize(img, label="NIR"):
        """Always resize img to (H, W) to match RGB — prints shape before/after."""
        print(f"    {label} array shape : {img.shape}")
        if img.shape[0] != H or img.shape[1] != W:
            print(f"    Resizing {label} → ({H}, {W})")
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return img

    if combo == "RGB":
        stack = np.stack([R, G, B], axis=-1)

    elif combo == "R_G_NIR1":
        if nir1_path is None:
            sys.exit("ERROR: NIR1_PATH is required for R_G_NIR1")
        nir1 = force_resize(load_gray(nir1_path), "NIR1")
        print(f"    Stacking shapes: R={R.shape} G={G.shape} NIR1={nir1.shape}")
        stack = np.stack([R, G, nir1], axis=-1)

    elif combo == "R_G_NIR2":
        if nir2_path is None:
            sys.exit("ERROR: NIR2_PATH is required for R_G_NIR2")
        nir2 = force_resize(load_gray(nir2_path, normalize=True), "NIR2")
        print(f"    Stacking shapes: R={R.shape} G={G.shape} NIR2={nir2.shape}")
        stack = np.stack([R, G, nir2], axis=-1)

    elif combo == "R_NIR1_NIR2":
        if nir1_path is None or nir2_path is None:
            sys.exit("ERROR: both NIR1_PATH and NIR2_PATH are required for R_NIR1_NIR2")
        nir1 = force_resize(load_gray(nir1_path), "NIR1")
        nir2 = force_resize(load_gray(nir2_path, normalize=True), "NIR2")
        print(f"    Stacking shapes: R={R.shape} NIR1={nir1.shape} NIR2={nir2.shape}")
        stack = np.stack([R, nir1, nir2], axis=-1)

    else:
        sys.exit(f"ERROR: unknown BAND_COMBO '{band_combo}'. "
                 "Choose: RGB | R_G_NIR1 | R_G_NIR2 | R_NIR1_NIR2")

    return stack.astype(np.uint8), rgb_bgr



# ─── draw predictions ─────────────────────────────────────────────────────────

def draw_predictions(bgr_display, result):
    """
    Overlay segmentation masks, bounding boxes, and labels on bgr_display.
    Returns annotated BGR image.
    """
    H, W = bgr_display.shape[:2]
    overlay = bgr_display.copy()
    out     = bgr_display.copy()

    boxes  = result.boxes
    masks  = result.masks

    if boxes is None or len(boxes) == 0:
        print("  No detections found.")
        return bgr_display

    n = len(boxes)
    print(f"  Detections: {n}")

    for i in range(n):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label  = f"{CLASS_NAMES[cls_id]}  {conf:.2f}"

        # ── segmentation mask ────────────────────────────────────────────────
        if masks is not None:
            # masks.data is shape (N, H_mask, W_mask), float32 0-1
            mask_data = masks.data[i].cpu().numpy()
            # resize to original image size if needed
            mask_resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
            binary = (mask_resized > 0.5).astype(np.uint8)

            # filled overlay
            colored = np.zeros_like(bgr_display, dtype=np.uint8)
            colored[binary == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, MASK_ALPHA, 0)

            # mask contour
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        # ── bounding box ─────────────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # ── label background + text ──────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(overlay, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(overlay, label, (x1 + 2, ty - 2),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        print(f"    [{i+1}] {CLASS_NAMES[cls_id]}  conf={conf:.3f}  "
              f"box=({x1},{y1},{x2},{y2})")

    return overlay


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  SWEET POTATO — SINGLE IMAGE PREDICTION")
    print("═" * 60)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  RGB        : {RGB_PATH}")
    print(f"  NIR1       : {NIR1_PATH}")
    print(f"  NIR2       : {NIR2_PATH}")
    print(f"  Band combo : {BAND_COMBO}")
    print(f"  Output     : {OUTPUT_PATH}\n")

    # ── compose input ─────────────────────────────────────────────────────────
    print("  Composing input image …")
    input_stack, bgr_display = compose_input(RGB_PATH, NIR1_PATH, NIR2_PATH, BAND_COMBO)
    print(f"  Input shape: {input_stack.shape}  dtype={input_stack.dtype}")

    # ── load model + predict ──────────────────────────────────────────────────
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        sys.exit(f"ERROR: model not found → {model_path.resolve()}")

    # ── resolve NMS IOU from mode ─────────────────────────────────────────────
    nms_map = {"strict": 0.30, "lenient": 0.70, "custom": NMS_IOU_CUSTOM}
    mode = NMS_MODE.strip().lower()
    if mode not in nms_map:
        sys.exit(f'ERROR: NMS_MODE must be "strict", "lenient", or "custom" — got "{mode}"')
    nms_iou = nms_map[mode]
    print(f"  NMS mode   : {mode}  (iou={nms_iou})")

    print("  Loading model …")
    model = YOLO(str(model_path))

    print("  Running inference …\n")
    results = model.predict(
        source    = input_stack,
        conf      = CONF_THRESH,
        iou       = nms_iou,
        verbose   = False,
        save      = False,
    )

    # ── draw and save ─────────────────────────────────────────────────────────
    annotated = draw_predictions(bgr_display, results[0])

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    print(f"\n  Saved → {out_path.resolve()}")
    print("═" * 60)


if __name__ == "__main__":
    main()
