"""
train_model9.py — Train Model 9: YOLO26-Large, R+G+NIR1, 1024px

Config: configs/model9/rg_nir1_large_1024.yaml
Run name: yolo26l_ch1_1024_2

Usage (on Linux workstation):
    conda activate sweetp
    cd /path/to/sweetpotatoes
    python scripts/train_model9.py

To resume an interrupted run, set RESUME = True below.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

RESUME = False   # set True to resume from last checkpoint

# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
from pathlib import Path
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed — run: pip install ultralytics")


def main():
    print("═" * 66)
    print("  MODEL 9 — YOLO26-Large  R+G+NIR1  1024px")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("═" * 66)

    # ── paths ─────────────────────────────────────────────────────────────────
    data_yaml   = "/media/workstation2/Elements/Sweetpotato/Original_image/New_RG-NIR1/dataset/"
    project     = "/media/workstation2/Elements/Sweetpotato/Original_image/07082026/runs"
    run_name    = "yolo26l_ch1_1024_2"
    model_base  = "yolo26l-seg.pt"

    print(f"  Model   : {model_base}")
    print(f"  Data    : {data_yaml}")
    print(f"  Project : {project}")
    print(f"  Name    : {run_name}")
    print(f"  Resume  : {RESUME}\n")

    if RESUME:
        resume_ckpt = Path(project) / run_name / "weights" / "last.pt"
        if not resume_ckpt.exists():
            sys.exit(f"ERROR: resume checkpoint not found → {resume_ckpt}")
        print(f"  Resuming from: {resume_ckpt}\n")
        model = YOLO(str(resume_ckpt))
    else:
        model = YOLO(model_base)

    # ── train ─────────────────────────────────────────────────────────────────
    results = model.train(
        task         = "segment",
        data         = data_yaml,
        epochs       = 150,
        patience     = 100,
        batch        = 16,
        imgsz        = 1024,
        workers      = 8,
        project      = project,
        name         = run_name,
        exist_ok     = False,
        pretrained   = True,
        optimizer    = "auto",
        seed         = 0,
        deterministic= True,
        amp          = True,
        fraction     = 1.0,
        close_mosaic = 10,
        resume       = RESUME,
        save         = True,
        save_period  = -1,
        plots        = True,
        verbose      = True,
        # Val / NMS
        val          = True,
        split        = "val",
        iou          = 0.7,
        max_det      = 300,
        # Loss
        box          = 7.5,
        cls          = 0.5,
        dfl          = 1.5,
        # Mask
        overlap_mask = True,
        mask_ratio   = 4,
        # LR
        lr0           = 0.01,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3.0,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        cos_lr        = False,
        nbs           = 64,
        # Augmentation
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 0.0,
        translate    = 0.1,
        scale        = 0.5,
        shear        = 0.0,
        perspective  = 0.0,
        flipud       = 0.0,
        fliplr       = 0.5,
        bgr          = 0.0,
        mosaic       = 1.0,
        mixup        = 0.0,
        cutmix       = 0.0,
        copy_paste   = 0.0,
        copy_paste_mode = "flip",
        auto_augment = "randaugment",
        erasing      = 0.4,
        # Misc
        dropout      = 0.0,
        single_cls   = False,
        rect         = False,
        cache        = False,
        device       = "",
        multi_scale  = 0.0,
        compile      = False,
        profile      = False,
        freeze       = None,
        distill_model= None,
        dis          = 6.0,
        cls_pw       = 0.0,
    )

    print("\n═" * 66)
    print(f"  Training complete — results saved to: {project}/{run_name}")
    print("═" * 66)


if __name__ == "__main__":
    main()
