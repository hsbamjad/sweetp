"""
train.py — Sweet Potato Multispectral YOLO Training

Supports 3-class segmentation (Normal / Moderate defect / Severe defect)
with optional multispectral input and SPD-Conv / NIRDiffFusion modifications.

Usage:
    # RGB baseline (Model 1)
    python models/train.py --config configs/model1/rgb_baseline.yaml --name model1_rgb

    # NIR baseline (Model 2)
    python models/train.py --config configs/model2/nir_baseline.yaml --name model2_nir

    # Full SweetPotatoYOLO (Model 5)
    python models/train.py --config configs/model5/sweetpotato_yolo.yaml --name model5_full --custom
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO


# ─── Multispectral YOLO wrapper ───────────────────────────────────────────────

class MultispectralYOLO:
    """Standard YOLO with optional first-layer channel adaptation."""

    def __init__(self, model_size="m", n_channels=3, nc=3, model_base=None):
        self.model_size = model_size
        self.n_channels = n_channels
        self.nc = nc

        if model_base is None:
            model_base = f"yolo26{model_size}-seg.pt"
        self.model_base = model_base

        print(f"\n🔧 Loading base model: {model_base}")
        self.model = YOLO(model_base)

        if n_channels != 3:
            self._adapt_first_conv()

    def _adapt_first_conv(self):
        """Widen / narrow the first conv layer to accept n_channels input."""
        print(f"🔧 Adapting first conv: 3 → {self.n_channels} channels")
        first_conv = self.model.model.model[0].conv
        old_w = first_conv.weight.data

        new_conv = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            if self.n_channels > 3:
                new_conv.weight[:, :3] = old_w
                for i in range(3, self.n_channels):
                    new_conv.weight[:, i] = old_w.mean(dim=1)
            else:
                new_conv.weight[:, :self.n_channels] = old_w[:, :self.n_channels]
            if first_conv.bias is not None:
                new_conv.bias = first_conv.bias

        self.model.model.model[0].conv = new_conv
        print(f"✓ First conv adapted")

    def train(self, data_yaml, epochs=150, imgsz=640, batch=16, **kwargs):
        print(f"\n{'='*60}")
        print(f"Training {self.model_base} — {self.n_channels}ch — nc={self.nc}")
        print(f"{'='*60}\n")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=8,
            patience=50,
            save=True,
            save_period=10,
            verbose=True,
            plots=True,
            **kwargs,
        )
        return results

    def validate(self, data_yaml, **kwargs):
        return self.model.val(data=data_yaml, split="val", plots=True, **kwargs)


# ─── Config loader ────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train sweet potato YOLO model")
    parser.add_argument("--config",  required=True, help="Path to config YAML")
    parser.add_argument("--name",    default=None,  help="Experiment name")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--imgsz",   type=int,   default=None)
    parser.add_argument("--workers", type=int,   default=None)
    parser.add_argument("--custom",  action="store_true",
                        help="Use SweetPotatoYOLO (SPD + NIRDiff) architecture")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Resolve relative paths from project root ─────────────────────────────
    # Allows configs to use 'processed_data/...' instead of absolute paths,
    # making them portable across machines.
    PROJECT_ROOT = Path(__file__).parent.parent

    data_yaml = cfg["data_yaml"]
    if not Path(data_yaml).is_absolute():
        cfg["data_yaml"] = str(PROJECT_ROOT / data_yaml)

    project = cfg.get("project", "runs")
    if not Path(project).is_absolute():
        cfg["project"] = str(PROJECT_ROOT / project)
    # ─────────────────────────────────────────────────────────────────────────

    if args.name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{cfg.get('experiment_name', 'exp')}_{ts}"

    print(f"\n{'='*60}")
    print(f"Experiment: {args.name}")
    print(f"{'='*60}\n")

    # ── build model ──────────────────────────────────────────────────────────
    if args.custom:
        from custom_yolo import SweetPotatoYOLO
        print("🚀 Using SweetPotatoYOLO architecture ...")
        model = SweetPotatoYOLO(
            model_base=cfg.get("model_base", "yolo26m-seg.pt"),
            nc=cfg.get("nc", 3),
            enable_spd=cfg.get("enable_spd", True),
            enable_nir_fusion=cfg.get("enable_nir_fusion", True),
            alpha_init=cfg.get("alpha_init", 0.1),
        )
    else:
        model = MultispectralYOLO(
            model_size=cfg.get("model_size", "m"),
            n_channels=cfg.get("n_channels", 3),
            nc=cfg.get("nc", 3),
            model_base=cfg.get("model_base", None),
        )

    epochs  = args.epochs  or cfg.get("epochs",  150)
    batch   = args.batch   or cfg.get("batch",   16)
    imgsz   = args.imgsz   or cfg.get("imgsz",   640)
    workers = args.workers or cfg.get("workers", 8)

    # ── train ────────────────────────────────────────────────────────────────
    results = model.train(
        data_yaml=cfg["data_yaml"],
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project=cfg.get("project", "runs"),
        name=args.name,
        resume=args.resume,
        # Learning rate schedule
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        warmup_momentum=cfg.get("warmup_momentum", 0.8),
        warmup_bias_lr=cfg.get("warmup_bias_lr", 0.1),
        # Loss weights
        box=cfg.get("box", 7.5),
        cls=cfg.get("cls", 0.5),
        dfl=cfg.get("dfl", 1.5),
        dropout=cfg.get("dropout", 0.0),
        # Augmentation
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 10.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.5),
        flipud=cfg.get("flipud", 0.5),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.0),
        copy_paste=cfg.get("copy_paste", 0.0),
        val=cfg.get("val", True),
    )

    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*60}\n")
    print(f"Results saved to: {cfg.get('project', 'runs')}/{args.name}")

    # ── final validation ──────────────────────────────────────────────────────
    print("\nRunning final validation ...")
    val_results = model.validate(
        data_yaml=cfg["data_yaml"],
        project=cfg.get("project", "runs"),
        name=f"{args.name}_val",
    )
    print("\n📊 Validation Metrics:")
    print(f"  mAP@0.5:     {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:{val_results.box.map:.4f}")
    print(f"  Precision:   {val_results.box.mp:.4f}")
    print(f"  Recall:      {val_results.box.mr:.4f}")


if __name__ == "__main__":
    main()
