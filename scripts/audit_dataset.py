"""
audit_dataset.py — Sweet Potato Dataset Audit Tool

Scans the 2024 (and optionally 2026) data directories and produces a
comprehensive report on:
  - Image counts per source / per session
  - Annotation counts and class distribution
  - Missing NIR counterparts for annotated RGB frames
  - Image dimension consistency
  - Label format validation

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --data2026 S:/MSU_Research/sweetpotatoes/2026
"""

import argparse
import re
from pathlib import Path
import cv2
import numpy as np

# ─── Paths (relative to project root) ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_2024    = PROJECT_ROOT / "2024" / "data"
RGB_DIR      = DATA_2024 / "ExtractedFrames" / "ConvertSet"   # RGB + labels
NIR1_DIR     = DATA_2024 / "ExtractedFrames1"                  # NIR1 only
NIR2_DIR     = DATA_2024 / "ExtractedFrames2"                  # NIR2 only

# 3-class names used in 2024 grading code
CLASS_NAMES = {0: "Normal", 1: "Moderate defect", 2: "Severe defect"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_session(filename: str):
    """Extract session-id from Source0 filename.

    e.g. 'Source0_20240703_101006_139(1)_frame100.jpg' → session='1', frame=100
    """
    m = re.search(r'\((\d+)\)_frame(\d+)', filename)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def get_nir_path(rgb_path: Path, nir_dir: Path, source_id: int) -> Path:
    """Derive NIR image path from RGB filename.

    RGB:  Source0_20240703_101006_139(1)_frame100.jpg
    NIR1: Source1_20240703_101006_frame100.jpg   (no session tag)
    """
    name = rgb_path.name
    # Replace "Source0_TIMESTAMP_NNN(ID)" → "Source{source_id}_TIMESTAMP"
    nir_name = re.sub(
        r'Source0_(\d{8}_\d{6})_\d+\(\d+\)',
        f'Source{source_id}_\\1',
        name
    )
    return nir_dir / nir_name


def load_label(txt_path: Path):
    """Parse YOLO polygon label file → list of (class_id, polygon_coords)."""
    annotations = []
    if not txt_path.exists():
        return annotations
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            annotations.append((cls, coords))
    return annotations


# ─── Main audit ──────────────────────────────────────────────────────────────

def audit_2024():
    print("\n" + "=" * 70)
    print("  2024 SWEET POTATO DATASET AUDIT")
    print("=" * 70)

    # 1. Inventory
    rgb_images = sorted(RGB_DIR.glob("*.jpg"))
    nir1_images = sorted(NIR1_DIR.glob("*.jpg"))
    nir2_images = sorted(NIR2_DIR.glob("*.jpg"))
    label_files = sorted(RGB_DIR.glob("*.txt"))

    print(f"\n[Inventory]")
    print(f"  RGB images (Source0 / ConvertSet): {len(rgb_images)}")
    print(f"  NIR1 images (ExtractedFrames1):    {len(nir1_images)}")
    print(f"  NIR2 images (ExtractedFrames2):    {len(nir2_images)}")
    print(f"  Label files (.txt):                {len(label_files)}")

    # 2. Session breakdown
    sessions = {}
    for img in rgb_images:
        sid, frame = parse_session(img.name)
        if sid:
            sessions.setdefault(sid, []).append(frame)

    print(f"\n[Sessions] — {len(sessions)} unique potato sessions")
    for sid in sorted(sessions):
        frames = sorted(sessions[sid])
        print(f"  Session {sid}: {len(frames)} frames  "
              f"(frame {frames[0]}–{frames[-1]})")

    # 3. Class distribution
    class_counts = {0: 0, 1: 0, 2: 0}
    total_instances = 0
    images_with_labels = 0
    images_missing_labels = 0
    polygon_lengths = []

    for img in rgb_images:
        txt = RGB_DIR / (img.stem + ".txt")
        anns = load_label(txt)
        if anns:
            images_with_labels += 1
            for cls, coords in anns:
                class_counts[cls] = class_counts.get(cls, 0) + 1
                total_instances += 1
                polygon_lengths.append(len(coords) // 2)
        else:
            images_missing_labels += 1

    print(f"\n[Annotations]")
    print(f"  Total instances:       {total_instances}")
    print(f"  Images with labels:    {images_with_labels}")
    print(f"  Images missing labels: {images_missing_labels}")
    print(f"\n  Class distribution:")
    for cls_id, name in CLASS_NAMES.items():
        n = class_counts.get(cls_id, 0)
        pct = 100 * n / total_instances if total_instances else 0
        bar = "*" * int(pct / 2)
        print(f"    [{cls_id}] {name:<18}: {n:4d}  ({pct:5.1f}%)  {bar}")

    if polygon_lengths:
        print(f"\n  Polygon vertices — min:{min(polygon_lengths)}  "
              f"max:{max(polygon_lengths)}  "
              f"avg:{np.mean(polygon_lengths):.1f}")

    # 4. NIR matching check
    print(f"\n[NIR Matching]")
    missing_nir1 = 0
    missing_nir2 = 0
    for img in rgb_images:
        nir1_path = get_nir_path(img, NIR1_DIR, 1)
        nir2_path = get_nir_path(img, NIR2_DIR, 2)
        if not nir1_path.exists():
            missing_nir1 += 1
        if not nir2_path.exists():
            missing_nir2 += 1

    print(f"  RGB images without NIR1 match: {missing_nir1}")
    print(f"  RGB images without NIR2 match: {missing_nir2}")
    if missing_nir1 == 0 and missing_nir2 == 0:
        print("  ✓ All RGB frames have corresponding NIR1 and NIR2 images")

    # 5. Image dimension check (sample first 20)
    print(f"\n[Image Dimensions — sample of 20 RGB frames]")
    dims = set()
    for img in rgb_images[:20]:
        frame = cv2.imread(str(img))
        if frame is not None:
            dims.add(frame.shape[:2])
    if len(dims) == 1:
        h, w = list(dims)[0]
        print(f"  ✓ All consistent: {w}×{h} px")
    else:
        print(f"  ⚠ Multiple sizes found: {dims}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Dataset year     : 2024")
    print(f"  Total frames     : {len(rgb_images)}")
    print(f"  Potato sessions  : {len(sessions)}")
    print(f"  Total instances  : {total_instances}")
    print(f"  Bands available  : RGB (Source0) + NIR1 (Source1) + NIR2 (Source2)")
    print(f"  Label format     : YOLO polygon (.txt)")
    print(f"  Classes          : 3  (Normal / Moderate defect / Severe defect)")
    print(f"  Class imbalance  : "
          f"Normal={class_counts.get(0,0)}, "
          f"Moderate={class_counts.get(1,0)}, "
          f"Severe={class_counts.get(2,0)}")
    severe = class_counts.get(2, 0)
    if severe < 150:
        print(f"  ⚠ WARNING: Severe defect class has only {severe} instances.")
        print(f"    Consider augmentation or weighted loss for this class.")
    print()

    return {
        "n_frames": len(rgb_images),
        "n_sessions": len(sessions),
        "n_instances": total_instances,
        "class_counts": class_counts,
        "missing_nir1": missing_nir1,
        "missing_nir2": missing_nir2,
    }


def audit_2026(data_dir_2026: Path):
    """Audit 2026 data.

    2026 structure:
      Selected_Source0/  - RGB .bmp  (G{N}_S1_{frame}.bmp)
      Selected_Source1/  - NIR1 .bmp (same names)
      Selected_Source2/  - NIR2 .bmp (same names)
      txt/               - YOLO polygon .txt labels (same stem names)
    """
    print("\n" + "=" * 70)
    print("  2026 SWEET POTATO DATASET AUDIT")
    print("=" * 70)

    if not data_dir_2026.exists():
        print(f"  2026 data directory not found: {data_dir_2026}")
        return None

    src0  = data_dir_2026 / "Selected_Source0"
    src1  = data_dir_2026 / "Selected_Source1"
    src2  = data_dir_2026 / "Selected_Source2"
    lbl_dir = data_dir_2026 / "txt"

    imgs0  = sorted(src0.glob("*.bmp"))
    imgs1  = sorted(src1.glob("*.bmp"))
    imgs2  = sorted(src2.glob("*.bmp"))
    labels = sorted(lbl_dir.glob("*.txt"))

    print(f"\n[Inventory]")
    print(f"  Source0 (RGB .bmp):  {len(imgs0)}")
    print(f"  Source1 (NIR1 .bmp): {len(imgs1)}")
    print(f"  Source2 (NIR2 .bmp): {len(imgs2)}")
    print(f"  Labels (.txt):       {len(labels)}")
    print(f"  Format: BMP (was JPG in 2024)")
    print(f"  Labels location: txt/ folder (was co-located in 2024)")

    # Session / grade breakdown — G{N}_S1_{frame}.bmp
    sessions = {}
    for img in imgs0:
        m = re.match(r'G(\d+)_S1_(\d+)', img.name)
        if m:
            grade, frame = m.group(1), int(m.group(2))
            sessions.setdefault(grade, []).append(frame)

    print(f"\n[Grade Groups] — {len(sessions)} groups (G=potato grade batch)")
    for g in sorted(sessions):
        frames = sorted(sessions[g])
        print(f"  G{g}: {len(frames)} frames  (frame {frames[0]}-{frames[-1]})")

    # Class distribution
    class_counts = {}
    total_instances = 0
    empty_labels = 0
    for txt in labels:
        anns = load_label(txt)
        if not anns:
            empty_labels += 1
            continue
        for cls, _ in anns:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            total_instances += 1

    print(f"\n[Annotations]")
    print(f"  Total instances:    {total_instances}")
    print(f"  Empty label files:  {empty_labels}")
    for cls_id, name in CLASS_NAMES.items():
        n = class_counts.get(cls_id, 0)
        pct = 100 * n / total_instances if total_instances else 0
        bar = "*" * int(pct / 2)
        print(f"    [{cls_id}] {name:<18}: {n:4d}  ({pct:5.1f}%)  {bar}")

    # NIR matching
    missing1 = sum(1 for f in imgs0 if not (src1 / f.name).exists())
    missing2 = sum(1 for f in imgs0 if not (src2 / f.name).exists())
    print(f"\n[NIR Matching]")
    print(f"  Missing NIR1: {missing1}")
    print(f"  Missing NIR2: {missing2}")
    if missing1 == 0 and missing2 == 0:
        print("  All RGB frames have NIR1 and NIR2 matches")

    print(f"\n[Summary]")
    print(f"  Frames: {len(imgs0)}  |  Instances: {total_instances}  |  Grade groups: {len(sessions)}")
    print(f"  Normal={class_counts.get(0,0)}, Moderate={class_counts.get(1,0)}, Severe={class_counts.get(2,0)}")
    print()

    return {
        "n_frames": len(imgs0),
        "n_sessions": len(sessions),
        "n_instances": total_instances,
        "class_counts": class_counts,
        "missing_nir1": missing1,
        "missing_nir2": missing2,
    }


def compare_datasets(stats_2024, stats_2026):
    """Print side-by-side comparison of 2024 vs 2026 datasets."""
    if stats_2024 is None or stats_2026 is None:
        return
    print("\n" + "=" * 70)
    print("  DATASET COMPARISON (2024 vs 2026)")
    print("=" * 70)
    cc24 = stats_2024["class_counts"]
    cc26 = stats_2026["class_counts"]
    tot24 = stats_2024["n_instances"]
    tot26 = stats_2026["n_instances"]
    print(f"{'Class':<20} {'2024':>10} {'2026':>10}  {'Ratio':>8}")
    print("-" * 50)
    for cls_id, name in CLASS_NAMES.items():
        n24 = cc24.get(cls_id, 0)
        n26 = cc26.get(cls_id, 0)
        p24 = 100 * n24 / tot24 if tot24 else 0
        p26 = 100 * n26 / tot26 if tot26 else 0
        ratio = p26 / p24 if p24 > 0 else float('inf')
        flag = " ⚠ SHIFT" if abs(ratio - 1.0) > 0.3 else ""
        print(f"{name:<20} {n24:>5} ({p24:4.1f}%)  {n26:>5} ({p26:4.1f}%){flag}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit sweet potato dataset")
    parser.add_argument("--data2026", type=str, default=None,
                        help="Path to 2026 data directory")
    args = parser.parse_args()

    stats_2024 = audit_2024()

    stats_2026 = None
    if args.data2026:
        stats_2026 = audit_2026(Path(args.data2026))
        compare_datasets(stats_2024, stats_2026)
    else:
        print("ℹ  To also audit 2026 data, run:")
        print("   python scripts/audit_dataset.py --data2026 <path/to/2026>")
