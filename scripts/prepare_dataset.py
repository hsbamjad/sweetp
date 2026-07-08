"""
prepare_dataset.py — Multispectral Sweet Potato Dataset Preparation

Merges 2024 + 2026 data and builds train/val/test splits for 4 band combinations.

2024 format:
  ExtractedFrames/ConvertSet/   — Source0 RGB  (.jpg) + labels (.txt co-located)
  ExtractedFrames1/             — Source1 NIR1 (.jpg)
  ExtractedFrames2/             — Source2 NIR2 (.jpg)
  Naming: Source0_YYYYMMDD_HHMMSS_NNN(ID)_frameN.jpg

2026 format:
  Selected_Source0/             — RGB  (.bmp)
  Selected_Source1/             — NIR1 (.bmp, same filename)
  Selected_Source2/             — NIR2 (.bmp, same filename)
  txt/                          — YOLO polygon labels (.txt, same stem)
  Naming: G{grade}_S1_{frame}.bmp

Split strategy:  stratified BY SESSION (not by frame) to prevent data leakage.
  - Each potato session / grade-batch stays entirely in train, val, OR test.
  - This is critical: frames from the same potato are highly correlated.
  - Test set is LOCKED — never seen during training or model selection.

Usage:
    python scripts/prepare_dataset.py                                   # 2024 only  → processed_data/
    python scripts/prepare_dataset.py --include2026                     # merge 2024+2026 → processed_data/
    python scripts/prepare_dataset.py --output-dir processed_data_2024only          # explicit output dir
    python scripts/prepare_dataset.py --include2026 --output-dir processed_data_combined
    python scripts/prepare_dataset.py --dryrun                          # count only, write nothing
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_2024    = PROJECT_ROOT / "2024" / "data"
DATA_2026    = PROJECT_ROOT / "2026"
OUTPUT_DIR   = PROJECT_ROOT / "processed_data"

CLASS_NAMES  = ["Normal", "Moderate defect", "Severe defect"]
NC           = 3

# Band combinations: name → (tag, channel_spec)
# channel_spec elements: 0=R, 1=G, 2=B, "nir1"=NIR1-grey, "nir2"=NIR2-grey
COMBINATIONS = {
    "RGB":         [0, 1, 2],
    "R_G_NIR1":    [0, 1, "nir1"],
    "R_G_NIR2":    [0, 1, "nir2"],
    "R_NIR1_NIR2": [0, "nir1", "nir2"],
}


# ─── Entry descriptor ─────────────────────────────────────────────────────────

class Entry:
    """Represents one annotated frame (RGB + NIR1 + NIR2 + label)."""
    def __init__(self, rgb_path, nir1_path, nir2_path, label_path, session_id):
        self.rgb   = Path(rgb_path)
        self.nir1  = Path(nir1_path)
        self.nir2  = Path(nir2_path)
        self.label = Path(label_path)
        self.session = session_id   # used for stratified split


# ─── 2024 collector ───────────────────────────────────────────────────────────

def collect_2024():
    """Collect all valid annotated frames from 2024 data."""
    rgb_dir  = DATA_2024 / "ExtractedFrames" / "ConvertSet"
    nir1_dir = DATA_2024 / "ExtractedFrames1"
    nir2_dir = DATA_2024 / "ExtractedFrames2"

    entries = []
    skipped = 0
    for rgb in sorted(rgb_dir.glob("*.jpg")):
        label = rgb_dir / (rgb.stem + ".txt")
        if not label.exists():
            skipped += 1
            continue

        # Derive NIR paths: Source0_DATE_TIME_NNN(ID)_frameN → Source1/2_DATE_TIME_frameN
        nir_name = re.sub(
            r'Source0_(\d{8}_\d{6})_\d+\(\d+\)',
            r'Source1_\1',
            rgb.name
        )
        nir1 = nir1_dir / nir_name
        nir2 = nir2_dir / nir_name.replace("Source1_", "Source2_")

        if not nir1.exists() or not nir2.exists():
            skipped += 1
            continue

        # Session ID from filename: (1), (2) ... (5)
        m = re.search(r'\((\d+)\)', rgb.name)
        sid = f"2024_s{m.group(1)}" if m else "2024_s0"

        entries.append(Entry(rgb, nir1, nir2, label, session_id=sid))

    print(f"  2024: {len(entries)} frames collected, {skipped} skipped")
    return entries


# ─── 2026 collector ───────────────────────────────────────────────────────────

def collect_2026():
    """Collect all valid annotated frames from 2026 data."""
    src0    = DATA_2026 / "Selected_Source0"
    src1    = DATA_2026 / "Selected_Source1"
    src2    = DATA_2026 / "Selected_Source2"
    lbl_dir = DATA_2026 / "txt"

    entries = []
    skipped = 0
    for bmp in sorted(src0.glob("*.bmp")):
        label = lbl_dir / (bmp.stem + ".txt")
        if not label.exists():
            skipped += 1
            continue

        nir1 = src1 / bmp.name
        nir2 = src2 / bmp.name
        if not nir1.exists() or not nir2.exists():
            skipped += 1
            continue

        # Session ID from grade-group prefix: G1, G2 ... G8
        m = re.match(r'G(\d+)_', bmp.name)
        sid = f"2026_g{m.group(1)}" if m else "2026_g0"

        entries.append(Entry(bmp, nir1, nir2, label, session_id=sid))

    print(f"  2026: {len(entries)} frames collected, {skipped} skipped")
    return entries


# ─── Session-stratified 3-way split ─────────────────────────────────────────

def session_three_split(entries, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Split by session into train / val / test — no frame-level leakage.

    All frames from the same potato session stay entirely in ONE split.
    Greedy assignment: fill test first, then val, rest goes to train.

    Why test first?
      Ensures test sessions are chosen before train pressure biases the pool.
      Val is also filled before train, so train always gets the majority.

    Args:
        entries:       list of Entry objects
        val_fraction:  target fraction for validation  (default 0.15)
        test_fraction: target fraction for test         (default 0.15)
        seed:          random seed for reproducibility

    Returns:
        (train_entries, val_entries, test_entries)
    """
    sessions = {}
    for e in entries:
        sessions.setdefault(e.session, []).append(e)

    rng = random.Random(seed)
    session_keys = list(sessions.keys())
    rng.shuffle(session_keys)

    total       = len(entries)
    test_target = total * test_fraction
    val_target  = total * val_fraction

    test_entries  = []
    val_entries   = []
    train_entries = []
    test_count, val_count = 0, 0

    for key in session_keys:
        group = sessions[key]
        if test_count < test_target:
            test_entries.extend(group)
            test_count += len(group)
        elif val_count < val_target:
            val_entries.extend(group)
            val_count += len(group)
        else:
            train_entries.extend(group)

    return train_entries, val_entries, test_entries


# ─── Image builder ────────────────────────────────────────────────────────────

def build_image(entry: Entry, channels: list) -> np.ndarray:
    """Compose a 3-channel uint8 image from specified channel spec."""
    # Load RGB (handles both .jpg and .bmp)
    rgb = cv2.imread(str(entry.rgb))
    if rgb is None:
        raise IOError(f"Cannot read RGB: {entry.rgb}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load NIR channels (always grayscale)
    nir1 = cv2.imread(str(entry.nir1), cv2.IMREAD_GRAYSCALE)
    nir2 = cv2.imread(str(entry.nir2), cv2.IMREAD_GRAYSCALE)

    if nir1 is None: raise IOError(f"Cannot read NIR1: {entry.nir1}")
    if nir2 is None: raise IOError(f"Cannot read NIR2: {entry.nir2}")

    # Normalize NIR2 (lower dynamic range in this sensor)
    nir2 = cv2.normalize(nir2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    band_map = {
        0: rgb[:, :, 0],     # R
        1: rgb[:, :, 1],     # G
        2: rgb[:, :, 2],     # B
        "nir1": nir1,
        "nir2": nir2,
    }
    planes = [band_map[c] for c in channels]
    return np.stack(planes, axis=-1).astype(np.uint8)


# ─── Dataset builder ─────────────────────────────────────────────────────────

def build_dataset(all_entries, combo_name, channels,
                  val_fraction, test_fraction, seed, output_dir, dryrun=False):
    """Write one band-combination dataset (train / valid / test splits)."""

    train_entries, val_entries, test_entries = session_three_split(
        all_entries, val_fraction, test_fraction, seed
    )

    print(f"\n  [{combo_name}]  "
          f"train={len(train_entries)}  "
          f"val={len(val_entries)}  "
          f"test={len(test_entries)}")

    if dryrun:
        for split_name, split_entries in [
            ("train", train_entries),
            ("val",   val_entries),
            ("test",  test_entries),
        ]:
            cc = {}
            for e in split_entries:
                for line in open(e.label):
                    line = line.strip()
                    if line:
                        cls = int(line.split()[0])
                        cc[cls] = cc.get(cls, 0) + 1
            total = sum(cc.values())
            dist = ", ".join(
                f"cls{c}={cc.get(c,0)} ({100*cc.get(c,0)/total:.1f}%)"
                for c in range(3)
            ) if total else "empty"
            print(f"    {split_name:5s}: {len(split_entries)} frames | {dist}")
        return

    combo_dir = output_dir / combo_name
    ok_total = 0
    skip_total = 0

    for split_name, split_entries in [
        ("train", train_entries),
        ("valid", val_entries),
        ("test",  test_entries),
    ]:
        img_out = combo_dir / split_name / "images"
        lbl_out = combo_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        ok, skip = 0, 0
        for entry in tqdm(split_entries, desc=f"    {combo_name}/{split_name}"):
            try:
                img = build_image(entry, channels)
                # PNG: lossless, preserves exact spectral values
                out_img = img_out / (entry.rgb.stem + ".png")
                cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                shutil.copy(entry.label, lbl_out / entry.label.name)
                ok += 1
            except Exception as ex:
                print(f"\n    ERROR {entry.rgb.name}: {ex}")
                skip += 1

        ok_total += ok
        skip_total += skip

    # Write data.yaml — uses relative path so it works on any machine
    yaml_content = (
        f"# Sweet potato dataset — {combo_name}\n"
        f"# Generated from 2024+2026 merged data\n"
        f"# IMPORTANT: test/ is LOCKED — only use for final paper numbers\n"
        f"path: .   # relative to this file's location — works on any machine\n"
        f"train: train/images\n"
        f"val:   valid/images\n"
        f"test:  test/images\n\n"
        f"nc: {NC}\n"
        f"names: ['Normal', 'Moderate defect', 'Severe defect']\n\n"
        f"# Channels: {channels}\n"
        f"# Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)} frames\n"
    )
    (combo_dir / "data.yaml").write_text(yaml_content)
    print(f"    Written: {ok_total} OK, {skip_total} skipped")
    print(f"    data.yaml -> {combo_dir / 'data.yaml'}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare sweet potato multispectral dataset")
    parser.add_argument("--include2026",  action="store_true",
                        help="Merge 2026 data with 2024")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: processed_data/ or processed_data_2024only/ "
                             "when not using --include2026 and a custom dir isn't set)")
    parser.add_argument("--val_split",  type=float, default=0.15,
                        help="Val fraction per session split (default 0.15)")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Test fraction per session split (default 0.15)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--combos",      nargs="+",
                        default=["RGB", "R_G_NIR1", "R_G_NIR2", "R_NIR1_NIR2"])
    parser.add_argument("--dryrun",      action="store_true",
                        help="Count and verify splits only — write nothing")
    args = parser.parse_args()

    # ── Resolve output directory ──────────────────────────────────────────────
    if args.output_dir:
        out_dir = PROJECT_ROOT / args.output_dir
    else:
        out_dir = OUTPUT_DIR   # default: processed_data/

    print("\n" + "=" * 60)
    print("  SWEET POTATO DATASET PREPARATION")
    print("=" * 60)

    if args.dryrun:
        print("  ** DRY RUN — no files will be written **")

    # Collect
    print("\n[Collecting frames]")
    all_entries = collect_2024()
    if args.include2026:
        all_entries += collect_2026()

    print(f"  Total: {len(all_entries)} frames from "
          f"{'2024+2026' if args.include2026 else '2024 only'}")

    # Class distribution in full set
    cc = {}
    for e in all_entries:
        for line in open(e.label):
            line = line.strip()
            if line:
                cls = int(line.split()[0])
                cc[cls] = cc.get(cls, 0) + 1
    total = sum(cc.values())
    print(f"\n  Combined class distribution ({total} instances):")
    for cls, name in enumerate(CLASS_NAMES):
        n = cc.get(cls, 0)
        print(f"    [{cls}] {name:<22}: {n:4d} ({100*n/total:.1f}%)")

    # Build each combination
    print("\n[Building datasets]")
    for combo in args.combos:
        if combo not in COMBINATIONS:
            print(f"  Unknown combo: {combo}  (valid: {list(COMBINATIONS.keys())})")
            continue
        build_dataset(
            all_entries   = all_entries,
            combo_name    = combo,
            channels      = COMBINATIONS[combo],
            val_fraction  = args.val_split,
            test_fraction = args.test_split,
            seed          = args.seed,
            output_dir    = out_dir,
            dryrun        = args.dryrun,
        )

    if not args.dryrun:
        print("\n" + "=" * 60)
        print("  DONE")
        print("=" * 60)
        print(f"\n  Datasets written to: {out_dir}")
        print("  Next: python models/train.py --config configs/model1/rgb_baseline.yaml --name model1_rgb")
    else:
        print("\n  Dry run complete. Run without --dryrun to write files.")


if __name__ == "__main__":
    main()
