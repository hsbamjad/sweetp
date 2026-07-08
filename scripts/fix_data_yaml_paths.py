"""
fix_data_yaml_paths.py
Run once on any machine after copying processed_data from another machine.
Writes the correct absolute path for THIS machine into each data.yaml so
YOLO can find the images. Safe to re-run — idempotent.

Usage:
    python scripts/fix_data_yaml_paths.py                        # fixes processed_data/
    python scripts/fix_data_yaml_paths.py processed_data_2024only  # fixes any other dir
"""
import re
import sys
from pathlib import Path

target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed_data")

if not target_dir.exists():
    print(f"ERROR: '{target_dir}' does not exist. Did you run prepare_dataset.py first?")
    sys.exit(1)

fixed = 0
for p in sorted(target_dir.rglob("data.yaml")):
    abs_path = p.parent.resolve().as_posix()
    txt = p.read_text()
    new_txt = re.sub(r"^path:.*$", f"path: {abs_path}", txt, flags=re.MULTILINE)
    if new_txt != txt:
        p.write_text(new_txt)
        print(f"  Fixed : {p}")
        print(f"    path -> {abs_path}")
        fixed += 1
    else:
        print(f"  OK    : {p}  (already correct)")

print(f"\nDone — {fixed} file(s) updated.")
