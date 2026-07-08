"""
fix_data_yaml_paths.py
Run once on any machine after copying processed_data from another machine.
Writes the correct absolute path for THIS machine into each data.yaml so
YOLO can find the images. Safe to re-run — idempotent.
"""
import re
from pathlib import Path

fixed = 0
for p in sorted(Path("processed_data").rglob("data.yaml")):
    abs_path = p.parent.resolve().as_posix()   # e.g. D:/HA/sweetp/processed_data/RGB
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
