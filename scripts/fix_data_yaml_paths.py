"""
fix_data_yaml_paths.py
Run once on any machine after copying processed_data from another machine.
Replaces hardcoded absolute paths in data.yaml with relative '.' so YOLO
finds images correctly regardless of where the folder is placed.
"""
import re
from pathlib import Path

fixed = 0
for p in Path("processed_data").rglob("data.yaml"):
    txt = p.read_text()
    new_txt = re.sub(r"^path:.*$", "path: .", txt, flags=re.MULTILINE)
    if new_txt != txt:
        p.write_text(new_txt)
        print(f"Fixed: {p}")
        fixed += 1
    else:
        print(f"Already OK: {p}")

print(f"\nDone — {fixed} file(s) updated.")
