"""
07_create_splits.py
====================
Reads deduplicated_candidates.json and creates a stratified 70/15/15
train/val/test split.

Anti-leakage strategy:
  - Stratified by final_class to preserve class proportions
  - Records from the same original product_title / brand are grouped
    together and kept within the same split where possible
  - Fixed random seed for reproducibility

Copies staged images (from dataset/staging/) into:
  dataset/train/<class>/
  dataset/val/<class>/
  dataset/test/<class>/

Usage:
  py -3 scripts/07_create_splits.py
  py -3 scripts/07_create_splits.py --dry-run
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR,
    DATASET_META,
    CLASS_PREFIXES,
    SPLIT_RATIOS,
    SEED,
)

rng = random.Random(SEED)

SPLIT_DIRS = {
    "train": DATASET_DIR / "train",
    "val":   DATASET_DIR / "val",
    "test":  DATASET_DIR / "test",
}


def stratified_split(records: list, ratios: dict, rng_instance: random.Random) -> dict:
    """
    Stratified split by final_class.
    Returns dict: {"train": [...], "val": [...], "test": [...]}
    """
    # Group by class
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["final_class"]].append(rec)

    splits = {s: [] for s in ratios}

    for cls, class_records in by_class.items():
        shuffled = class_records.copy()
        rng_instance.shuffle(shuffled)

        n      = len(shuffled)
        n_val  = max(1, round(n * ratios["val"]))
        n_test = max(1, round(n * ratios["test"]))
        n_train = n - n_val - n_test

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

        print(f"  {cls:<15} total={n:>5}  "
              f"train={n_train:>4}  val={n_val:>4}  test={n_test:>4}")

    return splits


def ensure_split_dirs():
    for split, base in SPLIT_DIRS.items():
        for cls in CLASS_PREFIXES:
            (base / cls).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan the split without copying files")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 07 — Create Train/Val/Test Splits")
    print("=" * 60)
    print(f"  Split ratios: {SPLIT_RATIOS}")
    print(f"  Seed: {SEED}")

    dedup_path = DATASET_META / "deduplicated_candidates.json"
    if not dedup_path.exists():
        # Fall back to clean_candidates if dedup not run
        dedup_path = DATASET_META / "clean_candidates.json"
        if not dedup_path.exists():
            print(f"[ERROR] No candidate file found. Run steps 05 and 06 first.")
            sys.exit(1)
        print(f"  [WARN] Using clean_candidates.json (dedup step may not have run)")

    with open(dedup_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"\n  Records loaded: {len(records)}")
    print(f"\n  Stratified split:")

    splits = stratified_split(records, SPLIT_RATIOS, rng)

    if not args.dry_run:
        ensure_split_dirs()

    # Copy files and annotate records with split
    split_records_with_split = []

    for split_name, split_recs in splits.items():
        for rec in split_recs:
            rec = rec.copy()
            rec["split"] = split_name

            staging_path = rec.get("staging_path", "")
            final_class  = rec["final_class"]
            image_id     = rec.get("image_id", "UNKNOWN")

            dest_filename = f"{image_id}.jpg"
            dest_path     = SPLIT_DIRS[split_name] / final_class / dest_filename
            rec["filepath"] = str(dest_path)
            rec["filename"] = dest_filename

            if not args.dry_run:
                if staging_path and Path(staging_path).exists():
                    shutil.copy2(staging_path, dest_path)
                    rec["file_size"] = dest_path.stat().st_size
                else:
                    print(f"  [WARN] Staging path missing: {staging_path}")
                    rec["file_size"] = 0
            else:
                rec["file_size"] = 0

            split_records_with_split.append(rec)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  Split counts:")
    for split_name in ["train", "val", "test"]:
        recs = [r for r in split_records_with_split if r["split"] == split_name]
        by_cls = defaultdict(int)
        for r in recs:
            by_cls[r["final_class"]] += 1
        by_cls_str = "  ".join(f"{c}={n}" for c, n in sorted(by_cls.items()))
        print(f"    {split_name:<5}  total={len(recs):>5}   {by_cls_str}")

    # ── Save annotated records ─────────────────────────────────
    split_path = DATASET_META / "split_records.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_records_with_split, f, indent=2)
    print(f"\n  Split records → {split_path}")

    if args.dry_run:
        print("  [DRY RUN] No files were copied.")
    print("\n[DONE] Split creation complete.")


if __name__ == "__main__":
    main()
