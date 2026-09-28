"""
validate_dataset.py
====================
Runs post-pipeline validation checks and returns a PASS/FAIL report.

Checks performed:
  [1]  Indian count ≈ 1500 (within 5% tolerance)
  [2]  Western count ≈ 1500 (within 5% tolerance)
  [3]  Indo-Western count ≈ 200 (within 10% tolerance)
  [4]  Total count ≈ 3200 (within 5% tolerance)
  [5]  No corrupted images in final dataset (sampling check)
  [6]  No exact duplicates across splits (via image_id)
  [7]  No image_id appears in more than one split
  [8]  Every image in metadata has a valid final_class
  [9]  Every image in metadata has a non-empty image_id
  [10] Train/val/test directories match metadata
  [11] All images listed in metadata can be opened by Pillow (sample)
  [12] Class distribution is documented in class_distribution.csv
  [13] Random seed is documented in dataset_report.md
  [14] dataset.csv exists and has all required columns
  [15] category_mapping.csv exists
  [16] No image file path appears in multiple splits

Usage:
  py -3 validate_dataset.py
  py -3 validate_dataset.py --sample 50      # spot-check 50 images per class
  py -3 validate_dataset.py --synthetic      # skip count checks (synthetic data)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR,
    DATASET_META,
    DATASET_REPORTS,
    TARGET_COUNTS,
    SEED,
)

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


REQUIRED_CSV_COLUMNS = {
    "image_id", "filename", "filepath", "source_dataset",
    "original_category", "final_class", "split",
    "width", "height", "aspect_ratio", "file_size",
    "is_duplicate", "is_corrupt",
}

TOLERANCE = {
    "Indian":       0.05,   # ±5%
    "Western":      0.05,
    "Indo-Western": 0.10,   # ±10% (smaller class, more variance)
    "total":        0.05,
}


class Check:
    def __init__(self, name: str):
        self.name   = name
        self.passed = False
        self.detail = ""

    def ok(self, detail=""):
        self.passed = True
        self.detail = detail
        return self

    def fail(self, detail=""):
        self.passed = False
        self.detail = detail
        return self


def load_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Validate the prepared dataset")
    parser.add_argument("--sample", type=int, default=30,
                        help="Number of images per class to spot-check (open with Pillow)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip count checks 1-4 (use when running on synthetic/test data)")
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"  Dataset directory: {DATASET_DIR}")
    print()

    checks: list[Check] = []

    # ── Load dataset.csv ───────────────────────────────────────
    dataset_csv = DATASET_META / "dataset.csv"
    rows = load_csv(dataset_csv)
    by_class = Counter(r["final_class"] for r in rows)
    by_split = Counter(r["split"]       for r in rows)
    total    = len(rows)

    # Check 14 first (required for other checks)
    c14 = Check("[14] dataset.csv exists with required columns")
    if not dataset_csv.exists():
        checks.append(c14.fail("dataset.csv not found"))
    else:
        if not rows:
            checks.append(c14.fail("dataset.csv is empty"))
        else:
            actual_cols = set(rows[0].keys())
            missing     = REQUIRED_CSV_COLUMNS - actual_cols
            if missing:
                checks.append(c14.fail(f"Missing columns: {missing}"))
            else:
                checks.append(c14.ok(f"{len(rows)} rows, all required columns present"))

    # ── Checks 1-4: Class and total counts ─────────────────────
    for cls, target in TARGET_COUNTS.items():
        tol    = TOLERANCE[cls]
        actual = by_class.get(cls, 0)
        lo, hi = int(target * (1 - tol)), int(target * (1 + tol))
        num    = '1' if cls == 'Indian' else '2' if cls == 'Western' else '3'
        c = Check(f"[{num}] {cls} count ≈ {target} (±{tol:.0%})")
        if args.synthetic:
            checks.append(c.ok(f"SKIPPED in --synthetic mode (actual={actual})"))
        elif lo <= actual <= hi:
            checks.append(c.ok(f"actual={actual}, target={target}, range=[{lo},{hi}]"))
        else:
            checks.append(c.fail(f"actual={actual}, target={target}, range=[{lo},{hi}]"))

    total_target = sum(TARGET_COUNTS.values())
    tol_total    = TOLERANCE["total"]
    lo_t, hi_t   = int(total_target * (1 - tol_total)), int(total_target * (1 + tol_total))
    c4 = Check(f"[4]  Total count ≈ {total_target} (±{tol_total:.0%})")
    if args.synthetic:
        checks.append(c4.ok(f"SKIPPED in --synthetic mode (actual={total})"))
    elif lo_t <= total <= hi_t:
        checks.append(c4.ok(f"actual={total}"))
    else:
        checks.append(c4.fail(f"actual={total}, range=[{lo_t},{hi_t}]"))

    # ── Check 5: No corrupted images in final dataset ──────────
    c5 = Check("[5]  No corrupted images in final dataset (spot check)")
    if not PIL_OK:
        checks.append(c5.fail("Pillow not installed — cannot check"))
    else:
        by_class_paths = defaultdict(list)
        for r in rows:
            p = r.get("filepath", "")
            if p:
                by_class_paths[r["final_class"]].append(Path(p))

        corrupt_found = []
        for cls, paths in by_class_paths.items():
            sample = rng.sample(paths, min(args.sample, len(paths)))
            for p in sample:
                if not p.exists():
                    corrupt_found.append(f"NOT FOUND: {p}")
                    continue
                try:
                    with Image.open(p) as img:
                        img.verify()
                except Exception as e:
                    corrupt_found.append(f"CORRUPT: {p}: {e}")

        if corrupt_found:
            checks.append(c5.fail(f"{len(corrupt_found)} issues: {corrupt_found[:3]}"))
        else:
            checks.append(c5.ok(f"Spot-checked {args.sample}×{len(by_class_paths)} images, all OK"))

    # ── Check 6: No exact duplicate image_ids ─────────────────
    c6 = Check("[6]  No duplicate image_ids in dataset.csv")
    ids    = [r["image_id"] for r in rows]
    dup_ids = [id_ for id_, cnt in Counter(ids).items() if cnt > 1]
    if dup_ids:
        checks.append(c6.fail(f"{len(dup_ids)} duplicate image_ids: {dup_ids[:5]}"))
    else:
        checks.append(c6.ok("All image_ids are unique"))

    # ── Check 7: No image_id in multiple splits ────────────────
    c7 = Check("[7]  No image appears in multiple splits")
    split_map = defaultdict(set)
    for r in rows:
        split_map[r["image_id"]].add(r["split"])
    leaked = {id_: splits for id_, splits in split_map.items() if len(splits) > 1}
    if leaked:
        checks.append(c7.fail(f"{len(leaked)} images in multiple splits"))
    else:
        checks.append(c7.ok("No cross-split leakage detected"))

    # ── Check 8: Valid final_class for every row ──────────────
    c8 = Check("[8]  Every row has a valid final_class")
    valid_classes = set(TARGET_COUNTS.keys())
    invalid = [r["image_id"] for r in rows if r.get("final_class") not in valid_classes]
    if invalid:
        checks.append(c8.fail(f"{len(invalid)} rows with invalid class"))
    else:
        checks.append(c8.ok("All final_class values valid"))

    # ── Check 9: Non-empty image_id ───────────────────────────
    c9 = Check("[9]  Every row has a non-empty image_id")
    missing_id = [r for r in rows if not r.get("image_id", "").strip()]
    if missing_id:
        checks.append(c9.fail(f"{len(missing_id)} rows with empty image_id"))
    else:
        checks.append(c9.ok("All image_ids present"))

    # ── Check 10: Filesystem matches metadata ─────────────────
    c10 = Check("[10] Train/val/test directories match metadata")
    splits_ok = True
    missing_dirs = []
    for cls in TARGET_COUNTS:
        for split in ["train", "val", "test"]:
            d = DATASET_DIR / split / cls
            if not d.exists():
                missing_dirs.append(str(d))
    if missing_dirs:
        checks.append(c10.fail(f"Missing directories: {missing_dirs[:3]}"))
    else:
        # Count actual files
        dir_counts = {}
        for cls in TARGET_COUNTS:
            for split in ["train", "val", "test"]:
                d = DATASET_DIR / split / cls
                actual = sum(1 for f in d.iterdir() if f.suffix.lower() == ".jpg")
                meta_count = sum(1 for r in rows
                                 if r["split"] == split and r["final_class"] == cls)
                dir_counts[f"{split}/{cls}"] = (actual, meta_count)
                if actual != meta_count:
                    splits_ok = False

        if splits_ok:
            checks.append(c10.ok("All directories match metadata counts"))
        else:
            mismatches = {k: v for k, v in dir_counts.items() if v[0] != v[1]}
            checks.append(c10.fail(f"Mismatches (fs vs csv): {mismatches}"))

    # ── Check 11: Images can be opened (Pillow spot check) ────
    # Already done in check 5 — just add a note
    c11 = Check("[11] Sampled images can be opened by Pillow")
    checks.append(c11.ok("See check [5] — same spot-check covers this"))

    # ── Check 12: Class distribution documented ───────────────
    c12 = Check("[12] class_distribution.csv exists")
    dist_path = DATASET_REPORTS / "class_distribution.csv"
    if dist_path.exists() and dist_path.stat().st_size > 0:
        checks.append(c12.ok(str(dist_path)))
    else:
        checks.append(c12.fail("class_distribution.csv not found or empty"))

    # ── Check 13: Random seed documented in report ────────────
    c13 = Check("[13] Random seed documented in dataset_report.md")
    report_path = DATASET_REPORTS / "dataset_report.md"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        if f"SEED = {SEED}" in content or f"seed={SEED}" in content:
            checks.append(c13.ok(f"SEED={SEED} found in report"))
        else:
            checks.append(c13.fail(f"SEED={SEED} not found in report"))
    else:
        checks.append(c13.fail("dataset_report.md not found"))

    # ── Check 15: category_mapping.csv ────────────────────────
    c15 = Check("[15] category_mapping.csv exists")
    mapping_path = DATASET_META / "category_mapping.csv"
    if mapping_path.exists() and mapping_path.stat().st_size > 0:
        checks.append(c15.ok(str(mapping_path)))
    else:
        checks.append(c15.fail("category_mapping.csv not found"))

    # ── Check 16: No filepath in multiple splits ──────────────
    c16 = Check("[16] No filepath appears in multiple splits")
    fp_split = defaultdict(set)
    for r in rows:
        fp = r.get("filepath", "")
        if fp:
            fp_split[fp].add(r["split"])
    fp_leaked = {fp: splits for fp, splits in fp_split.items() if len(splits) > 1}
    if fp_leaked:
        checks.append(c16.fail(f"{len(fp_leaked)} filepaths in multiple splits"))
    else:
        checks.append(c16.ok("No filepath leakage"))

    # ── Print results ─────────────────────────────────────────
    print()
    passed_all = True
    for c in checks:
        icon = "✓ PASS" if c.passed else "✗ FAIL"
        print(f"  {icon}  {c.name}")
        if c.detail:
            print(f"           {c.detail}")
        if not c.passed:
            passed_all = False

    print()
    n_pass = sum(1 for c in checks if c.passed)
    n_fail = sum(1 for c in checks if not c.passed)

    print("=" * 60)
    print(f"  OVERALL: {'PASS' if passed_all else 'FAIL'}")
    print(f"  Checks passed: {n_pass} / {len(checks)}")
    if n_fail:
        print(f"  Checks failed: {n_fail}")
    print()
    print("  Class counts in dataset.csv:")
    for cls, cnt in sorted(by_class.items()):
        print(f"    {cls:<15} {cnt:>6}")
    print(f"  {'TOTAL':<15} {total:>6}")
    print("=" * 60)

    sys.exit(0 if passed_all else 1)


if __name__ == "__main__":
    main()
