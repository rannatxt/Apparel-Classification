"""
02_inspect_datasets.py
========================
Inspects the downloaded datasets and reports:
  - Directory structure
  - JSON metadata files
  - Category names and counts
  - Image formats and dimensions (sampled)
  - Corrupted / unreadable images

Usage:
  py -3 scripts/02_inspect_datasets.py
  py -3 scripts/02_inspect_datasets.py --sample 100   # inspect N images per dataset
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import json
import sys
import os
import random
from pathlib import Path
from collections import Counter, defaultdict

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INDOFASHION_DIR,
    DEEPFASHION_DIR,
    DATASET_REPORTS,
    SEED,
)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed — image inspection will be skipped.")
    print("       Run: pip install Pillow")


def _safe_open(path: Path):
    """Try to open an image. Return (Image, None) or (None, error_str)."""
    if not PIL_AVAILABLE:
        return None, "Pillow not available"
    try:
        img = Image.open(path)
        img.verify()          # detect truncated files
        img = Image.open(path)  # reopen after verify
        img.load()
        return img, None
    except Exception as e:
        return None, str(e)


def inspect_indofashion(sample_n: int):
    """Inspect the IndoFashion dataset."""
    print("\n" + "=" * 60)
    print("  DATASET 1 — IndoFashion")
    print("=" * 60)

    if not INDOFASHION_DIR.exists():
        print(f"[ERROR] Directory not found: {INDOFASHION_DIR}")
        print("  Run: py -3 scripts/01_download_datasets.py first.")
        return {}

    # ── Directory structure ────────────────────────────────────
    print(f"\nRoot: {INDOFASHION_DIR}")
    for item in sorted(INDOFASHION_DIR.iterdir()):
        if item.is_dir():
            n = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  [DIR ] {item.name}  ({n} files)")
        else:
            print(f"  [FILE] {item.name}  ({item.stat().st_size / 1024:.1f} KB)")

    # ── JSON metadata ──────────────────────────────────────────
    # Real dataset uses: train_data.json, val_data.json, test_data.json
    # Synthetic test data uses: train.json, val.json
    results = {}
    for split in ["train", "val", "test"]:
        json_path = None
        for candidate_name in [f"{split}_data.json", f"{split}.json"]:
            p = INDOFASHION_DIR / candidate_name
            if p.exists():
                json_path = p
                break
        if json_path is None:
            # rglob fallback
            for pattern in [f"{split}_data.json", f"{split}.json"]:
                found = list(INDOFASHION_DIR.rglob(pattern))
                if found:
                    json_path = found[0]
                    break
        if json_path is None:
            print(f"\n[WARN] {split}_data.json / {split}.json not found")
            continue

        print(f"\n  ── {json_path.name} ({json_path.stat().st_size / 1024 / 1024:.1f} MB) ──")
        records = []
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("data", [])
            except json.JSONDecodeError:
                f.seek(0)
                records = [json.loads(line) for line in f if line.strip()]
        print(f"  Records: {len(records)}")
        if records:
            print(f"  Sample entry keys: {list(records[0].keys())}")

        class_counter = Counter(r.get("class_label", "UNKNOWN") for r in records)
        print(f"\n  Class distribution ({split}):")
        for cls, cnt in sorted(class_counter.items(), key=lambda x: -x[1]):
            bar = "█" * (cnt // 200)
            print(f"    {cls:<30} {cnt:>6}  {bar}")
        results[split] = {"records": records, "class_counter": class_counter}

    # ── Image folder scan ──────────────────────────────────────
    img_dirs = [d for d in INDOFASHION_DIR.rglob("*") if d.is_dir() and
                any(f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    for f in d.iterdir() if f.is_file())]

    if img_dirs:
        print(f"\n  Image directories found: {len(img_dirs)}")
        all_imgs = []
        fmt_counter = Counter()
        for d in img_dirs:
            imgs = [f for f in d.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
            all_imgs.extend(imgs)
            for img in imgs:
                fmt_counter[img.suffix.lower()] += 1
        print(f"  Total image files found: {len(all_imgs)}")
        print(f"  Formats: {dict(fmt_counter)}")

        if PIL_AVAILABLE and all_imgs:
            rng = random.Random(SEED)
            sample = rng.sample(all_imgs, min(sample_n, len(all_imgs)))
            widths, heights, corrupt = [], [], []
            for p in sample:
                img, err = _safe_open(p)
                if err:
                    corrupt.append((str(p), err))
                else:
                    widths.append(img.width)
                    heights.append(img.height)
            if widths:
                print(f"\n  Image stats (sample={len(sample)}):")
                print(f"    Width  min={min(widths)} max={max(widths)} avg={sum(widths)//len(widths)}")
                print(f"    Height min={min(heights)} max={max(heights)} avg={sum(heights)//len(heights)}")
            if corrupt:
                print(f"\n  Corrupted images in sample: {len(corrupt)}")
                for p, e in corrupt[:5]:
                    print(f"    {p}: {e}")

    return results


def inspect_deepfashion(sample_n: int):
    """Inspect the DeepFashion-MultiModal dataset."""
    print("\n" + "=" * 60)
    print("  DATASET 2 — DeepFashion-MultiModal")
    print("=" * 60)

    if not DEEPFASHION_DIR.exists():
        print(f"[ERROR] Directory not found: {DEEPFASHION_DIR}")
        print("  Run: py -3 scripts/01_download_datasets.py first.")
        return {}

    print(f"\nRoot: {DEEPFASHION_DIR}")
    for item in sorted(DEEPFASHION_DIR.iterdir()):
        if item.is_dir():
            n = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  [DIR ] {item.name}  ({n} files)")
        else:
            print(f"  [FILE] {item.name}  ({item.stat().st_size / 1024:.1f} KB)")

    # ── labels_front.csv ──────────────────────────────────────────
    results = {}
    csv_path = DEEPFASHION_DIR / "labels_front.csv"
    if not csv_path.exists():
        candidates = list(DEEPFASHION_DIR.rglob("labels_front.csv"))
        if candidates:
            csv_path = candidates[0]

    if csv_path.exists():
        print(f"\n  ── labels_front.csv ({csv_path.stat().st_size / 1024 / 1024:.1f} MB) ──")
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            entries = sum(1 for _ in reader)
        print(f"  Entries: {entries}")
        print(f"  Header columns: {header}")
        results["captions"] = {} # Mock dict for compatibility if needed downstream
    else:
        print("\n  [WARN] labels_front.csv not found.")

    # ── texture_ann / shape_ann ────────────────────────────────
    for ann_name in ["texture_ann", "shape_ann", "annos"]:
        ann_dir = DEEPFASHION_DIR / ann_name
        if ann_dir.exists():
            files = list(ann_dir.iterdir())
            print(f"\n  ── {ann_name}/ ({len(files)} files) ──")
            for f in files[:10]:
                print(f"    {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    # ── Image folders ──────────────────────────────────────────
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_imgs = [f for f in DEEPFASHION_DIR.rglob("*") if f.suffix.lower() in img_exts]
    fmt_counter = Counter(f.suffix.lower() for f in all_imgs)
    print(f"\n  Total image files: {len(all_imgs)}")
    print(f"  Formats: {dict(fmt_counter)}")

    if PIL_AVAILABLE and all_imgs:
        rng = random.Random(SEED)
        sample = rng.sample(all_imgs, min(sample_n, len(all_imgs)))
        widths, heights, corrupt = [], [], []
        for p in sample:
            img, err = _safe_open(p)
            if err:
                corrupt.append((str(p), err))
            else:
                widths.append(img.width)
                heights.append(img.height)
        if widths:
            print(f"\n  Image stats (sample={len(sample)}):")
            print(f"    Width  min={min(widths)} max={max(widths)} avg={sum(widths)//len(widths)}")
            print(f"    Height min={min(heights)} max={max(heights)} avg={sum(heights)//len(heights)}")
        if corrupt:
            print(f"\n  Corrupted images in sample: {len(corrupt)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inspect downloaded datasets")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of images to sample per dataset for dimension inspection")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 02 — Inspect Datasets")
    print("=" * 60)

    indo  = inspect_indofashion(args.sample)
    deep  = inspect_deepfashion(args.sample)

    DATASET_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = DATASET_REPORTS / "inspection_summary.txt"

    # Simple text summary
    lines = ["DATASET INSPECTION SUMMARY\n", "=" * 40 + "\n"]
    total_indo = sum(
        sum(v["class_counter"].values())
        for v in indo.values() if isinstance(v, dict) and "class_counter" in v
    )
    lines.append(f"IndoFashion  total records: {total_indo}\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n  Inspection summary saved → {report_path}")

    print("\n[DONE] Inspection complete.")


if __name__ == "__main__":
    main()
