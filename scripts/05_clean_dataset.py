"""
05_clean_dataset.py
====================
Reads candidates.json, validates every candidate image, and copies
clean images into the final dataset directory with standardised names.

Checks performed:
  1. File exists on disk
  2. File can be opened by Pillow
  3. Image is not corrupt (verify + load)
  4. Image dimensions ≥ MIN_DIMENSION on both axes
  5. Aspect ratio within MAX_ASPECT_RATIO
  6. Convert to RGB (handles RGBA, grayscale, palette)
  7. Resize to TARGET_SIZE using LANCZOS (letterbox — pad to preserve aspect ratio)
  8. Save as JPEG with JPEG_QUALITY

Outputs:
  dataset/train|val|test/<class>/<prefix>_<ID>.jpg   (created in step 07)
  dataset/metadata/clean_candidates.json             (passed to step 06)
  dataset/metadata/corrupted_images.csv

NOTE: This step only VALIDATES and PREPARES the images.
      The actual split copy happens in step 07.
      Images are written to a staging area: dataset/staging/<class>/

Usage:
  py -3 scripts/05_clean_dataset.py
  py -3 scripts/05_clean_dataset.py --dry-run
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR,
    DATASET_META,
    CLASS_PREFIXES,
    TARGET_SIZE,
    OUTPUT_FORMAT,
    JPEG_QUALITY,
    MIN_DIMENSION,
    MAX_ASPECT_RATIO,
)

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[ERROR] Pillow is required. Run: pip install Pillow")
    sys.exit(1)

STAGING_DIR = DATASET_DIR / "staging"


def make_staging_dirs():
    for cls in CLASS_PREFIXES:
        (STAGING_DIR / cls).mkdir(parents=True, exist_ok=True)


def letterbox_resize(img: Image.Image, size: tuple) -> Image.Image:
    """
    Resize image to target size while preserving aspect ratio.
    Pads remaining area with black.
    """
    img.thumbnail(size, Image.LANCZOS)
    padded = Image.new("RGB", size, (0, 0, 0))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    padded.paste(img, offset)
    return padded


def validate_and_prepare(record: dict, dry_run: bool) -> tuple:
    """
    Validate one candidate image.
    Returns: (clean_record | None, error_dict | None)
    """
    src_path = Path(record.get("image_path", ""))
    final_class = record.get("final_class", "")

    # ── 1. File existence ──────────────────────────────────────
    if not src_path.exists() or str(src_path) == "NOT_FOUND":
        return None, {"reason": "file_not_found", "path": str(src_path)}

    # ── 2. Open image ─────────────────────────────────────────
    try:
        with Image.open(src_path) as img:
            img.verify()
    except Exception as e:
        return None, {"reason": f"corrupt_verify: {e}", "path": str(src_path)}

    try:
        img = Image.open(src_path)
        img.load()
    except Exception as e:
        return None, {"reason": f"corrupt_load: {e}", "path": str(src_path)}

    # ── 3. Dimensions ─────────────────────────────────────────
    w, h = img.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        return None, {"reason": f"too_small ({w}×{h})", "path": str(src_path)}

    # ── 4. Aspect ratio ───────────────────────────────────────
    ar = max(w, h) / min(w, h)
    if ar > MAX_ASPECT_RATIO:
        return None, {"reason": f"extreme_aspect_ratio ({ar:.2f})", "path": str(src_path)}

    if dry_run:
        clean_record = record.copy()
        clean_record.update({
            "staging_path": "DRY_RUN",
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 4),
            "original_format": img.format or "UNKNOWN",
        })
        return clean_record, None

    # ── 5. Convert to RGB ─────────────────────────────────────
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # ── 6. Letterbox resize ───────────────────────────────────
    prepared = letterbox_resize(img, TARGET_SIZE)

    clean_record = record.copy()
    clean_record.update({
        "staging_path":   "PENDING",   # filled during actual copy
        "width":          prepared.width,
        "height":         prepared.height,
        "aspect_ratio":   round(w / h, 4),
        "original_format": img.format or "UNKNOWN",
        "_pil_image":     prepared,    # temporary, not serialised
    })
    return clean_record, None


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare candidate images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without writing any files")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 05 — Clean Dataset")
    print("=" * 60)

    candidates_path = DATASET_META / "candidates.json"
    if not candidates_path.exists():
        print(f"[ERROR] {candidates_path} not found. Run step 04 first.")
        sys.exit(1)

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates)} candidates")

    if not args.dry_run:
        make_staging_dirs()

    clean_records  = []
    corrupted      = []
    class_counters = {cls: 0 for cls in CLASS_PREFIXES}

    for i, record in enumerate(candidates):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(candidates)}...", end="\r")

        clean_rec, error = validate_and_prepare(record, dry_run=args.dry_run)

        if error:
            error["source_dataset"]    = record.get("source_dataset", "")
            error["original_category"] = record.get("original_category", "")
            error["final_class"]       = record.get("final_class", "")
            corrupted.append(error)
            continue

        final_class  = clean_rec["final_class"]
        prefix       = CLASS_PREFIXES.get(final_class, "UNK")
        class_counters[final_class] = class_counters.get(final_class, 0) + 1
        img_id       = f"{prefix}_{class_counters[final_class]:06d}"

        clean_rec["image_id"] = img_id

        # Stage the prepared image
        if not args.dry_run and "_pil_image" in clean_rec:
            pil_img      = clean_rec.pop("_pil_image")
            staging_path = STAGING_DIR / final_class / f"{img_id}.jpg"
            pil_img.save(staging_path, format="JPEG", quality=JPEG_QUALITY,
                         subsampling=0, optimize=True)
            clean_rec["staging_path"] = str(staging_path)
        else:
            clean_rec.pop("_pil_image", None)

        clean_records.append(clean_rec)

    print(f"\n\n  Results:")
    print(f"    Total candidates : {len(candidates)}")
    print(f"    Clean            : {len(clean_records)}")
    print(f"    Corrupted/invalid: {len(corrupted)}")
    print(f"\n  Clean counts by class:")
    for cls, cnt in class_counters.items():
        print(f"    {cls:<15} {cnt:>6}")

    # ── Save outputs ───────────────────────────────────────────
    DATASET_META.mkdir(parents=True, exist_ok=True)

    clean_path = DATASET_META / "clean_candidates.json"
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2)
    print(f"\n  Clean candidates → {clean_path}")

    corrupt_csv = DATASET_META / "corrupted_images.csv"
    with open(corrupt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_dataset", "original_category", "final_class", "reason", "path"
        ])
        writer.writeheader()
        writer.writerows(corrupted)
    print(f"  Corrupted list   → {corrupt_csv}")

    if args.dry_run:
        print("\n  [DRY RUN] No images were written to disk.")
    print("\n[DONE] Cleaning complete.")


if __name__ == "__main__":
    main()
