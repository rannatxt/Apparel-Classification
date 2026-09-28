"""
04_sample_dataset.py
=====================
Reads source dataset metadata, applies category mapping, and builds
a stratified candidate list of images to copy.

Strategy:
  1. Load IndoFashion JSON files (train/val/test splits combined for maximum pool)
  2. Build candidate lists per mapped category
  3. Calculate per-category quotas using proportional allocation
  4. Randomly sample (with fixed seed) across categories
  5. For DeepFashion: parse captions, keyword-match, sample
  6. Output: candidates.json — the definitive list before cleaning

Output:
  dataset/metadata/candidates.json
  dataset/metadata/sampling_plan.csv

Usage:
  py -3 scripts/04_sample_dataset.py
  py -3 scripts/04_sample_dataset.py --dry-run
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import json
import random
import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INDOFASHION_DIR,
    DEEPFASHION_DIR,
    DATASET_META,
    TARGET_COUNTS,
    CLASS_PREFIXES,
    DEEPFASHION_WESTERN_KEYWORDS,
    DEEPFASHION_IW_KEYWORDS,
    DEEPFASHION_INDIAN_KEYWORDS,
    SEED,
)

rng = random.Random(SEED)

# ─────────────────────────────────────────────
#  IndoFashion category → final class mapping
#  (must match 03_build_category_mapping.py)
# ─────────────────────────────────────────────
INDO_CLASS_MAP = {
    "saree":                ("Indian",       350),
    "women_kurta":          ("Indian",       300),
    "leggings_and_salwars": ("Indian",       200),
    "lehenga":              ("Indian",       300),
    "kurta_men":            ("Indian",       200),
    "sherwanis":            ("Indian",       150),
    "dhoti_pants":          ("Indo-Western", 100),
    "nehru_jackets":        ("Indo-Western", 100),
}

# Categories explicitly excluded (documented in 03)
INDO_EXCLUDED = {
    "palazzos", "gowns", "dupattas", "blouse",
    "petticoats", "mojaris_women", "mojaris_men",
}


def _all_indofashion_records():
    """Load all records from train/val/test JSON files combined.
    Handles both naming conventions:
      - Real dataset: train_data.json, val_data.json, test_data.json
      - Synthetic test: train.json, val.json
    """
    records = []
    for split in ["train", "val", "test"]:
        json_path = None
        for candidate_name in [f"{split}_data.json", f"{split}.json"]:
            p = INDOFASHION_DIR / candidate_name
            if p.exists():
                json_path = p
                break
        if json_path is None:
            for pattern in [f"{split}_data.json", f"{split}.json"]:
                found = list(INDOFASHION_DIR.rglob(pattern))
                if found:
                    json_path = found[0]
                    break
        if json_path is None:
            print(f"  [WARN] {split}_data.json / {split}.json not found — skipping")
            continue
        recs = []
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                recs = data if isinstance(data, list) else data.get("data", [])
            except json.JSONDecodeError:
                f.seek(0)
                recs = [json.loads(line) for line in f if line.strip()]
        records.extend(recs)
        print(f"  Loaded {len(recs):>6} records from {json_path.name}")
    return records


def _resolve_indofashion_path(record, base_dir: Path) -> Path | None:
    """
    Find the actual image file for an IndoFashion record.
    The dataset stores images under images/<split>/<class>/<filename>
    OR directly at the path given in image_path field.
    """
    img_path_field = record.get("image_path", "")
    if img_path_field:
        candidate = base_dir / img_path_field
        if candidate.exists():
            return candidate
        # Try stripping leading slash
        candidate = base_dir / img_path_field.lstrip("/")
        if candidate.exists():
            return candidate

    # Fallback: search by URL filename
    url = record.get("image_url", "")
    if url:
        filename = url.split("/")[-1].split("?")[0]
        matches = list(base_dir.rglob(filename))
        if matches:
            return matches[0]
    return None


def sample_indofashion(dry_run: bool):
    """Sample from IndoFashion dataset."""
    print("\n  ── IndoFashion Sampling ──")

    if not INDOFASHION_DIR.exists():
        print(f"  [ERROR] {INDOFASHION_DIR} does not exist. Download first.")
        return []

    records = _all_indofashion_records()
    print(f"  Total records loaded: {len(records)}")

    # Group by class label
    # Normalise: strip whitespace, lowercase, spaces → underscore, '&' → '_and_'
    # Handles real-dataset label 'Leggings & Salwar' → 'leggings_and_salwar'
    by_class = defaultdict(list)
    for rec in records:
        raw = rec.get("class_label", "").strip()
        label = raw.lower().replace(" & ", "_and_").replace("&", "_and_").replace(" ", "_")
        by_class[label].append(rec)

    print(f"\n  Available classes:")
    for cls, recs in sorted(by_class.items(), key=lambda x: -len(x[1])):
        status = "✓ mapped" if cls in INDO_CLASS_MAP else ("✗ excluded" if cls in INDO_EXCLUDED else "? unknown")
        print(f"    {cls:<30} {len(recs):>6}  {status}")

    candidates = []
    for orig_class, (final_class, target) in INDO_CLASS_MAP.items():
        pool = by_class.get(orig_class, [])
        if not pool:
            print(f"  [WARN] No records found for '{orig_class}' — skipping")
            continue

        n = min(target, len(pool))
        sampled = rng.sample(pool, n)
        print(f"  {orig_class:<30} pool={len(pool):>6}  target={target:>4}  sampled={n:>4}  → {final_class}")

        for rec in sampled:
            img_path = None if dry_run else _resolve_indofashion_path(rec, INDOFASHION_DIR)
            candidates.append({
                "source_dataset":    "IndoFashion",
                "original_category": orig_class,
                "final_class":       final_class,
                "image_path":        str(img_path) if img_path else "NOT_FOUND",
                "image_url":         rec.get("image_url", ""),
                "product_title":     rec.get("product_title", ""),
                "brand":             rec.get("brand", ""),
                "path_resolved":     img_path is not None,
            })

    return candidates


def _keyword_match(text: str, keywords: list) -> bool:
    """Case-insensitive substring match of any keyword in text."""
    text_l = text.lower()
    return any(kw.lower() in text_l for kw in keywords)


def _classify_caption(caption: str):
    """
    Classify a DeepFashion caption into a final class.
    Returns: 'Western' | 'Indo-Western' | 'Indian' | None
    Priority: Indo-Western > Indian > Western (most specific wins)
    """
    if _keyword_match(caption, DEEPFASHION_IW_KEYWORDS):
        return "Indo-Western"
    if _keyword_match(caption, DEEPFASHION_INDIAN_KEYWORDS):
        return "Indian"
    if _keyword_match(caption, DEEPFASHION_WESTERN_KEYWORDS):
        return "Western"
    return None


def _find_image_files_deepfashion() -> dict:
    """
    Build filename → Path index for all images in the DeepFashion directory.
    Returns dict mapping stem (no extension) → Path.
    """
    index = {}
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in DEEPFASHION_DIR.rglob("*"):
        if p.suffix.lower() in img_exts:
            index[p.stem] = p
    return index


def sample_deepfashion(
    target_western: int,
    target_iw: int,
    target_indian: int,
    dry_run: bool,
):
    """Sample from DeepFashion-MultiModal dataset via caption keyword matching."""
    print("\n  ── DeepFashion-MultiModal Sampling ──")

    if not DEEPFASHION_DIR.exists():
        print(f"  [ERROR] {DEEPFASHION_DIR} does not exist. Download first.")
        return []

    csv_path = DEEPFASHION_DIR / "labels_front.csv"
    if not csv_path.exists():
        candidates_list = list(DEEPFASHION_DIR.rglob("labels_front.csv"))
        if not candidates_list:
            print("  [ERROR] labels_front.csv not found. Dataset may be incomplete.")
            return []
        csv_path = candidates_list[0]

    print(f"  Loading captions from {csv_path.name}...")
    import csv
    caption_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("image_id", "")
            if not key:
                key = Path(row.get("path", "")).stem
            caption_map[key] = row.get("caption", "")

    print(f"  Caption entries: {len(caption_map)}")

    # Build image file index
    if not dry_run:
        print("  Building image file index (may take a moment)...")
        img_index = _find_image_files_deepfashion()
        print(f"  Image files indexed: {len(img_index)}")
    else:
        img_index = {}

    # Classify captions
    classified = defaultdict(list)
    for img_key, caption in caption_map.items():
        cls = _classify_caption(caption)
        if cls:
            classified[cls].append({
                "img_key": img_key,
                "caption": caption,
            })

    print(f"\n  Caption classification results:")
    for cls, items in classified.items():
        print(f"    {cls:<15} {len(items):>6}")

    targets = {
        "Western":      target_western,
        "Indo-Western": target_iw,
        "Indian":       target_indian,
    }

    candidates = []
    for final_class, target in targets.items():
        pool = classified.get(final_class, [])
        if not pool:
            print(f"  [WARN] No DeepFashion candidates for class '{final_class}'")
            continue

        n = min(target, len(pool))
        sampled = rng.sample(pool, n)
        print(f"  {final_class:<15} pool={len(pool):>6}  target={target:>4}  sampled={n:>4}")

        for item in sampled:
            img_key  = item["img_key"]
            img_stem = Path(img_key).stem if img_key else ""
            img_path = img_index.get(img_stem) or img_index.get(img_key)

            candidates.append({
                "source_dataset":    "DeepFashion-MultiModal",
                "original_category": "caption-derived",
                "final_class":       final_class,
                "image_path":        str(img_path) if img_path else "NOT_FOUND",
                "image_url":         "",
                "product_title":     item["caption"][:120],
                "brand":             "",
                "path_resolved":     img_path is not None,
            })

    return candidates


def write_sampling_plan(candidates: list):
    """Write a summary CSV of the sampling plan."""
    plan_path = DATASET_META / "sampling_plan.csv"
    class_cat_counter = defaultdict(Counter)
    for c in candidates:
        class_cat_counter[c["final_class"]][c["original_category"]] += 1

    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "final_class", "source_dataset", "original_category", "sampled_count"
        ])
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "final_class":       c["final_class"],
                "source_dataset":    c["source_dataset"],
                "original_category": c["original_category"],
                "sampled_count":     1,
            })
    print(f"\n  Sampling plan → {plan_path}")
    return plan_path


def main():
    parser = argparse.ArgumentParser(description="Sample candidate images from source datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the sampling plan without resolving actual file paths")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 04 — Sample Dataset")
    print("=" * 60)
    print(f"  Targets: {TARGET_COUNTS}")
    print(f"  Seed:    {SEED}")
    if args.dry_run:
        print("  [DRY RUN] File paths will not be resolved.")

    DATASET_META.mkdir(parents=True, exist_ok=True)

    # ── IndoFashion ───────────────────────────────────────────
    indo_candidates = sample_indofashion(dry_run=args.dry_run)

    # Count what IndoFashion contributes per class
    indo_by_class = Counter(c["final_class"] for c in indo_candidates)
    print(f"\n  IndoFashion contributions: {dict(indo_by_class)}")

    # ── DeepFashion (fills remaining slots) ───────────────────
    remaining_western   = max(0, TARGET_COUNTS["Western"]      - indo_by_class.get("Western", 0))
    remaining_iw        = max(0, TARGET_COUNTS["Indo-Western"]  - indo_by_class.get("Indo-Western", 0))
    remaining_indian    = max(0, TARGET_COUNTS["Indian"]        - indo_by_class.get("Indian", 0))

    print(f"\n  Remaining slots for DeepFashion:")
    print(f"    Indian:       {remaining_indian}")
    print(f"    Western:      {remaining_western}")
    print(f"    Indo-Western: {remaining_iw}")

    deep_candidates = sample_deepfashion(
        target_western=remaining_western,
        target_iw=remaining_iw,
        target_indian=remaining_indian,
        dry_run=args.dry_run,
    )

    all_candidates = indo_candidates + deep_candidates

    # ── Final summary ─────────────────────────────────────────
    final_counts = Counter(c["final_class"] for c in all_candidates)
    print(f"\n  Final candidate counts:")
    for cls, cnt in final_counts.items():
        target = TARGET_COUNTS.get(cls, "?")
        diff   = cnt - (target if isinstance(target, int) else 0)
        flag   = f"  ({'+'if diff >= 0 else ''}{diff})" if isinstance(target, int) else ""
        print(f"    {cls:<15} {cnt:>6} / {target}{flag}")

    resolved = sum(1 for c in all_candidates if c.get("path_resolved"))
    not_resolved = len(all_candidates) - resolved
    print(f"\n  Paths resolved:     {resolved}")
    print(f"  Paths NOT resolved: {not_resolved}")
    if not_resolved > 0 and not args.dry_run:
        print("  [WARN] Some images could not be found on disk.")
        print("         They will be filtered out during cleaning (step 05).")

    # ── Save candidates list ───────────────────────────────────
    candidates_path = DATASET_META / "candidates.json"
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2)
    print(f"\n  Candidates saved → {candidates_path}")

    write_sampling_plan(all_candidates)

    print("\n[DONE] Sampling complete.")


if __name__ == "__main__":
    main()
