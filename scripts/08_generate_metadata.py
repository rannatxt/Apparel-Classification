"""
08_generate_metadata.py
========================
Generates the final metadata CSV files:
  - dataset/metadata/dataset.csv     (all images)
  - dataset/metadata/train.csv
  - dataset/metadata/val.csv
  - dataset/metadata/test.csv
  - dataset/metadata/class_distribution.csv
  - dataset/metadata/image_statistics.csv
  - dataset/metadata/edge_case_candidates.csv

Edge-case identification:
  Because edge-case labels cannot be determined automatically with
  high reliability without a trained classifier, this script:
  1. Sets edge_case = "normal" by default for all images
  2. Flags potential edge cases as "candidate" based on heuristics:
       - Indo-Western class images → tagged "hybrid" (they are by definition edge cases)
       - Images with extreme aspect ratios (0.8 < AR < 1.25) → "visually_similar"
       - Images where caption contains "fusion", "indo", "ethnic" + Western keyword
  3. Writes edge_case_candidates.csv for manual review

Usage:
  py -3 scripts/08_generate_metadata.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import csv
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_META,
    DEEPFASHION_IW_KEYWORDS,
    DEEPFASHION_WESTERN_KEYWORDS,
)


def determine_edge_case(rec: dict) -> str:
    """
    Heuristic edge-case labelling.
    Returns one of:
      normal | hybrid | visually_similar | ambiguous
    This is a CANDIDATE label — manual review recommended.
    """
    final_class = rec.get("final_class", "")
    caption     = (rec.get("product_title", "") or "").lower()
    ar          = rec.get("aspect_ratio", 1.0)

    if final_class == "Indo-Western":
        return "hybrid"   # all IW images are by definition edge cases

    # Caption-based heuristics for Western/Indian images that might be confusable
    has_iw_keyword = any(kw.lower() in caption for kw in DEEPFASHION_IW_KEYWORDS)
    if has_iw_keyword:
        return "ambiguous"

    if final_class == "Indian":
        # Check if caption mentions any Western keywords (potential confusion)
        has_west_kw = any(kw.lower() in caption for kw in DEEPFASHION_WESTERN_KEYWORDS[:5])
        if has_west_kw:
            return "visually_similar"

    if final_class == "Western":
        # Unusually square images (portrait fashion photos look similar across cultures)
        if 0.85 <= ar <= 1.18:
            return "visually_similar"

    return "normal"


def main():
    print("=" * 60)
    print("  STEP 08 — Generate Metadata")
    print("=" * 60)

    split_path = DATASET_META / "split_records.json"
    if not split_path.exists():
        print(f"[ERROR] {split_path} not found. Run step 07 first.")
        sys.exit(1)

    with open(split_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} split records")

    DATASET_META.mkdir(parents=True, exist_ok=True)

    # ── Full dataset CSV columns ───────────────────────────────
    COLUMNS = [
        "image_id", "filename", "filepath",
        "source_dataset", "original_category", "final_class",
        "split", "width", "height", "aspect_ratio", "file_size",
        "is_duplicate", "is_corrupt", "edge_case",
        "product_title", "brand",
    ]

    full_rows       = []
    edge_candidates = []

    for rec in records:
        edge = determine_edge_case(rec)
        row = {
            "image_id":          rec.get("image_id", ""),
            "filename":          rec.get("filename", ""),
            "filepath":          rec.get("filepath", ""),
            "source_dataset":    rec.get("source_dataset", ""),
            "original_category": rec.get("original_category", ""),
            "final_class":       rec.get("final_class", ""),
            "split":             rec.get("split", ""),
            "width":             rec.get("width", ""),
            "height":            rec.get("height", ""),
            "aspect_ratio":      rec.get("aspect_ratio", ""),
            "file_size":         rec.get("file_size", ""),
            "is_duplicate":      rec.get("is_duplicate", False),
            "is_corrupt":        rec.get("is_corrupt", False),
            "edge_case":         edge,
            "product_title":     rec.get("product_title", ""),
            "brand":             rec.get("brand", ""),
        }
        full_rows.append(row)
        if edge != "normal":
            edge_candidates.append({
                "image_id":      row["image_id"],
                "final_class":   row["final_class"],
                "edge_case":     edge,
                "product_title": row["product_title"],
                "filepath":      row["filepath"],
                "note":          "Auto-flagged; manual review recommended",
            })

    # ── Write CSVs ─────────────────────────────────────────────
    def write_csv(path: Path, rows: list, columns: list):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Written → {path}  ({len(rows)} rows)")

    write_csv(DATASET_META / "dataset.csv",   full_rows, COLUMNS)
    write_csv(DATASET_META / "train.csv",
              [r for r in full_rows if r["split"] == "train"], COLUMNS)
    write_csv(DATASET_META / "val.csv",
              [r for r in full_rows if r["split"] == "val"],   COLUMNS)
    write_csv(DATASET_META / "test.csv",
              [r for r in full_rows if r["split"] == "test"],  COLUMNS)

    # Edge-case candidates
    ec_path = DATASET_META / "edge_case_candidates.csv"
    edge_cols = ["image_id", "final_class", "edge_case", "product_title", "filepath", "note"]
    write_csv(ec_path, edge_candidates, edge_cols)
    print(f"  Edge-case candidates: {len(edge_candidates)}")

    # ── Class distribution ─────────────────────────────────────
    dist_rows = []
    by_split_class = defaultdict(Counter)
    for r in full_rows:
        by_split_class[r["split"]][r["final_class"]] += 1

    all_classes = sorted({r["final_class"] for r in full_rows})
    for split in ["train", "val", "test", "total"]:
        for cls in all_classes:
            if split == "total":
                cnt = sum(by_split_class[s][cls] for s in ["train", "val", "test"])
            else:
                cnt = by_split_class[split][cls]
            dist_rows.append({"split": split, "class": cls, "count": cnt})

    write_csv(
        DATASET_META.parent / "reports" / "class_distribution.csv",
        dist_rows,
        ["split", "class", "count"],
    )

    # ── Image statistics ──────────────────────────────────────
    widths       = [r["width"]  for r in full_rows if r["width"]]
    heights      = [r["height"] for r in full_rows if r["height"]]
    aspects      = [r["aspect_ratio"] for r in full_rows if r["aspect_ratio"]]
    file_sizes   = [r["file_size"] for r in full_rows if r["file_size"]]

    def safe_stat(lst):
        lst = [x for x in lst if x]
        if not lst:
            return {"min": "N/A", "max": "N/A", "mean": "N/A", "median": "N/A"}
        lst_sorted = sorted(lst)
        n = len(lst_sorted)
        return {
            "min":    round(min(lst_sorted), 4),
            "max":    round(max(lst_sorted), 4),
            "mean":   round(sum(lst_sorted) / n, 4),
            "median": round(lst_sorted[n // 2], 4),
        }

    stats = {
        "width":      safe_stat(widths),
        "height":     safe_stat(heights),
        "aspect_ratio": safe_stat(aspects),
        "file_size_bytes": safe_stat(file_sizes),
    }

    stat_rows = []
    for metric, s in stats.items():
        for stat, val in s.items():
            stat_rows.append({"metric": metric, "statistic": stat, "value": val})

    write_csv(
        DATASET_META.parent / "reports" / "image_statistics.csv",
        stat_rows,
        ["metric", "statistic", "value"],
    )

    # ── Print summary ─────────────────────────────────────────
    print(f"\n  Class distribution:")
    for row in dist_rows:
        if row["split"] == "total":
            print(f"    {row['class']:<15} {row['count']:>6}")

    print(f"\n  Edge-case breakdown:")
    ec_counter = Counter(r["edge_case"] for r in full_rows)
    for ec, cnt in ec_counter.most_common():
        print(f"    {ec:<20} {cnt:>6}")

    print("\n[DONE] Metadata generation complete.")


if __name__ == "__main__":
    main()
