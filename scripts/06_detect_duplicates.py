"""
06_detect_duplicates.py
========================
Detects and removes:
  1. Exact duplicates (MD5 hash)
  2. Near-duplicates (perceptual hash / pHash, Hamming distance)

Strategy:
  - Exact: compute MD5 of file bytes; keep first occurrence
  - Near:  compute pHash (16-bit); build BK-Tree of hashes;
           flag pairs within PHASH_THRESHOLD Hamming distance
  - Output: deduplicated clean_candidates.json + duplicate_report.csv

IMPORTANT: Near-duplicate removal operates WITHIN the candidate pool
before any split. This prevents the same garment from appearing in
both training and test sets.

Usage:
  py -3 scripts/06_detect_duplicates.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_META,
    PHASH_BITS,
    PHASH_THRESHOLD,
)

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import imagehash as _imagehash_lib
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False

import math


def _compute_phash(img_path: Path, hash_size: int = 8) -> int | None:
    """
    Compute perceptual hash for near-duplicate detection.
    Uses imagehash library (phash) when available for accuracy;
    falls back to fast built-in average-hash implementation.
    Returns an integer hash, or None on error.
    """
    if not PIL_OK:
        return None
    try:
        with Image.open(img_path) as img:
            if IMAGEHASH_OK:
                # imagehash.phash is the gold standard — 64-bit DCT hash
                h = _imagehash_lib.phash(img, hash_size=hash_size)
                return int(str(h), 16)   # convert hex string to int
            else:
                # Fast fallback: average-hash (mean-based)
                img_g = img.convert("L").resize((hash_size * 4, hash_size * 4),
                                                Image.LANCZOS)
                raw  = img_g.tobytes()
                pix  = list(raw)
                avg  = sum(pix) / len(pix)
                bits = [1 if p >= avg else 0 for p in pix]
                h = 0
                for b in bits:
                    h = (h << 1) | b
                return h
    except Exception:
        return None


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    return (h1 ^ h2).bit_count()


def compute_md5(path: Path) -> str | None:
    """Compute MD5 hash of file bytes."""
    try:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None


def main():
    print("=" * 60)
    print("  STEP 06 — Detect Duplicates")
    print("=" * 60)

    clean_path = DATASET_META / "clean_candidates.json"
    if not clean_path.exists():
        print(f"[ERROR] {clean_path} not found. Run step 05 first.")
        sys.exit(1)

    with open(clean_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} clean records")

    # ─────────────────────────────────────────────
    #  Pass 1: Exact duplicate detection (MD5)
    # ─────────────────────────────────────────────
    print("\n  Pass 1: Exact duplicate detection (MD5)...")
    md5_seen  = {}      # md5 → first record image_id
    exact_dups = []
    keep_after_exact = []

    for rec in records:
        src = rec.get("staging_path", "")
        if not src or src in ("DRY_RUN", "PENDING", "NOT_FOUND"):
            # No staged file yet — skip MD5 check
            keep_after_exact.append(rec)
            continue

        p    = Path(src)
        md5  = compute_md5(p) if p.exists() else None

        if md5 is None:
            keep_after_exact.append(rec)
            continue

        if md5 in md5_seen:
            exact_dups.append({
                "image_id":        rec["image_id"],
                "duplicate_of":    md5_seen[md5],
                "type":            "exact_md5",
                "staging_path":    src,
            })
        else:
            md5_seen[md5] = rec["image_id"]
            keep_after_exact.append(rec)

    print(f"  Exact duplicates found: {len(exact_dups)}")
    print(f"  Records remaining: {len(keep_after_exact)}")

    # ─────────────────────────────────────────────
    #  Pass 2: Near-duplicate detection (pHash)
    # ─────────────────────────────────────────────
    print(f"\n  Pass 2: Near-duplicate detection (pHash, threshold={PHASH_THRESHOLD})...")

    hashes      = []   # list of (image_id, phash)
    no_hash     = []

    for rec in keep_after_exact:
        src = rec.get("staging_path", "")
        if not src or src in ("DRY_RUN", "PENDING", "NOT_FOUND"):
            hashes.append((rec["image_id"], None))
            continue
        p = Path(src)
        h = _compute_phash(p, hash_size=PHASH_BITS) if p.exists() else None
        hashes.append((rec["image_id"], h))
        if h is None:
            no_hash.append(rec["image_id"])

    # Brute-force pairwise check (O(n²)) — acceptable for n ≤ 3500
    print(f"  Computing pairwise Hamming distances (n={len(hashes)})...")
    near_dup_pairs = []
    near_dup_ids   = set()

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            h1, h2 = hashes[i][1], hashes[j][1]
            if h1 is None or h2 is None:
                continue
            dist = hamming_distance(h1, h2)
            if dist <= PHASH_THRESHOLD:
                id1, id2 = hashes[i][0], hashes[j][0]
                near_dup_pairs.append({
                    "image_id_1": id1,
                    "image_id_2": id2,
                    "hamming_distance": dist,
                    "type": "near_duplicate_phash",
                })
                # Keep the first occurrence
                near_dup_ids.add(id2)

    print(f"  Near-duplicate pairs found: {len(near_dup_pairs)}")

    keep_after_near = [
        rec for rec in keep_after_exact
        if rec["image_id"] not in near_dup_ids
    ]
    print(f"  Records remaining after dedup: {len(keep_after_near)}")

    # Update is_duplicate field
    dup_id_set = {d["image_id"] for d in exact_dups} | near_dup_ids
    for rec in keep_after_near:
        rec["is_duplicate"] = False
    for rec in records:
        if rec.get("image_id") in dup_id_set:
            rec["is_duplicate"] = True

    # ─────────────────────────────────────────────
    #  Save outputs
    # ─────────────────────────────────────────────
    dedup_path = DATASET_META / "deduplicated_candidates.json"
    with open(dedup_path, "w", encoding="utf-8") as f:
        json.dump(keep_after_near, f, indent=2)
    print(f"\n  Deduplicated candidates → {dedup_path}")

    dup_report_path = DATASET_META / "duplicate_report.csv"
    all_dups = exact_dups + near_dup_pairs
    with open(dup_report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list({k for d in all_dups for k in d.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_dups)
    print(f"  Duplicate report         → {dup_report_path}")

    print(f"\n  Summary:")
    print(f"    Input records    : {len(records)}")
    print(f"    Exact dups       : {len(exact_dups)}")
    print(f"    Near-dups (pHash): {len(near_dup_ids)}")
    print(f"    Final unique     : {len(keep_after_near)}")

    print("\n[DONE] Duplicate detection complete.")


if __name__ == "__main__":
    main()
