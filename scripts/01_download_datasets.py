"""
01_download_datasets.py
========================
Downloads both Kaggle datasets using the Kaggle API.

Requirements:
  pip install kaggle

Setup (one-time):
  1. Go to https://www.kaggle.com/settings  →  API  →  "Create New Token"
  2. Place the downloaded kaggle.json at:
       Windows : C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
       Linux   : ~/.kaggle/kaggle.json
  3. Set permissions (Linux only):
       chmod 600 ~/.kaggle/kaggle.json

  OR set environment variables:
       set KAGGLE_USERNAME=your_username
       set KAGGLE_KEY=your_api_key

Usage:
  py -3 scripts/01_download_datasets.py            # normal
  py -3 scripts/01_download_datasets.py --dry-run  # show what would happen
  py -3 scripts/01_download_datasets.py --skip-deep-fashion   # skip large dataset

Disk estimates:
  Indo-Fashion    : ~4 GB   (106K images)
  DeepFashion-MM  : ~15 GB  (44K high-res images)
  Total raw data  : ~19 GB
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import subprocess
import sys
import os
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR,
    INDOFASHION_DIR,
    DEEPFASHION_DIR,
    KAGGLE_INDOFASHION,
    KAGGLE_DEEPFASHION,
    DISK_ESTIMATE,
)


def check_kaggle_auth() -> bool:
    """Return True if Kaggle credentials are reachable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    has_file = kaggle_json.exists() or access_token.exists()
    has_env  = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or bool(os.environ.get("KAGGLE_API_TOKEN"))
    if not has_file and not has_env:
        print("\n[ERROR] Kaggle credentials not found.")
        print("  Option A: Place kaggle.json at:")
        print(f"            {kaggle_json}")
        print("  Option B: Set environment variables:")
        print("            set KAGGLE_USERNAME=your_username")
        print("            set KAGGLE_KEY=your_api_key")
        print("\nGet your API token from: https://www.kaggle.com/settings → API → Create New Token")
        return False
    if kaggle_json.exists():
        print(f"[OK] Found kaggle.json at {kaggle_json}")
    elif access_token.exists():
        print(f"[OK] Found access_token at {access_token}")
    else:
        print("[OK] Found KAGGLE_USERNAME / KAGGLE_KEY or KAGGLE_API_TOKEN environment variables")
    return True


def dataset_exists(target_dir: Path, marker_file: str) -> bool:
    """Check if a dataset has already been downloaded."""
    return (target_dir / marker_file).exists()


def download_dataset(dataset_slug: str, target_dir: Path, dry_run: bool) -> bool:
    """
    Download a Kaggle dataset and unzip into target_dir.
    Returns True on success.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "--dataset", dataset_slug,
        "--unzip",
        "--path", str(target_dir),
        "--force",    # overwrite partial downloads
    ]
    print(f"\n  Command: {' '.join(cmd)}")
    if dry_run:
        print("  [DRY RUN] Would execute the above command.")
        return True
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] kaggle CLI not found. Run:  pip install kaggle")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets for the Apparel Classification project"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without downloading")
    parser.add_argument("--skip-deep-fashion", action="store_true",
                        help="Skip the large DeepFashion-MultiModal download (~15 GB)")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 01 — Download Datasets")
    print("=" * 60)
    print("\nDisk space estimates:")
    for k, v in DISK_ESTIMATE.items():
        print(f"  {k:<30}: {v}")

    if not check_kaggle_auth():
        sys.exit(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Dataset 1: IndoFashion ──────────────────────────────────
    print("\n" + "─" * 50)
    print("Dataset 1: Indo-Fashion Dataset")
    print(f"  Slug    : {KAGGLE_INDOFASHION}")
    print(f"  Target  : {INDOFASHION_DIR}")
    print(f"  Size    : ~4 GB")

    if dataset_exists(INDOFASHION_DIR, "train.json"):
        print("  [SKIP] Already downloaded (train.json found).")
    else:
        ok = download_dataset(KAGGLE_INDOFASHION, INDOFASHION_DIR, args.dry_run)
        if not ok and not args.dry_run:
            print("[WARN] IndoFashion download may have failed. Check above.")

    # ── Dataset 2: DeepFashion-MultiModal ─────────────────────
    print("\n" + "─" * 50)
    print("Dataset 2: DeepFashion-MultiModal")
    print(f"  Slug    : {KAGGLE_DEEPFASHION}")
    print(f"  Target  : {DEEPFASHION_DIR}")
    print(f"  Size    : ~15 GB")

    if args.skip_deep_fashion:
        print("  [SKIP] --skip-deep-fashion flag set.")
    elif dataset_exists(DEEPFASHION_DIR, "captions.json"):
        print("  [SKIP] Already downloaded (captions.json found).")
    else:
        ok = download_dataset(KAGGLE_DEEPFASHION, DEEPFASHION_DIR, args.dry_run)
        if not ok and not args.dry_run:
            print("[WARN] DeepFashion download may have failed. Check above.")

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  DRY RUN complete. No files downloaded.")
    else:
        print("  Download step complete.")
        print(f"  Raw data location: {RAW_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
