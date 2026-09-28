"""
prepare_dataset.py
===================
Top-level pipeline runner. Executes all 9 steps in order.

Usage:
  py -3 prepare_dataset.py              # full pipeline
  py -3 prepare_dataset.py --dry-run   # inspect only, no file writes
  py -3 prepare_dataset.py --from 3    # start from step 3
  py -3 prepare_dataset.py --to 5      # run steps 1-5 only
  py -3 prepare_dataset.py --skip-download   # skip step 01

Project: Robust Indian, Western, and Indo-Western Apparel Classification
         with Edge-Case Profiling for Inclusive E-Commerce

Students:
  1. Swaroop Vijaykumar Purandare  — 1GA23CS176
  2. Ranna                         — 1GA23CS135
  3. SUKESH                        — 1GA23CS171
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    (1,  "01_download_datasets.py",     "Download Kaggle datasets"),
    (2,  "02_inspect_datasets.py",      "Inspect datasets"),
    (3,  "03_build_category_mapping.py","Build category mapping"),
    (4,  "04_sample_dataset.py",        "Sample candidates"),
    (5,  "05_clean_dataset.py",         "Clean and validate images"),
    (6,  "06_detect_duplicates.py",     "Detect duplicates"),
    (7,  "07_create_splits.py",         "Create train/val/test splits"),
    (8,  "08_generate_metadata.py",     "Generate metadata CSVs"),
    (9,  "09_generate_report.py",       "Generate dataset report"),
]

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def run_step(step_num: int, script: str, description: str,
             dry_run: bool, extra_args: list) -> bool:
    script_path = SCRIPTS_DIR / script
    cmd = [sys.executable, str(script_path)] + extra_args
    if dry_run and step_num not in (3, 9):    # steps 3 and 9 don't support dry-run
        cmd.append("--dry-run")

    print(f"\n{'='*60}")
    print(f"  STEP {step_num:02d}: {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Apparel Classification Dataset Preparation Pipeline"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run in inspection mode — no images copied")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--to", dest="to_step", type=int, default=9,
                        help="Stop after this step number (default: 9)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip step 01 (download) — use if already downloaded")
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  APPAREL CLASSIFICATION DATASET PIPELINE")
    print("  Project: Indian / Western / Indo-Western")
    print("  Students:")
    print("    1. Swaroop Vijaykumar Purandare — 1GA23CS176")
    print("    2. Ranna                         — 1GA23CS135")
    print("    3. SUKESH                        — 1GA23CS171")
    print("█"*60)

    if args.dry_run:
        print("\n  *** DRY RUN MODE — no files will be copied ***\n")

    failed_steps = []
    for step_num, script, desc in STEPS:
        if step_num < args.from_step or step_num > args.to_step:
            continue
        if args.skip_download and step_num == 1:
            print(f"\n[SKIP] Step 01 (--skip-download flag set)")
            continue

        ok = run_step(step_num, script, desc, args.dry_run, [])
        if not ok:
            print(f"\n[ERROR] Step {step_num} failed: {script}")
            failed_steps.append(step_num)
            ans = input("  Continue anyway? [y/N]: ").strip().lower()
            if ans != "y":
                break

    print("\n" + "█"*60)
    if not failed_steps:
        print("  PIPELINE COMPLETE — all steps succeeded")
        print("  Run:  py -3 validate_dataset.py  to verify the dataset")
    else:
        print(f"  PIPELINE FINISHED WITH ERRORS in steps: {failed_steps}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
