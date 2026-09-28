"""
09_generate_report.py
======================
Reads all metadata and generates:
  - dataset/reports/dataset_report.md   (comprehensive Markdown report)
  - dataset/reports/class_distribution.csv (if not already written by step 08)

Usage:
  py -3 scripts/09_generate_report.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_META,
    DATASET_REPORTS,
    TARGET_COUNTS,
    SPLIT_RATIOS,
    SEED,
    TARGET_SIZE,
    PHASH_THRESHOLD,
    PHASH_BITS,
    MIN_DIMENSION,
    MAX_ASPECT_RATIO,
    KAGGLE_INDOFASHION,
    KAGGLE_DEEPFASHION,
    DISK_ESTIMATE,
)

REPORT_DATE = datetime.now().strftime("%Y-%m-%d %H:%M")


def load_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_report():
    print("=" * 60)
    print("  STEP 09 — Generate Dataset Report")
    print("=" * 60)

    DATASET_REPORTS.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────────
    dataset_rows = load_csv_rows(DATASET_META / "dataset.csv")
    corrupted    = load_csv_rows(DATASET_META / "corrupted_images.csv")
    dup_report   = load_csv_rows(DATASET_META / "duplicate_report.csv")
    ambiguous    = load_csv_rows(DATASET_META / "ambiguous_excluded.csv")
    mapping      = load_csv_rows(DATASET_META / "category_mapping.csv")
    edge_cands   = load_csv_rows(DATASET_META / "edge_case_candidates.csv")
    stat_rows    = load_csv_rows(DATASET_REPORTS / "image_statistics.csv")
    dist_rows    = load_csv_rows(DATASET_REPORTS / "class_distribution.csv")

    # ── Compute summary statistics ─────────────────────────────
    total_imgs      = len(dataset_rows)
    by_class        = Counter(r["final_class"] for r in dataset_rows)
    by_split        = Counter(r["split"]       for r in dataset_rows)
    by_split_class  = defaultdict(Counter)
    for r in dataset_rows:
        by_split_class[r["split"]][r["final_class"]] += 1

    n_corrupt       = len(corrupted)
    n_exact_dup     = sum(1 for r in dup_report if r.get("type","") == "exact_md5")
    n_near_dup      = sum(1 for r in dup_report if "near_duplicate" in r.get("type",""))
    n_ambiguous_exc = len(ambiguous)
    n_edge_cands    = len(edge_cands)

    # ── Stats helper ──────────────────────────────────────────
    def get_stat(metric: str, stat: str) -> str:
        for r in stat_rows:
            if r.get("metric") == metric and r.get("statistic") == stat:
                return r.get("value", "N/A")
        return "N/A"

    # ── Build Markdown report ─────────────────────────────────
    lines = []
    a = lines.append

    a(f"# Dataset Report")
    a(f"")
    a(f"**Project:** Robust Indian, Western, and Indo-Western Apparel Classification  ")
    a(f"**Subtitle:** with Edge-Case Profiling for Inclusive E-Commerce  ")
    a(f"**Generated:** {REPORT_DATE}")
    a(f"")
    a(f"## Students")
    a(f"")
    a(f"| # | Name | USN |")
    a(f"|---|------|-----|")
    a(f"| 1 | Swaroop Vijaykumar Purandare | 1GA23CS176 |")
    a(f"| 2 | Ranna | 1GA23CS135 |")
    a(f"| 3 | SUKESH | 1GA23CS171 |")
    a(f"")
    a(f"---")
    a(f"")
    a(f"## 1. Dataset Sources")
    a(f"")
    a(f"| Dataset | Kaggle Slug | Source | Original Size |")
    a(f"|---------|------------|--------|--------------|")
    a(f"| IndoFashion | `{KAGGLE_INDOFASHION}` | Rajput & Aneja, CVPRW 2021 | ~106,000 images, 15 classes |")
    a(f"| DeepFashion-MultiModal | `{KAGGLE_DEEPFASHION}` | Jiang et al., SIGGRAPH 2022 | ~44,096 images |")
    a(f"")
    a(f"## 2. Download Information")
    a(f"")
    a(f"```bash")
    a(f"# Install Kaggle API")
    a(f"pip install kaggle")
    a(f"")
    a(f"# Place credentials at: %USERPROFILE%\\.kaggle\\kaggle.json")
    a(f"")
    a(f"# Download IndoFashion (~4 GB)")
    a(f"kaggle datasets download --dataset {KAGGLE_INDOFASHION} --unzip --path raw_data/indo-fashion-dataset")
    a(f"")
    a(f"# Download DeepFashion-MultiModal (~15 GB)")
    a(f"kaggle datasets download --dataset {KAGGLE_DEEPFASHION} --unzip --path raw_data/deep-fashion-multimodal")
    a(f"```")
    a(f"")
    a(f"### Disk Space Estimates")
    a(f"")
    a(f"| Stage | Estimate |")
    a(f"|-------|---------|")
    for k, v in DISK_ESTIMATE.items():
        a(f"| {k.replace('_', ' ')} | {v} |")
    a(f"")
    a(f"## 3. Source Dataset Statistics")
    a(f"")
    a(f"| Dataset | Source Images | Classes | Type |")
    a(f"|---------|--------------|---------|------|")
    a(f"| IndoFashion | ~106,000 | 15 confirmed | Discrete folders + JSON metadata |")
    a(f"| DeepFashion-MM | ~44,096 | None (caption-derived) | captions.json + attribute annotations |")
    a(f"")
    a(f"## 4. Usable / Corrupted / Duplicate Statistics")
    a(f"")
    a(f"| Metric | Count |")
    a(f"|--------|-------|")
    a(f"| Final clean images | {total_imgs} |")
    a(f"| Corrupted/invalid removed | {n_corrupt} |")
    a(f"| Exact duplicates removed | {n_exact_dup} |")
    a(f"| Near-duplicates removed (pHash) | {n_near_dup} |")
    a(f"| Ambiguous categories excluded | {n_ambiguous_exc} |")
    a(f"| Edge-case candidates flagged | {n_edge_cands} |")
    a(f"")
    a(f"## 5. Final Class Counts")
    a(f"")
    a(f"| Class | Target | Actual | Δ |")
    a(f"|-------|--------|--------|---|")
    for cls, target in TARGET_COUNTS.items():
        actual = by_class.get(cls, 0)
        delta  = actual - target
        flag   = f"+{delta}" if delta >= 0 else str(delta)
        a(f"| {cls} | {target} | {actual} | {flag} |")
    a(f"| **TOTAL** | **{sum(TARGET_COUNTS.values())}** | **{total_imgs}** | - |")
    a(f"")
    a(f"## 6. Category Distribution (Source → Final Class)")
    a(f"")
    a(f"| Source Dataset | Original Category | Final Class | Reason | Count |")
    a(f"|---------------|-------------------|-------------|--------|-------|")
    for row in mapping:
        a(f"| {row.get('source_dataset','')} | {row.get('original_category','')} | "
          f"{row.get('final_class','')} | {row.get('reason','')} | {row.get('selected_count','')} |")
    a(f"")
    a(f"### Excluded / Ambiguous Categories")
    a(f"")
    a(f"| Source | Category | Reason Excluded |")
    a(f"|--------|----------|----------------|")
    for row in ambiguous:
        reason = row.get('reason_excluded', '')[:100] + "..."
        a(f"| {row.get('source_dataset','')} | {row.get('original_category','')} | {reason} |")
    a(f"")
    a(f"## 7. Train / Validation / Test Distribution")
    a(f"")
    a(f"| Split | Indian | Western | Indo-Western | Total |")
    a(f"|-------|--------|---------|--------------|-------|")
    for split in ["train", "val", "test"]:
        ind  = by_split_class[split].get("Indian", 0)
        west = by_split_class[split].get("Western", 0)
        iw   = by_split_class[split].get("Indo-Western", 0)
        tot  = ind + west + iw
        a(f"| {split} | {ind} | {west} | {iw} | {tot} |")
    tot_ind  = by_class.get("Indian", 0)
    tot_west = by_class.get("Western", 0)
    tot_iw   = by_class.get("Indo-Western", 0)
    a(f"| **Total** | **{tot_ind}** | **{tot_west}** | **{tot_iw}** | **{total_imgs}** |")
    a(f"")
    a(f"Split ratios: train={SPLIT_RATIOS['train']:.0%}, "
      f"val={SPLIT_RATIOS['val']:.0%}, test={SPLIT_RATIOS['test']:.0%}")
    a(f"")
    a(f"## 8. Image Resolution Statistics (after preprocessing)")
    a(f"")
    a(f"| Metric | Min | Max | Mean | Median |")
    a(f"|--------|-----|-----|------|--------|")
    for metric in ["width", "height", "aspect_ratio", "file_size_bytes"]:
        a(f"| {metric} | {get_stat(metric,'min')} | {get_stat(metric,'max')} | "
          f"{get_stat(metric,'mean')} | {get_stat(metric,'median')} |")
    a(f"")
    a(f"Output image size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px (letterbox-padded JPEG)")
    a(f"")
    a(f"## 9. Sampling Strategy")
    a(f"")
    a(f"1. **IndoFashion** — 15 confirmed class labels are mapped to Indian/Indo-Western.")
    a(f"   - Categories mapped to **Indian**: saree, women_kurta, leggings_and_salwar,")
    a(f"     lehenga, men_kurta, sherwani")
    a(f"   - Categories mapped to **Indo-Western**: dhoti_pants, nehru_jacket")
    a(f"   - Sampling uses per-category quotas to ensure diversity across categories.")
    a(f"   - Maximum quota per category prevents any single category from dominating.")
    a(f"")
    a(f"2. **DeepFashion-MultiModal** — No discrete garment categories exist.")
    a(f"   - Images are classified by keyword-matching on `captions.json` entries.")
    a(f"   - Keywords are priority-ordered: Indo-Western > Indian > Western.")
    a(f"   - DeepFashion primarily supplies **Western** class images (1,500 target).")
    a(f"   - IndoFashion supplies all **Indian** (1,350) and partial **Indo-Western** (200).")
    a(f"")
    a(f"## 10. Cleaning Strategy")
    a(f"")
    a(f"For each candidate image:")
    a(f"")
    a(f"1. Verify file exists on disk")
    a(f"2. Open with Pillow + `verify()` call to detect truncation/corruption")
    a(f"3. Reload and `load()` to catch lazy-load failures")
    a(f"4. Check minimum dimension ≥ {MIN_DIMENSION}px on both axes")
    a(f"5. Check aspect ratio ≤ {MAX_ASPECT_RATIO}:1")
    a(f"6. Convert to RGB (RGBA → composite on white, grayscale → RGB)")
    a(f"7. Letterbox-resize to {TARGET_SIZE[0]}×{TARGET_SIZE[1]} (LANCZOS, black padding)")
    a(f"8. Save as JPEG quality={95} with subsampling=0")
    a(f"")
    a(f"## 11. Duplicate Detection Strategy")
    a(f"")
    a(f"**Pass 1 — Exact duplicates (MD5)**")
    a(f"- Compute MD5 hash of raw file bytes")
    a(f"- Keep first occurrence; flag remainder")
    a(f"")
    a(f"**Pass 2 — Near-duplicates (pHash)**")
    a(f"- Compute perceptual hash (average-hash variant, {PHASH_BITS}×{PHASH_BITS} bits)")
    a(f"- Brute-force O(n²) Hamming distance comparison across all pairs")
    a(f"- Flag pairs with Hamming distance ≤ {PHASH_THRESHOLD}")
    a(f"- Keep first occurrence; flag remainder")
    a(f"- Applied BEFORE the train/val/test split to prevent leakage")
    a(f"")
    a(f"## 12. Edge-Case Profiling")
    a(f"")
    a(f"| Label | Meaning | How Assigned |")
    a(f"|-------|---------|-------------|")
    a(f"| normal | No detected edge-case properties | Default |")
    a(f"| hybrid | Confirmed fusion garment | All Indo-Western class images |")
    a(f"| visually_similar | May be confused across classes | Heuristic (caption keywords + AR) |")
    a(f"| ambiguous | Cannot be cleanly assigned | Fusion keywords in Indian/Western image |")
    a(f"")
    a(f"> **Note:** Automatic edge-case labels are heuristic. `edge_case_candidates.csv`")
    a(f"> lists all non-normal images for optional manual review.")
    a(f"")
    a(f"## 13. Anti-Leakage Measures")
    a(f"")
    a(f"- Deduplication (both exact and near-duplicate) is performed **before** splitting")
    a(f"- Stratified split preserves class proportions in each partition")
    a(f"- Fixed random seed (`SEED = {SEED}`) ensures reproducibility")
    a(f"- No image appears in more than one split")
    a(f"")
    a(f"## 14. Potential Limitations")
    a(f"")
    a(f"1. **DeepFashion classification relies on captions** — if a Western garment")
    a(f"   caption contains no recognisable Western keywords, it will be missed.")
    a(f"   Manual spot-checking of a random 5% sample is recommended.")
    a(f"")
    a(f"2. **IndoFashion gown and palazzo categories** are excluded because the label")
    a(f"   alone does not distinguish Indian ethnic gowns from Western gowns or Indian")
    a(f"   ethnic palazzo pants from Western-derived ones. This exclusion is documented")
    a(f"   in `ambiguous_excluded.csv`.")
    a(f"")
    a(f"3. **Indo-Western class is small (200 images)** — this reflects the genuine")
    a(f"   scarcity of reliably labelled fusion garment data. Do not upscale this class")
    a(f"   by randomly mixing Indian and Western images; use class-weighted loss instead.")
    a(f"")
    a(f"4. **No demographic filtering is applied** — classification is based on garment")
    a(f"   characteristics, not on person attributes. However, source datasets are not")
    a(f"   demographically balanced and this should be noted in any publication.")
    a(f"")
    a(f"5. **pHash near-duplicate detection may miss non-standard duplicates** (e.g.,")
    a(f"   same garment photographed on different backgrounds). Caption similarity checks")
    a(f"   on product titles could supplement pHash for the Indo-Fashion dataset.")
    a(f"")
    a(f"## 15. Reproducibility Instructions")
    a(f"")
    a(f"```bash")
    a(f"# 1. Install dependencies")
    a(f"pip install -r requirements.txt")
    a(f"")
    a(f"# 2. Place Kaggle credentials")
    a(f"# Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
    a(f"")
    a(f"# 3. Run the full pipeline (produces identical output with seed={SEED})")
    a(f"py -3 prepare_dataset.py")
    a(f"")
    a(f"# 4. Validate the dataset")
    a(f"py -3 validate_dataset.py")
    a(f"")
    a(f"# 5. Dry-run (inspection only, no file writes)")
    a(f"py -3 prepare_dataset.py --dry-run")
    a(f"```")
    a(f"")
    a(f"## 16. PyTorch DataLoader Usage")
    a(f"")
    a(f"```python")
    a(f"import torchvision.transforms as T")
    a(f"from torchvision.datasets import ImageFolder")
    a(f"from torch.utils.data import DataLoader")
    a(f"")
    a(f"# Normalisation statistics (ImageNet defaults — suitable for pretrained models)")
    a(f"mean = [0.485, 0.456, 0.406]")
    a(f"std  = [0.229, 0.224, 0.225]")
    a(f"")
    a(f"train_tf = T.Compose([")
    a(f"    T.Resize((224, 224)),          # already 224×224; kept for safety")
    a(f"    T.RandomHorizontalFlip(),")
    a(f"    T.RandomRotation(15),")
    a(f"    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),")
    a(f"    T.ToTensor(),")
    a(f"    T.Normalize(mean, std),")
    a(f"])")
    a(f"")
    a(f"val_tf = T.Compose([")
    a(f"    T.Resize((224, 224)),")
    a(f"    T.ToTensor(),")
    a(f"    T.Normalize(mean, std),")
    a(f"])")
    a(f"")
    a(f"train_ds = ImageFolder('dataset/train', transform=train_tf)")
    a(f"val_ds   = ImageFolder('dataset/val',   transform=val_tf)")
    a(f"test_ds  = ImageFolder('dataset/test',  transform=val_tf)")
    a(f"")
    a(f"# Class order: {{'Indian': 0, 'Indo-Western': 1, 'Western': 2}} (alphabetical)")
    a(f"print(train_ds.class_to_idx)")
    a(f"")
    a(f"train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)")
    a(f"val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)")
    a(f"test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)")
    a(f"```")
    a(f"")
    a(f"### Class Imbalance Handling")
    a(f"")
    a(f"```python")
    a(f"import torch")
    a(f"# Compute class weights (inverse frequency)")
    a(f"counts = torch.tensor([train_ds.targets.count(i) for i in range(3)], dtype=torch.float)")
    a(f"weights = 1.0 / counts")
    a(f"weights /= weights.sum()")
    a(f"criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))")
    a(f"```")
    a(f"")
    a(f"---")
    a(f"*Report generated by `09_generate_report.py` on {REPORT_DATE}*")

    report_path = DATASET_REPORTS / "dataset_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report written → {report_path}")
    print(f"\n[DONE] Report generation complete.")
    return report_path


if __name__ == "__main__":
    generate_report()
