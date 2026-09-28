# Dataset Report

**Project:** Robust Indian, Western, and Indo-Western Apparel Classification  
**Subtitle:** with Edge-Case Profiling for Inclusive E-Commerce  
**Generated:** 2026-09-28 16:21

## Students

| # | Name | USN |
|---|------|-----|
| 1 | Swaroop Vijaykumar Purandare | 1GA23CS176 |
| 2 | Ranna | 1GA23CS135 |
| 3 | SUKESH | 1GA23CS171 |

---

## 1. Dataset Sources

| Dataset | Kaggle Slug | Source | Original Size |
|---------|------------|--------|--------------|
| IndoFashion | `validmodel/indo-fashion-dataset` | Rajput & Aneja, CVPRW 2021 | ~106,000 images, 15 classes |
| DeepFashion-MultiModal | `silverstone1903/deep-fashion-multimodal` | Jiang et al., SIGGRAPH 2022 | ~44,096 images |

## 2. Download Information

```bash
# Install Kaggle API
pip install kaggle

# Place credentials at: %USERPROFILE%\.kaggle\kaggle.json

# Download IndoFashion (~4 GB)
kaggle datasets download --dataset validmodel/indo-fashion-dataset --unzip --path raw_data/indo-fashion-dataset

# Download DeepFashion-MultiModal (~15 GB)
kaggle datasets download --dataset silverstone1903/deep-fashion-multimodal --unzip --path raw_data/deep-fashion-multimodal
```

### Disk Space Estimates

| Stage | Estimate |
|-------|---------|
| indo fashion download gb | ~4 GB  (106K images + JSON) |
| deep fashion download gb | ~15 GB (44K high-res images + annotations) |
| final dataset gb | ~0.5 GB (3200 images at 224×224 JPEG) |
| intermediate gb | ~1 GB  (candidate lists + clean copies) |

## 3. Source Dataset Statistics

| Dataset | Source Images | Classes | Type |
|---------|--------------|---------|------|
| IndoFashion | ~106,000 | 15 confirmed | Discrete folders + JSON metadata |
| DeepFashion-MM | ~44,096 | None (caption-derived) | captions.json + attribute annotations |

## 4. Usable / Corrupted / Duplicate Statistics

| Metric | Count |
|--------|-------|
| Final clean images | 3194 |
| Corrupted/invalid removed | 0 |
| Exact duplicates removed | 1 |
| Near-duplicates removed (pHash) | 5 |
| Ambiguous categories excluded | 9 |
| Edge-case candidates flagged | 614 |

## 5. Final Class Counts

| Class | Target | Actual | Δ |
|-------|--------|--------|---|
| Indian | 1500 | 1494 | -6 |
| Western | 1500 | 1500 | +0 |
| Indo-Western | 200 | 200 | +0 |
| **TOTAL** | **3200** | **3194** | - |

## 6. Category Distribution (Source → Final Class)

| Source Dataset | Original Category | Final Class | Reason | Count |
|---------------|-------------------|-------------|--------|-------|
| IndoFashion | saree | Indian | Quintessential traditional Indian drape | 350 |
| IndoFashion | women_kurta | Indian | Traditional Indian subcontinent top for women | 300 |
| IndoFashion | leggings_and_salwar | Indian | Salwar is a traditional Indian lower garment | 200 |
| IndoFashion | lehenga | Indian | Traditional Indian skirt-blouse combination | 300 |
| IndoFashion | men_kurta | Indian | Traditional Indian men's top | 200 |
| IndoFashion | sherwani | Indian | Formal traditional Indian men's attire | 150 |
| IndoFashion | dhoti_pants | Indo-Western | Dhoti-pant fuses Indian dhoti drape with Western trouser construction | 100 |
| IndoFashion | nehru_jacket | Indo-Western | Prototype Indo-Western fusion: Indian collar + Western jacket body | 100 |
| DeepFashion-MultiModal | shirt / t-shirt / polo | Western | Core Western casual tops | 250 |
| DeepFashion-MultiModal | dress / skirt | Western | Western feminine garments | 200 |
| DeepFashion-MultiModal | jeans / trousers / pants | Western | Western bottoms | 200 |
| DeepFashion-MultiModal | jacket / coat / blazer | Western | Western outerwear | 200 |
| DeepFashion-MultiModal | hoodie / sweater / cardigan | Western | Western knitwear/casualwear | 200 |
| DeepFashion-MultiModal | suit / formal wear | Western | Western formal attire | 200 |
| DeepFashion-MultiModal | shorts / tops / casual | Western | Western casualwear miscellaneous | 250 |
| DeepFashion-MultiModal | nehru / bandhgala / fusion | Indo-Western | Fusion keyword detected in caption | 0 |

### Excluded / Ambiguous Categories

| Source | Category | Reason Excluded |
|--------|----------|----------------|
| IndoFashion | palazzo | Palazzo pants are a Western silhouette adopted into Indian ethnic wear; cannot reliably assign India... |
| IndoFashion | gown | IndoFashion 'gown' includes both Western-style gowns and heavily embroidered Indian ethnic gowns. La... |
| IndoFashion | dupatta | Dupatta is an accessory/scarf, not a complete outfit. Excluded to avoid noise in outfit-level classi... |
| IndoFashion | blouse | In IndoFashion context, 'blouse' refers to the saree blouse (choli), which is a partial garment. Exc... |
| IndoFashion | petticoats | Undergarment worn beneath sarees. Not an outer outfit. Excluded.... |
| IndoFashion | women_mojari | Footwear (traditional Indian shoe). Not clothing. Excluded.... |
| IndoFashion | men_mojari | Footwear. Not clothing. Excluded.... |
| DeepFashion-MultiModal | gown (DeepFashion) | DeepFashion gowns could be Western ball gowns or fusion ethnic gowns. Will classify per caption: if ... |
| DeepFashion-MultiModal | top (generic) | 'Top' is too broad — could mean blouse, crop top, tank top. Will include if paired with other Wester... |

## 7. Train / Validation / Test Distribution

| Split | Indian | Western | Indo-Western | Total |
|-------|--------|---------|--------------|-------|
| train | 1046 | 1050 | 140 | 2236 |
| val | 224 | 225 | 30 | 479 |
| test | 224 | 225 | 30 | 479 |
| **Total** | **1494** | **1500** | **200** | **3194** |

Split ratios: train=70%, val=15%, test=15%

## 8. Image Resolution Statistics (after preprocessing)

| Metric | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| width | 224 | 224 | 224.0 | 224 |
| height | 224 | 224 | 224.0 | 224 |
| aspect_ratio | 0.2562 | 2.2069 | 0.6471 | 0.6812 |
| file_size_bytes | 5741 | 47095 | 15130.9562 | 14475 |

Output image size: 224×224 px (letterbox-padded JPEG)

## 9. Sampling Strategy

1. **IndoFashion** — 15 confirmed class labels are mapped to Indian/Indo-Western.
   - Categories mapped to **Indian**: saree, women_kurta, leggings_and_salwar,
     lehenga, men_kurta, sherwani
   - Categories mapped to **Indo-Western**: dhoti_pants, nehru_jacket
   - Sampling uses per-category quotas to ensure diversity across categories.
   - Maximum quota per category prevents any single category from dominating.

2. **DeepFashion-MultiModal** — No discrete garment categories exist.
   - Images are classified by keyword-matching on `captions.json` entries.
   - Keywords are priority-ordered: Indo-Western > Indian > Western.
   - DeepFashion primarily supplies **Western** class images (1,500 target).
   - IndoFashion supplies all **Indian** (1,350) and partial **Indo-Western** (200).

## 10. Cleaning Strategy

For each candidate image:

1. Verify file exists on disk
2. Open with Pillow + `verify()` call to detect truncation/corruption
3. Reload and `load()` to catch lazy-load failures
4. Check minimum dimension ≥ 32px on both axes
5. Check aspect ratio ≤ 5.0:1
6. Convert to RGB (RGBA → composite on white, grayscale → RGB)
7. Letterbox-resize to 224×224 (LANCZOS, black padding)
8. Save as JPEG quality=95 with subsampling=0

## 11. Duplicate Detection Strategy

**Pass 1 — Exact duplicates (MD5)**
- Compute MD5 hash of raw file bytes
- Keep first occurrence; flag remainder

**Pass 2 — Near-duplicates (pHash)**
- Compute perceptual hash (average-hash variant, 16×16 bits)
- Brute-force O(n²) Hamming distance comparison across all pairs
- Flag pairs with Hamming distance ≤ 6
- Keep first occurrence; flag remainder
- Applied BEFORE the train/val/test split to prevent leakage

## 12. Edge-Case Profiling

| Label | Meaning | How Assigned |
|-------|---------|-------------|
| normal | No detected edge-case properties | Default |
| hybrid | Confirmed fusion garment | All Indo-Western class images |
| visually_similar | May be confused across classes | Heuristic (caption keywords + AR) |
| ambiguous | Cannot be cleanly assigned | Fusion keywords in Indian/Western image |

> **Note:** Automatic edge-case labels are heuristic. `edge_case_candidates.csv`
> lists all non-normal images for optional manual review.

## 13. Anti-Leakage Measures

- Deduplication (both exact and near-duplicate) is performed **before** splitting
- Stratified split preserves class proportions in each partition
- Fixed random seed (`SEED = 42`) ensures reproducibility
- No image appears in more than one split

## 14. Potential Limitations

1. **DeepFashion classification relies on captions** — if a Western garment
   caption contains no recognisable Western keywords, it will be missed.
   Manual spot-checking of a random 5% sample is recommended.

2. **IndoFashion gown and palazzo categories** are excluded because the label
   alone does not distinguish Indian ethnic gowns from Western gowns or Indian
   ethnic palazzo pants from Western-derived ones. This exclusion is documented
   in `ambiguous_excluded.csv`.

3. **Indo-Western class is small (200 images)** — this reflects the genuine
   scarcity of reliably labelled fusion garment data. Do not upscale this class
   by randomly mixing Indian and Western images; use class-weighted loss instead.

4. **No demographic filtering is applied** — classification is based on garment
   characteristics, not on person attributes. However, source datasets are not
   demographically balanced and this should be noted in any publication.

5. **pHash near-duplicate detection may miss non-standard duplicates** (e.g.,
   same garment photographed on different backgrounds). Caption similarity checks
   on product titles could supplement pHash for the Indo-Fashion dataset.

## 15. Reproducibility Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place Kaggle credentials
# Windows: %USERPROFILE%\.kaggle\kaggle.json

# 3. Run the full pipeline (produces identical output with seed=42)
py -3 prepare_dataset.py

# 4. Validate the dataset
py -3 validate_dataset.py

# 5. Dry-run (inspection only, no file writes)
py -3 prepare_dataset.py --dry-run
```

## 16. PyTorch DataLoader Usage

```python
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Normalisation statistics (ImageNet defaults — suitable for pretrained models)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((224, 224)),          # already 224×224; kept for safety
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean, std),
])

val_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_ds = ImageFolder('dataset/train', transform=train_tf)
val_ds   = ImageFolder('dataset/val',   transform=val_tf)
test_ds  = ImageFolder('dataset/test',  transform=val_tf)

# Class order: {'Indian': 0, 'Indo-Western': 1, 'Western': 2} (alphabetical)
print(train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)
```

### Class Imbalance Handling

```python
import torch
# Compute class weights (inverse frequency)
counts = torch.tensor([train_ds.targets.count(i) for i in range(3)], dtype=torch.float)
weights = 1.0 / counts
weights /= weights.sum()
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
```

---
*Report generated by `09_generate_report.py` on 2026-09-28 16:21*