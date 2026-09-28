# Apparel Classification Dataset Pipeline

**Project:** Robust Indian, Western, and Indo-Western Apparel Classification  
**with Edge-Case Profiling for Inclusive E-Commerce**

## Students

| # | Name | USN |
|---|------|-----|
| 1 | Swaroop Vijaykumar Purandare | 1GA23CS176 |
| 2 | Ranna | 1GA23CS135 |
| 3 | SUKESH | 1GA23CS171 |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials (one-time)

Download `kaggle.json` from https://www.kaggle.com/settings → API → **Create New Token**

Place it at:
- **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
- **Linux / Mac:** `~/.kaggle/kaggle.json`  *(then `chmod 600 ~/.kaggle/kaggle.json`)*

Alternatively, set environment variables:
```bash
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_api_key
```

### 3. Run the pipeline

```bash
py -3 prepare_dataset.py
```

### 4. Validate

```bash
py -3 validate_dataset.py
```

Expected result: `OVERALL: PASS` (16/16 checks).

---

## Dataset Targets

| Class | Count |
|-------|-------|
| Indian | 1,500 |
| Western | 1,500 |
| Indo-Western | 200 |
| **Total** | **3,200** |

Split: 70% train / 15% val / 15% test — stratified by class, seed = 42.

---

## Source Datasets

| Dataset | Kaggle Slug | Size |
|---------|------------|------|
| IndoFashion | `validmodel/indo-fashion-dataset` | ~4 GB, 106K images, 15 classes |
| DeepFashion-MultiModal | `silverstone1903/deep-fashion-multimodal` | ~15 GB, 44K images + captions |

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_download_datasets.py` | Download from Kaggle (skip if already downloaded) |
| 02 | `02_inspect_datasets.py` | Inspect structure, class counts, image dimensions |
| 03 | `03_build_category_mapping.py` | Document category → class mapping with exclusion reasons |
| 04 | `04_sample_dataset.py` | Stratified sampling by category |
| 05 | `05_clean_dataset.py` | Validate, convert to RGB, resize to 224×224 JPEG |
| 06 | `06_detect_duplicates.py` | MD5 exact + pHash near-duplicate removal |
| 07 | `07_create_splits.py` | 70/15/15 stratified train/val/test split |
| 08 | `08_generate_metadata.py` | Write all metadata CSVs |
| 09 | `09_generate_report.py` | Generate `dataset_report.md` |

Run individual steps:
```bash
py -3 scripts/02_inspect_datasets.py
py -3 scripts/04_sample_dataset.py --dry-run
```

Resume the pipeline from a step:
```bash
py -3 prepare_dataset.py --from 5 --to 9
py -3 prepare_dataset.py --skip-download
```

---

## Testing Without Kaggle Credentials

```bash
# Generate ~1200 synthetic images (fast, no download required)
py -3 tests/generate_synthetic_sources.py

# Run pipeline skipping download
py -3 prepare_dataset.py --skip-download

# Validate in synthetic mode (skips count checks 1-4)
py -3 validate_dataset.py --synthetic
```

Expected result: `OVERALL: PASS` (16/16 including 4 skipped count checks).

---

## PyTorch ImageFolder Loading

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = ImageFolder("dataset/train", transform=train_tf)
val_ds   = ImageFolder("dataset/val",   transform=val_tf)
test_ds  = ImageFolder("dataset/test",  transform=val_tf)

# Alphabetical order: Indian=0, Indo-Western=1, Western=2
print(train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

# Class-weighted loss (recommended — Indo-Western is a smaller class)
import torch
counts  = torch.tensor([train_ds.targets.count(i) for i in range(3)], dtype=torch.float)
weights = (1.0 / counts) / (1.0 / counts).sum()
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
```

---

## Output Structure

```
dataset/
├── train/  Indian/  Western/  Indo-Western/
├── val/    Indian/  Western/  Indo-Western/
├── test/   Indian/  Western/  Indo-Western/
├── metadata/
│   ├── dataset.csv               (all images, full metadata)
│   ├── train.csv / val.csv / test.csv
│   ├── category_mapping.csv      (with documented reasons per category)
│   ├── ambiguous_excluded.csv    (categories excluded with reasons)
│   ├── edge_case_candidates.csv  (flagged for review)
│   ├── duplicate_report.csv
│   └── corrupted_images.csv
└── reports/
    ├── dataset_report.md         (comprehensive report)
    ├── class_distribution.csv
    └── image_statistics.csv
```

---

## Reproducibility

- Fixed seed: `SEED = 42` in `config.py`
- Re-running on the same raw data produces identical output
- All parameters in `config.py` — no magic numbers scattered in scripts
