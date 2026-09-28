"""
config.py — Central configuration for the Apparel Classification Dataset Pipeline
===================================================================================
Project: Robust Indian, Western, and Indo-Western Apparel Classification
         with Edge-Case Profiling for Inclusive E-Commerce

Students:
  1. Swaroop Vijaykumar Purandare  — 1GA23CS176
  2. Ranna                         — 1GA23CS135
  3. SUKESH                        — 1GA23CS171
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
RAW_DATA_DIR  = BASE_DIR / "raw_data"
DATASET_DIR   = BASE_DIR / "dataset"
SCRIPTS_DIR   = BASE_DIR / "scripts"
TESTS_DIR     = BASE_DIR / "tests"

INDOFASHION_DIR  = RAW_DATA_DIR / "indo-fashion-dataset"
DEEPFASHION_DIR  = RAW_DATA_DIR / "deep-fashion-multimodal"

DATASET_TRAIN   = DATASET_DIR / "train"
DATASET_VAL     = DATASET_DIR / "val"
DATASET_TEST    = DATASET_DIR / "test"
DATASET_META    = DATASET_DIR / "metadata"
DATASET_REPORTS = DATASET_DIR / "reports"

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────
#  Target counts (final dataset)
# ─────────────────────────────────────────────
TARGET_COUNTS = {
    "Indian":       1500,
    "Western":      1500,
    "Indo-Western":  200,
}
TOTAL_TARGET = sum(TARGET_COUNTS.values())   # 3200

# ─────────────────────────────────────────────
#  Train / Val / Test split ratios
# ─────────────────────────────────────────────
SPLIT_RATIOS = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15,
}

# ─────────────────────────────────────────────
#  Image settings
# ─────────────────────────────────────────────
TARGET_SIZE       = (224, 224)       # pixels
OUTPUT_FORMAT     = "JPEG"
JPEG_QUALITY      = 95
MIN_DIMENSION     = 32              # pixels — images smaller than this are discarded
MAX_ASPECT_RATIO  = 5.0             # width/height or height/width — images beyond this are discarded

# ─────────────────────────────────────────────
#  Duplicate detection
# ─────────────────────────────────────────────
PHASH_BITS       = 16               # perceptual hash bit size
PHASH_THRESHOLD  = 6                # Hamming distance threshold for near-duplicates

# ─────────────────────────────────────────────
#  Kaggle dataset identifiers
# ─────────────────────────────────────────────
KAGGLE_INDOFASHION  = "validmodel/indo-fashion-dataset"
KAGGLE_DEEPFASHION  = "silverstone1903/deep-fashion-multimodal"

# ─────────────────────────────────────────────
#  Naming prefixes
# ─────────────────────────────────────────────
CLASS_PREFIXES = {
    "Indian":       "IND",
    "Western":      "WEST",
    "Indo-Western": "IW",
}

# ─────────────────────────────────────────────
#  IndoFashion — confirmed 15 class labels
#  Source: Rajput & Aneja, CVPRW 2021
# ─────────────────────────────────────────────
INDOFASHION_ALL_CLASSES = [
    "saree",
    "women_kurta",
    "leggings_and_salwar",
    "palazzo",
    "lehenga",
    "dupatta",
    "blouse",
    "gown",
    "dhoti_pants",
    "petticoats",
    "women_mojari",
    "men_kurta",
    "nehru_jacket",
    "sherwani",
    "men_mojari",
]

# ─────────────────────────────────────────────
#  DeepFashion-MultiModal — caption keyword sets
#  (no discrete category folders; classification
#   done via keyword-matching on captions/titles)
# ─────────────────────────────────────────────
DEEPFASHION_WESTERN_KEYWORDS = [
    "t-shirt", "tshirt", "shirt", "blouse",
    "jeans", "denim", "trousers", "pants", "chinos",
    "skirt", "dress", "gown", "coat", "jacket",
    "suit", "blazer", "hoodie", "sweatshirt",
    "shorts", "polo", "cardigan", "sweater",
    "top", "crop top", "tank top",
]
DEEPFASHION_IW_KEYWORDS = [
    "indo western", "indo-western", "fusion",
    "nehru", "ethnic fusion", "kurta with jeans",
    "anarkali", "indo", "bandhgala",
]
DEEPFASHION_INDIAN_KEYWORDS = [
    "saree", "sari", "salwar", "kameez", "kurta",
    "lehenga", "dupatta", "sherwani", "dhoti",
    "churidar", "patiala", "ethnic",
]

# ─────────────────────────────────────────────
#  Disk space estimates (informational)
# ─────────────────────────────────────────────
DISK_ESTIMATE = {
    "indo_fashion_download_gb":   "~4 GB  (106K images + JSON)",
    "deep_fashion_download_gb":   "~15 GB (44K high-res images + annotations)",
    "final_dataset_gb":           "~0.5 GB (3200 images at 224×224 JPEG)",
    "intermediate_gb":            "~1 GB  (candidate lists + clean copies)",
}
