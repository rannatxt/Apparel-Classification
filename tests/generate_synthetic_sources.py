"""
tests/generate_synthetic_sources.py
=====================================
Creates synthetic test fixtures that mimic the real IndoFashion and
DeepFashion-MultiModal dataset structures. Allows the full pipeline
to be smoke-tested without downloading the real datasets.

Features:
  - Generates textured JPEG images (not flat-colour, to avoid false
    pHash collisions)
  - Creates IndoFashion-style JSON files (train.json / val.json)
  - Creates DeepFashion-style captions.json
  - Uses SMALL targets so the test runs quickly

Usage:
  py -3 tests/generate_synthetic_sources.py
  py -3 tests/generate_synthetic_sources.py --clean   # remove synthetic data first
"""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR,
    INDOFASHION_DIR,
    DEEPFASHION_DIR,
    SEED,
    INDOFASHION_ALL_CLASSES,
)

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[ERROR] Pillow is required. Run: pip install Pillow")
    sys.exit(1)

rng = random.Random(SEED)

# ── Reduced targets for fast smoke tests ──────────────────────
SYNTH_COUNTS_PER_CLASS = 60   # IndoFashion records per class
SYNTH_DEEPFASHION_IMGS = 300   # DeepFashion images
IMAGE_SIZE = (224, 224)


def make_textured_image(seed_val: int, hue_deg: float) -> Image.Image:
    """
    Generate a visually distinct JPEG-safe RGB image with:
      - A gradient background in a hue derived from hue_deg
      - Random noise overlay
      - A simple geometric shape
    This ensures each image has a unique pHash.
    """
    rng_img = random.Random(seed_val)
    w, h = IMAGE_SIZE

    # Base HSV→RGB
    import colorsys
    hue = (hue_deg / 360.0) % 1.0
    sat = 0.5 + rng_img.random() * 0.4
    val = 0.5 + rng_img.random() * 0.4
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    base_color = (int(r * 255), int(g * 255), int(b * 255))

    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)

    # Gradient strips
    for i in range(0, h, 8):
        factor = (i / h) * 0.5 + 0.5
        strip_color = tuple(int(c * factor) for c in base_color)
        draw.rectangle([0, i, w, i + 7], fill=strip_color)

    # Random shapes (gives pHash texture)
    for _ in range(8 + rng_img.randint(0, 8)):
        x0 = rng_img.randint(0, w - 30)
        y0 = rng_img.randint(0, h - 30)
        x1 = x0 + rng_img.randint(15, 60)
        y1 = y0 + rng_img.randint(15, 60)
        shape_color = tuple(rng_img.randint(0, 255) for _ in range(3))
        shape_type = rng_img.choice(["rect", "ellipse", "line"])
        if shape_type == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=shape_color, outline=None)
        elif shape_type == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=shape_color)
        else:
            draw.line([x0, y0, x1, y1], fill=shape_color, width=3)

    # Diagonal gradient overlay for uniqueness
    pixels = img.load()
    for py in range(h):
        for px in range(w):
            factor = 0.85 + 0.15 * math.sin(px * 0.05 + py * 0.03 + seed_val * 0.001)
            r_, g_, b_ = pixels[px, py]
            pixels[px, py] = (
                min(255, int(r_ * factor)),
                min(255, int(g_ * factor)),
                min(255, int(b_ * factor)),
            )

    return img


def generate_indofashion():
    """Create a synthetic IndoFashion dataset structure."""
    print("\n  Generating synthetic IndoFashion dataset...")
    INDOFASHION_DIR.mkdir(parents=True, exist_ok=True)

    img_root = INDOFASHION_DIR / "images"
    img_root.mkdir(exist_ok=True)

    records_train = []
    records_val   = []

    global_idx = 0
    hue_step = 360.0 / len(INDOFASHION_ALL_CLASSES)

    for class_idx, class_name in enumerate(INDOFASHION_ALL_CLASSES):
        class_dir = img_root / class_name
        class_dir.mkdir(exist_ok=True)
        hue_base = class_idx * hue_step

        n_train = int(SYNTH_COUNTS_PER_CLASS * 0.8)
        n_val   = SYNTH_COUNTS_PER_CLASS - n_train

        for i in range(SYNTH_COUNTS_PER_CLASS):
            img_name = f"{class_name}_{i:04d}.jpg"
            img_path = class_dir / img_name
            hue = hue_base + rng.uniform(0, hue_step * 0.8)
            img = make_textured_image(seed_val=global_idx * 7 + i * 13,
                                      hue_deg=hue)
            img.save(img_path, format="JPEG", quality=90)

            rec = {
                "image_url":     f"http://synthetic.example.com/{class_name}/{img_name}",
                "image_path":    f"images/{class_name}/{img_name}",
                "brand":         f"Brand_{class_idx % 5}",
                "product_title": f"Synthetic {class_name.replace('_', ' ').title()} #{i}",
                "class_label":   class_name,
            }

            if i < n_train:
                records_train.append(rec)
            else:
                records_val.append(rec)

        global_idx += SYNTH_COUNTS_PER_CLASS
        print(f"    {class_name:<30} {SYNTH_COUNTS_PER_CLASS} images", end="\r")

    print(f"\n    Total IndoFashion images: {global_idx}")

    with open(INDOFASHION_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(records_train, f)
    with open(INDOFASHION_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(records_val, f)

    print(f"    train.json: {len(records_train)} records")
    print(f"    val.json:   {len(records_val)} records")


def generate_deepfashion():
    """Create a synthetic DeepFashion-MultiModal dataset structure."""
    print("\n  Generating synthetic DeepFashion-MultiModal dataset...")
    DEEPFASHION_DIR.mkdir(parents=True, exist_ok=True)

    img_root = DEEPFASHION_DIR / "images"
    img_root.mkdir(exist_ok=True)

    WESTERN_CAPTIONS = [
        "a person wearing a {item}",
        "model in {item} and jeans",
        "{item} outfit on white background",
        "casual {item} for everyday wear",
    ]
    WESTERN_ITEMS = [
        "t-shirt", "shirt", "dress", "skirt", "jeans", "trousers",
        "jacket", "blazer", "coat", "hoodie", "sweater", "shorts",
        "polo shirt", "cardigan", "suit", "blouse", "top",
    ]
    IW_CAPTIONS = [
        "indo western fusion {item}",
        "nehru jacket with trousers",
        "ethnic fusion outfit",
        "bandhgala jacket on model",
    ]
    INDIAN_CAPS = [
        "saree on model", "kurta with churidar",
        "lehenga choli outfit", "traditional ethnic wear",
    ]

    captions = {}
    for i in range(SYNTH_DEEPFASHION_IMGS):
        img_name = f"deepfashion_{i:05d}.jpg"
        hue = rng.uniform(0, 360)
        img = make_textured_image(seed_val=i * 31 + 1000, hue_deg=hue)
        (img_root / img_name).parent.mkdir(exist_ok=True)
        img.save(img_root / img_name, format="JPEG", quality=90)

        # Assign caption
        if i < int(SYNTH_DEEPFASHION_IMGS * 0.70):
            item = rng.choice(WESTERN_ITEMS)
            cap  = rng.choice(WESTERN_CAPTIONS).format(item=item)
        elif i < int(SYNTH_DEEPFASHION_IMGS * 0.85):
            cap = rng.choice(IW_CAPTIONS)
        else:
            cap = rng.choice(INDIAN_CAPS)

        captions[img_name] = cap

    with open(DEEPFASHION_DIR / "captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f)

    print(f"    Generated {SYNTH_DEEPFASHION_IMGS} images")
    print(f"    captions.json: {len(captions)} entries")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset fixtures")
    parser.add_argument("--clean", action="store_true",
                        help="Remove synthetic data before generating")
    args = parser.parse_args()

    print("=" * 60)
    print("  SYNTHETIC TEST FIXTURE GENERATOR")
    print("=" * 60)

    if args.clean:
        print("\n  Cleaning synthetic data...")
        if INDOFASHION_DIR.exists():
            shutil.rmtree(INDOFASHION_DIR)
            print(f"  Removed {INDOFASHION_DIR}")
        if DEEPFASHION_DIR.exists():
            shutil.rmtree(DEEPFASHION_DIR)
            print(f"  Removed {DEEPFASHION_DIR}")

    generate_indofashion()
    generate_deepfashion()

    print(f"\n  [DONE] Synthetic datasets created at:")
    print(f"    {INDOFASHION_DIR}")
    print(f"    {DEEPFASHION_DIR}")
    print(f"\n  You can now run:")
    print(f"    py -3 prepare_dataset.py --skip-download")


if __name__ == "__main__":
    main()
