"""
03_build_category_mapping.py
=============================
Builds and documents:
  - category_mapping.csv   : which source categories map to which final class
  - ambiguous_excluded.csv : categories not assigned, with reasons

Mapping is grounded on:
  • IndoFashion's 15 confirmed class labels (CVPRW 2021 paper)
  • DeepFashion-MultiModal's caption-keyword structure (no discrete category folders)

Decision rationale is documented inline and written to the CSVs.

Usage:
  py -3 scripts/03_build_category_mapping.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_META, DEEPFASHION_IW_KEYWORDS

DATASET_META.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  INDOFASHION MAPPING
#  Reference: Rajput & Aneja, CVPRW 2021
#  15 classes → Indian / Western / Indo-Western / EXCLUDED
#
#  Decision notes:
#    saree            → Indian   (quintessential Indian drape)
#    women_kurta      → Indian   (traditional Indo-subcontinent top)
#    leggings_salwar  → Indian   (salwar is traditional; leggings are accessories)
#    palazzo          → AMBIGUOUS (palazzo pants are a Western silhouette now used
#                        widely in Indian ethnic wear; cannot reliably assign from
#                        label alone without image inspection)
#    lehenga          → Indian   (traditional skirt-blouse combination)
#    dupatta          → EXCLUDED (accessory, not a standalone outfit)
#    blouse           → EXCLUDED (often just the saree blouse; too generic as a label)
#    gown             → AMBIGUOUS (may be Western gown OR Indian ethnic gown with
#                        embroidery; cannot distinguish from label alone)
#    dhoti_pants      → Indo-Western (dhoti-pant is a fusion silhouette — Indian
#                        dhoti drape adapted into a Western trouser-like construction,
#                        widely used in Indo-Western fashion)
#    petticoats       → EXCLUDED (undergarment/inner layer, not an outer outfit)
#    women_mojari     → EXCLUDED (footwear, not clothing)
#    men_kurta        → Indian   (traditional Indian men's top)
#    nehru_jacket     → Indo-Western (prototype Indo-Western fusion piece —
#                        Indian-origin collar + Western jacket construction)
#    sherwani         → Indian   (formal Indian men's outfit)
#    men_mojari       → EXCLUDED (footwear)
# ─────────────────────────────────────────────────────────────

INDOFASHION_MAPPING = [
    # (original_category, final_class, reason, target_count)
    ("saree",             "Indian",       "Quintessential traditional Indian drape",                                 350),
    ("women_kurta",       "Indian",       "Traditional Indian subcontinent top for women",                           300),
    ("leggings_and_salwar","Indian",      "Salwar is a traditional Indian lower garment",                            200),
    ("lehenga",           "Indian",       "Traditional Indian skirt-blouse combination",                             300),
    ("men_kurta",         "Indian",       "Traditional Indian men's top",                                            200),
    ("sherwani",          "Indian",       "Formal traditional Indian men's attire",                                  150),
    ("dhoti_pants",       "Indo-Western", "Dhoti-pant fuses Indian dhoti drape with Western trouser construction",   100),
    ("nehru_jacket",      "Indo-Western", "Prototype Indo-Western fusion: Indian collar + Western jacket body",      100),
]

INDOFASHION_AMBIGUOUS = [
    # (original_category, reason_excluded)
    ("palazzo",    "Palazzo pants are a Western silhouette adopted into Indian ethnic wear; "
                   "cannot reliably assign Indian vs Western from category label alone. "
                   "Would require image-level inspection."),
    ("gown",       "IndoFashion 'gown' includes both Western-style gowns and heavily embroidered "
                   "Indian ethnic gowns. Label alone is insufficient — requires per-image review."),
    ("dupatta",    "Dupatta is an accessory/scarf, not a complete outfit. Excluded to avoid "
                   "noise in outfit-level classification."),
    ("blouse",     "In IndoFashion context, 'blouse' refers to the saree blouse (choli), which is "
                   "a partial garment. Excluding to avoid misleading Western blouse confusion."),
    ("petticoats", "Undergarment worn beneath sarees. Not an outer outfit. Excluded."),
    ("women_mojari","Footwear (traditional Indian shoe). Not clothing. Excluded."),
    ("men_mojari", "Footwear. Not clothing. Excluded."),
]

# ─────────────────────────────────────────────────────────────
#  DEEPFASHION-MULTIMODAL MAPPING
#
#  DeepFashion-MultiModal does NOT have discrete garment category
#  folders. Classification is done by keyword-matching on image
#  captions/product titles.
#
#  Source: Jiang et al., Text2Human (SIGGRAPH 2022); Kaggle mirror
#  Annotation: captions.json + texture_ann/ + shape_ann/
#
#  The dataset contains ~44,096 images of persons in Western-style
#  clothing (sourced from fashion magazines and shopping sites,
#  predominantly Western fashion context).
#
#  Category derivation strategy:
#    - Parse captions.json for keywords
#    - Match Western apparel terms → Western class
#    - Match Indo-Western keywords → Indo-Western class (rare)
#    - Match Indian apparel terms  → Indian class (rare, supplement only)
#
#  Western keyword examples (from actual captions observed in dataset):
#    shirt, t-shirt, jeans, dress, skirt, coat, jacket, blazer,
#    trousers, shorts, hoodie, sweater, top, suit, gown (Western)
# ─────────────────────────────────────────────────────────────

DEEPFASHION_MAPPING = [
    # (keyword_group, final_class, reason, target_count)
    ("shirt / t-shirt / polo",      "Western",      "Core Western casual tops",                          250),
    ("dress / skirt",               "Western",      "Western feminine garments",                         200),
    ("jeans / trousers / pants",    "Western",      "Western bottoms",                                   200),
    ("jacket / coat / blazer",      "Western",      "Western outerwear",                                 200),
    ("hoodie / sweater / cardigan", "Western",      "Western knitwear/casualwear",                       200),
    ("suit / formal wear",          "Western",      "Western formal attire",                             200),
    ("shorts / tops / casual",      "Western",      "Western casualwear miscellaneous",                  250),
    ("nehru / bandhgala / fusion",  "Indo-Western", "Fusion keyword detected in caption",                 0),  # supplementary
]

DEEPFASHION_AMBIGUOUS = [
    ("gown (DeepFashion)",  "DeepFashion gowns could be Western ball gowns or fusion ethnic gowns. "
                            "Will classify per caption: if caption contains 'ethnic/Indian/traditional' → "
                            "exclude; otherwise → Western. This is documented in the sampling step."),
    ("top (generic)",       "'Top' is too broad — could mean blouse, crop top, tank top. "
                            "Will include if paired with other Western context keywords."),
]


def write_mapping_csv():
    mapping_path = DATASET_META / "category_mapping.csv"
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_dataset", "original_category", "final_class",
            "reason", "selected_count"
        ])
        writer.writeheader()

        for row in INDOFASHION_MAPPING:
            writer.writerow({
                "source_dataset":     "IndoFashion",
                "original_category":  row[0],
                "final_class":        row[1],
                "reason":             row[2],
                "selected_count":     row[3],
            })

        for row in DEEPFASHION_MAPPING:
            writer.writerow({
                "source_dataset":     "DeepFashion-MultiModal",
                "original_category":  row[0],
                "final_class":        row[1],
                "reason":             row[2],
                "selected_count":     row[3],
            })

    print(f"  Written → {mapping_path}")
    return mapping_path


def write_ambiguous_csv():
    ambiguous_path = DATASET_META / "ambiguous_excluded.csv"
    with open(ambiguous_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_dataset", "original_category", "reason_excluded"
        ])
        writer.writeheader()

        for row in INDOFASHION_AMBIGUOUS:
            writer.writerow({
                "source_dataset":     "IndoFashion",
                "original_category":  row[0],
                "reason_excluded":    row[1],
            })
        for row in DEEPFASHION_AMBIGUOUS:
            writer.writerow({
                "source_dataset":     "DeepFashion-MultiModal",
                "original_category":  row[0],
                "reason_excluded":    row[1],
            })

    print(f"  Written → {ambiguous_path}")
    return ambiguous_path


def print_summary():
    from collections import Counter
    class_counts = Counter()
    print("\n  Confirmed mapping summary:")
    print(f"  {'Dataset':<25} {'Category':<28} {'→'} {'Class':<15} {'Target':>7}")
    print("  " + "─" * 80)
    for row in INDOFASHION_MAPPING:
        cat, cls, _, cnt = row
        class_counts[cls] += cnt
        print(f"  {'IndoFashion':<25} {cat:<28} → {cls:<15} {cnt:>7}")
    for row in DEEPFASHION_MAPPING:
        kw, cls, _, cnt = row
        class_counts[cls] += cnt
        print(f"  {'DeepFashion-MM':<25} {kw:<28} → {cls:<15} {cnt:>7}")
    print("\n  Projected counts from mapping:")
    for cls, total in class_counts.items():
        print(f"    {cls:<15} {total:>6}")
    print()
    print("  NOTE: Final counts are determined by sampling logic in step 04.")
    print("        These are directional targets per category group.")

    print(f"\n  Excluded / Ambiguous categories:")
    for row in INDOFASHION_AMBIGUOUS + DEEPFASHION_AMBIGUOUS:
        print(f"    [{row[0]}]  — see ambiguous_excluded.csv for reason")


def main():
    print("=" * 60)
    print("  STEP 03 — Build Category Mapping")
    print("=" * 60)

    mapping_path   = write_mapping_csv()
    ambiguous_path = write_ambiguous_csv()
    print_summary()

    print("\n[DONE] Category mapping complete.")
    print(f"  category_mapping.csv    → {mapping_path}")
    print(f"  ambiguous_excluded.csv  → {ambiguous_path}")


if __name__ == "__main__":
    main()
