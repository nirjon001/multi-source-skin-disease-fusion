"""
CSE475 Skin Disease Classification - Cross-dataset label harmonizer (Phase 2)

Builds results/label_map.csv mapping every source class folder name to a
unified class. Verifies that each expected folder actually exists on disk
before anything depends on the mapping.

Datasets handled:
  1. DermNet 23-class (train set - universe of model classes)
  2. SkinDiseaseBD (5 classes, test set)
  3. Fitzpatrick17k black (114 classes, only the 3-5 that overlap are used at test)

Usage:
  py -3.11 src/harmonize.py --dermnet-root "F:/Downloads/...dataset/train" --skindiseasebd-root "F:/cse475_skin/data/SkinDiseaseBD/Updated Images" --fitzpatrick-root "F:/Downloads/.../fitzpatrick-black-images" --out results/label_map.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Harmonization map: source folder name(s) -> unified class
# The paper's unified 5-class core (plus acne) lives in section 4 of
# docs/DEEPSEEK_CONVERSATION_PLAN.md.
# ---------------------------------------------------------------------------

DERMNET_UNIFIED = {
    "Eczema Photos": "eczema",
    "Scabies Lyme Disease and other Infestations and Bites": "scabies",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "tinea",
    "Atopic Dermatitis Photos": "dermatitis",
    "Poison Ivy Photos and other Contact Dermatitis": "dermatitis",
    "Light Diseases and Disorders of Pigmentation": "vitiligo",
    "Acne and Rosacea Photos": "acne",
}

SKINDISEASEBD_UNIFIED = {
    "Eczema": "eczema",
    "Scabies": "scabies",
    "Tinea Ringworm": "tinea",
    "Dermatitis": "dermatitis",
    "Vitiligo": "vitiligo",
}

FITZPATRICK_UNIFIED = {
    "acne": "acne",
    "acne_vulgaris": "acne",
    "eczema": "eczema",
    "dyshidrotic_eczema": "eczema",
    "allergic_contact_dermatitis": "dermatitis",
}


def _folder_map(root: Path) -> dict[str, str]:
    """Return {folder_name: unified_class} for folders present on disk."""
    if not root.exists():
        raise SystemExit(f"[FATAL] dataset root not found: {root}")
    dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    return {d: d for d in dirs}


def _apply(mapping: dict[str, str], folder_names: dict[str, str], dataset: str) -> list[tuple[str, str, str]]:
    """Build rows, verifying every mapped folder exists. Unknown folders pass through unchanged."""
    rows = []
    missing = []
    for folder, unified in mapping.items():
        if folder not in folder_names:
            missing.append(folder)
            continue
        rows.append((dataset, folder, unified))
    # Folders on disk not mentioned in the map -> identity mapping
    for folder in folder_names:
        if folder not in mapping:
            rows.append((dataset, folder, folder))
    if missing:
        print(f"[WARN] {dataset}: expected folders missing on disk: {missing}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-dataset label map")
    ap.add_argument("--dermnet-root", required=True)
    ap.add_argument("--skindiseasebd-root", required=True)
    ap.add_argument("--fitzpatrick-root", required=True)
    ap.add_argument("--out", default="results/label_map.csv")
    args = ap.parse_args()

    dermnet = _folder_map(Path(args.dermnet_root))
    sdb = _folder_map(Path(args.skindiseasebd_root))
    fitz = _folder_map(Path(args.fitzpatrick_root))

    rows: list[tuple[str, str, str]] = []
    rows += _apply(DERMNET_UNIFIED, dermnet, "dermnet")
    rows += _apply(SKINDISEASEBD_UNIFIED, sdb, "skindiseasebd")
    rows += _apply(FITZPATRICK_UNIFIED, fitz, "fitzpatrick_black")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "source_class", "unified_class"])
        w.writerows(rows)

    print(f"[MAP] {len(rows)} rows -> {out}")
    print("[MAP] DermNet folders:", len(dermnet), "| SkinDiseaseBD:", len(sdb),
          "| Fitzpatrick:", len(fitz))
    # Quick sanity printout of the unified 5-class core
    core = {r[2] for r in rows if r[2] in {"eczema", "scabies", "tinea", "dermatitis", "vitiligo", "acne"}}
    print("[MAP] unified core present:", sorted(core))


if __name__ == "__main__":
    main()