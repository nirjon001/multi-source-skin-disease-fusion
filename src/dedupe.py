"""
CSE475 Skin Disease Classification - Near-duplicate detector (Phase 2)

Leakage guard for the cross-dataset evaluation. Finds near-identical images
WITHIN and ACROSS datasets using perceptual hashing (imagehash.phash,
Hamming distance <= threshold). If the same photo appears in DermNet and
SkinDiseaseBD, cross-dataset accuracy is inflated and the paper is invalid,
so this MUST run before eval_cross.py.

First pass is --dry-run (default): report only, never delete.

Usage:
  py -3.11 src/dedupe.py --roots "F:/Downloads/.../train;F:/cse475_skin/data/SkinDiseaseBD/Updated Images" --out results/dedupe_report.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import imagehash
    from PIL import Image
except ImportError:
    print("[FATAL] need: pip install imagehash pillow")
    sys.exit(1)


def walk_images(root: Path, max_files: int | None = None) -> list[Path]:
    """Return image files under root, parsed as grayscale-safe RGB."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Perceptual near-duplicate detection across datasets")
    ap.add_argument("--roots", required=True, help="semicolon-separated dataset roots")
    ap.add_argument("--threshold", type=int, default=5, help="phash Hamming distance threshold")
    ap.add_argument("--out", default="results/dedupe_report.json")
    ap.add_argument("--dry-run", action="store_true", default=True, help="report only, do not delete")
    ap.add_argument("--max-files", type=int, default=None, help="cap total images (testing)")
    args = ap.parse_args()

    roots = [Path(r.strip()) for r in args.roots.split(";") if r.strip()]
    all_files = []
    per_dataset: dict[str, int] = {}
    for r in roots:
        files = walk_images(r, args.max_files)
        per_dataset[str(r)] = len(files)
        all_files.extend(files)
        print(f"[DEDUPE] {r}: {len(files)} images")

    print(f"[DEDUPE] hashing {len(all_files)} images...")
    hashes = {}
    duplicates = []
    for i, p in enumerate(all_files):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                h = imagehash.phash(im)
            hashes.setdefault(h, []).append(p)
        except Exception as e:
            print(f"[DEDUPE] skip {p}: {e}")
        if (i + 1) % 500 == 0:
            print(f"[DEDUPE] ... {i + 1}/{len(all_files)} hashed")

    # Within-exact-bucket keeps first, flags rest
    removed = []
    for h, group in hashes.items():
        if len(group) > 1:
            kept = group[0]
            for dropped in group[1:]:
                removed.append({"kept": str(kept), "dropped": str(dropped), "distance": 0})

    # Cross-bucket near duplicates (brute force; buckets are small)
    bucket_list = list(hashes.keys())
    for i in range(len(bucket_list)):
        for j in range(i + 1, len(bucket_list)):
            d = int(bucket_list[i] - bucket_list[j])
            if 0 < d <= args.threshold:
                a, b = hashes[bucket_list[i]][0], hashes[bucket_list[j]][0]
                removed.append({"kept": str(a), "dropped": str(b), "distance": d})

    n_removed = len(removed)
    print(f"[DEDUPE] {n_removed} near-duplicate pairs found (threshold={args.threshold})")

    report = {
        "method": "imagehash.phash",
        "threshold": args.threshold,
        "per_dataset": per_dataset,
        "total_images": len(all_files),
        "dupes_removed": n_removed,
        "removed": removed,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DEDUPE] report -> {out}")

    if not args.dry_run:
        # Delete dropped files only among the drop set that still exist
        dropped = {Path(r["dropped"]) for r in removed}
        deleted = 0
        for p in dropped:
            if p.exists():
                p.unlink()
                deleted += 1
        print(f"[DEDUPE] deleted {deleted} dropped files")


if __name__ == "__main__":
    main()