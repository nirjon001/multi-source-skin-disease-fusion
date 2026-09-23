"""Audit an ImageFolder-style dataset: class counts, image sizes, corrupt files.

Usage:
    py -3.11 src/audit.py --data "F:/Downloads/skin_disease_images"
    py -3.11 src/audit.py --data "F:/Downloads/skin_disease_images" --check-corrupt
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def audit(root: Path, check_corrupt: bool):
    if not root.exists():
        print(f"ERROR: path does not exist: {root}")
        sys.exit(1)

    splits = [d for d in root.iterdir() if d.is_dir()]
    report = {"root": str(root), "splits": {}}

    for split in splits:
        classes = sorted([d for d in split.iterdir() if d.is_dir()])
        split_info = {"num_classes": len(classes), "classes": {}, "total": 0}
        for cls in classes:
            imgs = list(list_images(cls))
            split_info["classes"][cls.name] = len(imgs)
            split_info["total"] += len(imgs)
        report["splits"][split.name] = split_info

    # If no splits (flat ImageFolder), treat root as one split
    if not splits:
        classes = sorted([d for d in root.iterdir() if d.is_dir()])
        info = {"num_classes": len(classes), "classes": {}, "total": 0}
        for cls in classes:
            imgs = list(list_images(cls))
            info["classes"][cls.name] = len(imgs)
            info["total"] += len(imgs)
        report["splits"]["all"] = info

    # Image size sample
    sample = list(list_images(root))[:200]
    sizes = Counter()
    modes = Counter()
    corrupt = []
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception as e:
            corrupt.append({"path": str(p), "error": str(e)})
    report["sample_sizes"] = {f"{w}x{h}": c for (w, h), c in sizes.most_common(10)}
    report["sample_modes"] = dict(modes)

    if check_corrupt:
        print("Checking ALL files for corruption (may take a while)...")
        all_corrupt = []
        for p in list_images(root):
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                all_corrupt.append({"path": str(p), "error": str(e)})
        report["corrupt_files"] = all_corrupt
        report["num_corrupt"] = len(all_corrupt)
    else:
        report["corrupt_in_sample"] = corrupt

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root")
    ap.add_argument("--check-corrupt", action="store_true")
    ap.add_argument("--out", default=None, help="Save JSON to path")
    args = ap.parse_args()

    root = Path(args.data)
    report = audit(root, args.check_corrupt)

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
