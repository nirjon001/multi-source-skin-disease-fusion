"""
CSE475 Skin Disease Classification - Cross-dataset evaluator (Phase 2)

Loads ONE trained checkpoint, evaluates it on N test sets from
configs/phase2_eval.yaml, and dumps one JSON to results/.

Test set modes:
  - "folder_per_class"  : ImageFolder with per-set source class list.
                          Model's raw DermNet argmax is mapped to a unified
                          class via results/label_map.csv, compared to the
                          test source's unified class.
  - "subfolder_binary"  : flat subfolders (e.g. DDI Black/White) with NO per-
                          image disease labels -> BIAS PROBE only: report
                          predicted-class distribution + confidence per group.

Usage:
  py -3.11 src/eval_cross.py --checkpoint results/phase2_dermnet_best.pt --config configs/phase2_eval.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def eval_tf(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def load_label_map(path: Path) -> dict[str, str]:
    """source_dataset+source_class -> unified_class"""
    m = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[(row["source_dataset"], row["source_class"])] = row["unified_class"]
    return m


def build_model(model_name: str, num_classes: int, ckpt: dict, device: torch.device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def dermnet_classes(train_root: Path) -> list[str]:
    """Reconstruct ImageFolder class order (sorted folder names)."""
    return sorted(d.name for d in train_root.iterdir() if d.is_dir())


@torch.inference_mode()
def eval_folder_per_class(cfg, model, device, label_map, derm_classes, unified_of):
    """Raw DermNet argmax, then map predicted class -> unified, compare to true unified."""
    inline = cfg.get("inline_map", {})
    folders = cfg.get("folders")
    if not folders or folders == "auto":
        folders = [d.name for d in Path(cfg["root"]).iterdir() if d.is_dir()]
    source_dataset = cfg.get("source_dataset", cfg["name"])

    src_to_unified = {}
    for src in folders:
        uni = inline.get(src)
        if uni is None:
            uni = label_map.get((source_dataset, src))
        if uni is not None:
            src_to_unified[src] = uni

    ds = datasets.ImageFolder(str(Path(cfg["root"])), eval_tf(cfg.get("image_size", 224)))
    exist = {ds.classes.index(c) for c in src_to_unified if c in ds.classes}
    missing = set(src_to_unified) - set(ds.classes)
    if missing:
        print(f"[WARN] {cfg['name']}: folders not found in root -> skipped: {sorted(missing)}")
    src_to_unified = {c: u for c, u in src_to_unified.items() if c in ds.classes}

    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 32), shuffle=False, num_workers=0)

    all_logits, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        all_logits.append(model(x).cpu())
        all_true.append(y)
    logits = torch.cat(all_logits)
    true_src_idx = torch.cat(all_true)

    # only scores within folders we actually evaluate
    allowed_idx = exist
    mask = [int(i.item() in allowed_idx) for i in true_src_idx]
    sel_idx = [i for i, m in enumerate(mask) if m]
    if not sel_idx:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n": 0, "classes": [], "per_class_acc": {},
                "confusion_matrix": [], "folders_used": [], "error": "no selected images"}

    pred_raw = logits[sel_idx].argmax(dim=1)              # 23-class argmax
    n = len(sel_idx)
    true_uni = [src_to_unified[ds.classes[true_src_idx[i].item()]] for i in sel_idx]
    pred_uni = [unified_of[int(p.item())] for p in pred_raw]  # map via label_map

    unis = sorted(set(true_uni) | set(pred_uni))
    correct = sum(1 for t, p in zip(true_uni, pred_uni) if t == p)
    acc = correct / n if n else 0.0

    per_class = {}
    for u in unis:
        in_class = [p for t, p in zip(true_uni, pred_uni) if t == u]
        if in_class:
            per_class[u] = round(sum(1 for p in in_class if p == u) / len(in_class), 4)

    macro_f1 = _macro_f1(true_uni, pred_uni, unis)
    cm = _confusion(true_uni, pred_uni, unis)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "n": n,
        "classes": unis,
        "per_class_acc": per_class,
        "confusion_matrix": cm,
        "folders_used": sorted(src_to_unified),
    }


def _macro_f1(true, pred, classes):
    eps = 1e-9
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(true, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, pred) if t == c and p != c)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1s.append(2 * prec * rec / (prec + rec + eps))
    return float(np.mean(f1s))


def _confusion(true, pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]
    for t, p in zip(true, pred):
        cm[idx[t]][idx[p]] += 1
    return cm


def eval_ddi_bias(cfg, model, device, derm_classes):
    """BIAS PROBE: no per-image labels -> predicted-class distribution per group."""
    result = {}
    for group in cfg.get("groups", ["Black", "White"]):
        gdir = Path(cfg["root"]) / group
        imgs = sorted(p for p in gdir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        dist = {c: 0 for c in derm_classes}
        confs = []
        tf = eval_tf(cfg.get("image_size", 224))
        for p in imgs:
            with Image.open(p) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
                logits = model(x)
                pv = F.softmax(logits, dim=1)
                idx = int(pv.argmax(1).item())
                dist[derm_classes[idx]] += 1
                confs.append(float(pv.max(1).values.item()))
        n = len(imgs)
        dist_frac = {c: round(v / n, 4) for c, v in dist.items() if v > 0}
        result[group] = {
            "n": n,
            "mean_confidence": round(float(np.mean(confs)), 4),
            "top5": sorted(dist_frac.items(), key=lambda kv: -kv[1])[:5],
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset evaluation of one checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "dml", "cpu"])
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = cfg.get("model", "efficientnet_b0")
    num_classes = cfg.get("num_classes", 23)

    device = _resolve_device(args.device)
    model = build_model(model_name, num_classes, ckpt, device)

    derm_root = Path(cfg["dermnet_train_root"])
    derm_classes = dermnet_classes(derm_root)
    if len(derm_classes) != num_classes:
        print(f"[WARN] dermnet folder count {len(derm_classes)} != num_classes {num_classes}")

    label_map = load_label_map(Path(cfg["label_map"])) if cfg.get("label_map") else {}

    # unified_of: model output index -> unified class (DermNet rows of the map)
    unified_of = {}
    for i, c in enumerate(derm_classes):
        unified_of[i] = label_map.get(("dermnet", c), c)

    results = {"checkpoint": args.checkpoint, "model": model_name, "num_classes": num_classes,
               "device": str(device), "test_sets": {}}

    for tset in cfg["test_sets"]:
        mode = tset.get("mode", "folder_per_class")
        name = tset["name"]
        print(f"[EVAL] {name} ({mode}) ...")
        if mode == "folder_per_class":
            res = eval_folder_per_class(tset, model, device, label_map, derm_classes, unified_of)
        elif mode == "subfolder_binary":
            res = eval_ddi_bias(tset, model, device, derm_classes)
        else:
            raise SystemExit(f"[FATAL] unknown mode '{mode}' for {name}")
        results["test_sets"][name] = res
        print(f"[EVAL]   {name} -> {json.dumps({k: v for k, v in res.items() if k != 'confusion_matrix'}, default=str)}")

    out = Path(cfg.get("out", f"results/{cfg.get('run_id', 'phase2')}_eval.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[EVAL] -> {out}")


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


if __name__ == "__main__":
    main()