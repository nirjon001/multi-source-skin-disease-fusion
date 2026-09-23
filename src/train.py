"""Train one image classification model and save results as JSON.

Usage:
    py -3.11 src/train.py --config configs/baseline.yaml
    py -3.11 src/train.py --data "F:/Downloads/skin_disease_images" --model efficientnet_b0 --epochs 15
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import timm

try:
    import yaml
except ImportError:
    yaml = None


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(data_root: Path, img_size: int, batch_size: int, num_workers: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "validation", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes, "Class mismatch across splits"

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return train_ds, val_ds, test_ds, train_ld, val_ld, test_ld


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="F:/Downloads/skin_disease_images")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", default="phase1_baseline")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--logs-dir", default="logs")
    args = ap.parse_args()

    if args.config and yaml:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        for k, v in cfg.items():
            if hasattr(args, k.replace("-", "_")):
                setattr(args, k.replace("-", "_"), v)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(args.data)
    train_ds, val_ds, test_ds, train_ld, val_ld, test_ld = build_loaders(
        data_root, args.img_size, args.batch_size, args.num_workers
    )
    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = timm.create_model(args.model, pretrained=True, num_classes=len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_path = results_dir / f"{args.run_name}_best.pt"

    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_ld, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # Load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, te_preds, te_labels = evaluate(model, test_ld, criterion, device)
    macro_f1 = f1_score(te_labels, te_preds, average="macro")
    per_class = classification_report(te_labels, te_preds,
                                     target_names=train_ds.classes,
                                     output_dict=True, zero_division=0)
    cm = confusion_matrix(te_labels, te_preds).tolist()

    print(f"\n=== TEST ===")
    print(f"Acc: {te_acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(te_labels, te_preds, target_names=train_ds.classes, zero_division=0))

    out = {
        "run_id": args.run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": data_root.name,
        "dataset_path": str(data_root),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "classes": train_ds.classes,
        "model": args.model,
        "pretrained": "imagenet",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "test_acc": te_acc,
        "test_macro_f1": macro_f1,
        "per_class_f1": {k: v["f1-score"] for k, v in per_class.items() if k in train_ds.classes},
        "confusion_matrix": cm,
        "history": history,
    }
    out_path = results_dir / f"{args.run_name}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
