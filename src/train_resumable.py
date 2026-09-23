"""
CSE475 Skin Disease Classification - Portable, Resumable Trainer

Runs unchanged on:
  - Lab A4000 (CUDA, AMP)
  - Kaggle T4 (CUDA, AMP, /kaggle/working)
  - Colab T4 (CUDA, AMP, /content/drive/MyDrive/cse475/results)
  - Home RX580 (DirectML with auto CPU fallback)
  - Any CPU

Key features:
  * Auto-detects platform and selects the right profile
  * Checkpoints model + optimizer + scheduler + epoch + step + RNG states
  * Resumes exactly where the previous machine stopped
  * Saves _last (for resume) and _best (for final model)
  * Handles Ctrl+C / power-loss via signal handler
  * Optional Hugging Face hub sync (--hub hf)

Usage:
  py -3.11 src/train_resumable.py --config configs/baseline.yaml
  py -3.11 src/train_resumable.py --config configs/baseline.yaml --resume auto
  py -3.11 src/train_resumable.py --config configs/baseline.yaml --profile kaggle_t4 --hub hf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import yaml
except ImportError:
    yaml = None

try:
    import timm
except ImportError:
    print("[FATAL] timm not installed. Run: py -3.11 -m pip install timm")
    raise

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def detect_environment() -> dict:
    env = {
        "is_kaggle": os.path.exists("/kaggle/working"),
        "is_colab": "google.colab" in sys.modules or os.path.exists("/content/drive"),
        "has_cuda": torch.cuda.is_available(),
        "has_dml": False,
        "gpu_name": None,
        "hostname": platform.node().lower(),
    }
    if env["has_cuda"]:
        env["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        import torch_directml  # noqa: F401
        env["has_dml"] = torch_directml.is_available()
    except Exception:
        env["has_dml"] = False
    env["is_lab"] = (
        os.environ.get("CSE475_MACHINE", "").lower() == "lab"
        or "a4000" in (env["gpu_name"] or "").lower()
    )
    env["is_home"] = (
        not env["is_kaggle"] and not env["is_colab"]
        and not env["is_lab"] and not env["has_cuda"]
    )
    return env


def pick_profile(env: dict, profiles: dict, override: str | None) -> tuple[str, dict]:
    if override and override != "auto":
        if override not in profiles:
            raise SystemExit(f"[FATAL] Unknown profile '{override}'. "
                             f"Available: {list(profiles)}")
        return override, dict(profiles[override])
    if env["is_kaggle"]:
        name = "kaggle_t4"
    elif env["is_colab"]:
        name = "colab_t4"
    elif env["is_lab"]:
        name = "lab_a4000"
    elif env["has_dml"]:
        name = "home_rx580_dml"
    else:
        name = "home_rx580_cpu"
    if name not in profiles:
        name = "home_rx580_cpu" if not env["has_cuda"] else "lab_a4000"
    return name, dict(profiles[name])


def resolve_device(spec: str, env: dict):
    if spec == "cuda":
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "dml":
        import torch_directml
        return torch_directml.device()
    if env["has_cuda"]:
        return torch.device("cuda")
    if env["has_dml"]:
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_loaders(cfg, prof, data_root: Path):
    size = prof.get("image_size", 224)
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def find_split(*cands):
        for c in cands:
            p = data_root / c
            if p.exists():
                return p
        return None

    train_dir = find_split("train", "Train")
    val_dir = find_split("validation", "val", "Val", "valid")
    test_dir = find_split("test", "Test")
    if train_dir is None:
        raise SystemExit(f"[FATAL] No train/ folder under {data_root}")

    ds_train = datasets.ImageFolder(str(train_dir), train_tf)
    ds_val = datasets.ImageFolder(str(val_dir), eval_tf) if val_dir else None
    ds_test = datasets.ImageFolder(str(test_dir), eval_tf) if test_dir else None

    nw = prof.get("num_workers", 4)
    bs = prof["batch_size"]
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available())
    if nw > 0:
        kw["persistent_workers"] = True

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=False, **kw)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, **kw) if ds_val else None
    dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False, **kw) if ds_test else None
    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------
def save_ckpt(path: Path, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    state["torch_version"] = torch.__version__
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_ckpt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
              device=None, dml_fallback_cpu=None, desc=""):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    use_amp = scaler is not None and str(device).startswith("cuda")

    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        try:
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out, y)
            else:
                out = model(x)
                loss = criterion(out, y)
        except RuntimeError as e:
            msg = str(e).lower()
            if "not implemented" in msg or "could not run" in msg or "dml" in msg:
                if dml_fallback_cpu is not None and not dml_fallback_cpu[0]:
                    print("\n[WARN] DirectML op unsupported -> switching to CPU for rest of run")
                    dml_fallback_cpu[0] = True
                x = x.cpu(); y = y.cpu()
                model_cpu = model.to("cpu")
                out = model_cpu(x)
                loss = criterion(out, y)
                if training:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                continue
            raise

        if training:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            try:
                out = model(x)
            except RuntimeError:
                x = x.cpu(); y = y.cpu()
                out = model.to("cpu")(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            preds_all.append(out.argmax(1).cpu())
            labels_all.append(y.cpu())
            correct += (out.argmax(1).cpu() == y.cpu()).sum().item()
            total += y.size(0)
    import torch as _t
    preds = _t.cat(preds_all).numpy()
    labels = _t.cat(labels_all).numpy()
    return loss_sum / max(total, 1), correct / max(total, 1), preds, labels


# ---------------------------------------------------------------------------
# Optional HF hub sync
# ---------------------------------------------------------------------------
def maybe_download_from_hf(repo_id: str, filename: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARN] huggingface_hub not installed; skipping HF download")
        return False
    try:
        p = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        dest.parent.mkdir(parents=True, exist_ok=True)
        Path(p).replace(dest)
        print(f"[HF] Downloaded {filename} -> {dest}")
        return True
    except Exception as e:
        print(f"[HF] download skipped: {e}")
        return False


def maybe_upload_to_hf(repo_id: str, local: Path, remote_name: str):
    if not repo_id:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[HF] Uploaded {local.name} -> {repo_id}/{remote_name}")
    except Exception as e:
        print(f"[HF] upload failed (keeping local copy): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profiles", default="configs/profiles.yaml")
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "dml", "cpu"])
    ap.add_argument("--data", default=None,
                    help="Override dataset root from config")
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint, or 'auto' to use results/<run>_last.pt")
    ap.add_argument("--hub", default="local",
                    choices=["local", "hf", "gdrive"])
    ap.add_argument("--hf-repo", default=os.environ.get("HF_CKPT_REPO", ""))
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    env = detect_environment()
    print("[ENV] platform=", env)

    project = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project / args.config
    cfg = yaml.safe_load(cfg_path.read_text()) if yaml else {}

    prof_path = Path(args.profiles)
    if not prof_path.is_absolute():
        prof_path = project / args.profiles
    profiles = yaml.safe_load(prof_path.read_text())["profiles"] if yaml else {}

    prof_name, prof = pick_profile(env, profiles, args.profile)
    print(f"[PROFILE] {prof_name} -> {prof}")

    device = resolve_device(args.device, env)
    print(f"[DEVICE] {device}")

    # Paths
    run_id = cfg.get("run_id", "phase1_baseline")
    results_dir = Path(cfg.get("results_dir", project / "results"))
    if prof.get("checkpoint_dir"):
        results_dir = Path(prof["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data or cfg.get("data_root") or cfg.get("data"))
    print(f"[DATA] {data_root}")

    seed_everything(cfg.get("seed", 42))

    # Data
    ds_train, ds_val, ds_test, dl_train, dl_val, dl_test = build_loaders(cfg, prof, data_root)
    num_classes = len(ds_train.classes)
    print(f"[DATA] classes={ds_train.classes}  n_train={len(ds_train)}")

    # Model
    model_name = cfg.get("model", "efficientnet_b0")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    # Optimizer + scheduler
    epochs = args.epochs or prof.get("epochs") or cfg.get("epochs", 15)
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(dl_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch)

    use_amp = bool(prof.get("amp", False)) and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"[AMP] {use_amp}")

    criterion = nn.CrossEntropyLoss()

    # Resume
    start_epoch = 0
    best_f1 = 0.0
    history = []
    ckpt_last = results_dir / f"{run_id}_last.pt"
    ckpt_best = results_dir / f"{run_id}_best.pt"
    state_json = results_dir / f"{run_id}_state.json"

    resume_arg = args.resume
    if resume_arg == "auto":
        if ckpt_last.exists():
            resume_arg = str(ckpt_last)
        elif args.hub == "hf" and args.hf_repo:
            if maybe_download_from_hf(args.hf_repo, ckpt_last.name, ckpt_last):
                resume_arg = str(ckpt_last)
    elif resume_arg is None and args.hub == "hf" and args.hf_repo:
        if maybe_download_from_hf(args.hf_repo, ckpt_last.name, ckpt_last):
            resume_arg = str(ckpt_last)

    if resume_arg and Path(resume_arg).exists():
        ck = load_ckpt(Path(resume_arg))
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck.get("epoch", -1) + 1
        best_f1 = ck.get("best_f1", 0.0)
        history = ck.get("history", [])
        hist_best = max((r.get("val_acc", 0.0) for r in history), default=0.0)
        if hist_best > best_f1:
            print(f"[RESUME] repairing best_f1 {best_f1:.4f} -> {hist_best:.4f} (from history)")
            best_f1 = hist_best
        if "rng_torch" in ck:
            torch.set_rng_state(ck["rng_torch"])
            try:
                random.setstate(ck["rng_py"])
                np.random.set_state(ck["rng_np"])
            except Exception:
                pass
            if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all(ck["rng_cuda"])
                except Exception:
                    pass
        print(f"[RESUME] from {resume_arg} -> starting epoch {start_epoch}, best_f1={best_f1:.4f}")
    else:
        print("[RESUME] no checkpoint, training from scratch")

    # Signal handler for clean checkpoint on Ctrl+C / shutdown
    stop_flag = {"stop": False}
    def _handler(signum, frame):
        print(f"\n[SIGNAL] received {signum}, saving checkpoint and exiting...")
        stop_flag["stop"] = True
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass

    dml_fallback_cpu = [False]

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, dl_train, criterion, optimizer, scaler, device,
            dml_fallback_cpu, desc=f"train e{epoch+1}/{epochs}")
        scheduler.step()
        if dl_val:
            va_loss, va_acc, _, _ = evaluate(model, dl_val, criterion, device, desc="val")
        else:
            va_loss, va_acc = 0.0, 0.0
        dt = time.time() - t0

        row = {
            "epoch": epoch + 1,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(dt, 1),
            "profile": prof_name,
        }
        history.append(row)
        print(f"[EPOCH {epoch+1}] {row}")

        if va_acc > best_f1:
            best_f1 = va_acc
            save_ckpt(
                ckpt_best,
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                best_f1=best_f1,
                history=history,
                config=cfg,
                profile=prof_name,
            )
            print(f"[BEST] new best val_acc={best_f1:.4f}")

        # Save _last every epoch (after best update, so best_f1 is current)
        save_ckpt(
            ckpt_last,
            epoch=epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            best_f1=best_f1,
            history=history,
            config=cfg,
            profile=prof_name,
            rng_torch=torch.get_rng_state(),
            rng_py=random.getstate(),
            rng_np=np.random.get_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

        state_json.write_text(json.dumps({
            "run_id": run_id,
            "profile": prof_name,
            "history": history,
            "best_f1": best_f1,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))

        if args.hub == "hf" and args.hf_repo:
            maybe_upload_to_hf(args.hf_repo, ckpt_last, ckpt_last.name)
            if ckpt_best.exists():
                maybe_upload_to_hf(args.hf_repo, ckpt_best, ckpt_best.name)

        if stop_flag["stop"]:
            print("[SIGNAL] stopping after epoch")
            break

    # Final test
    test_acc = None
    if dl_test:
        _, test_acc, preds, labels = evaluate(model, dl_test, criterion, device, desc="test")
        try:
            from sklearn.metrics import classification_report, confusion_matrix, f1_score
            rep = classification_report(labels, preds, target_names=ds_train.classes, output_dict=True)
            cm = confusion_matrix(labels, preds).tolist()
            macro_f1 = f1_score(labels, preds, average="macro")
        except Exception as e:
            print(f"[WARN] sklearn report failed: {e}")
            rep, cm, macro_f1 = {}, [], 0.0

        results_json = results_dir / f"{run_id}.json"
        results_json.write_text(json.dumps({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": cfg.get("dataset_name", data_root.name),
            "dataset_path": str(data_root),
            "classes": ds_train.classes,
            "model": model_name,
            "profile": prof_name,
            "device": str(device),
            "epochs": epochs,
            "test_acc": round(float(test_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "classification_report": rep,
            "confusion_matrix": cm,
            "history": history,
        }, indent=2))
        print(f"[DONE] test_acc={test_acc:.4f}  -> {results_json}")


if __name__ == "__main__":
    main()