# CSE475 — Skin Disease Classification (Portable Training Harness)

East West University · CSE475 Machine Learning · Ratul (nirjon001 / HF: Nirob-jon)

Paper target: *Addressing label heterogeneity and data leakage in multi-source dermatological dataset fusion for South Asian / Bangladesh skin disease classification.*

**Read `AGENT.md` before every session — it is the project memory.**

HF checkpoint repo (public): `Nirob-jon/cse475-skin-checkpoints`

---

## Current state

- Phase 1 (baseline) — **not started**
- Phase 2 (add one dataset) — pending
- Phase 3 (full fusion + bias) — pending
- Phase 4 (paper) — pending

Golden rule: **do not open a second dataset until `results/phase1_baseline.json` exists with a `test_acc` value.**

---

## Folder layout

```
F:\cse475_skin\
  AGENT.md               <- project memory (READ FIRST)
  SETUP_HOME.md          <- RX580 DirectML / CPU setup (home PC only)
  README.md              <- this file
  requirements.txt       <- pip list
  .gitignore
  configs\
    baseline.yaml        <- Phase 1 config
    profiles.yaml        <- per-platform training profiles
  src\
    audit.py             <- inspect ImageFolder dataset
    train_resumable.py   <- PORTABLE trainer (auto-detect + resume + HF sync)
  data\                  <- working copies (gitignored)
  results\               <- checkpoints + results JSON (gitignored)
  logs\
  notebooks\
```

---

## Quick start

### A. On the university lab A4000 (recommended)

```powershell
cd F:\cse475_skin
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# sanity check GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# audit then train
py -3.11 src\audit.py --data "F:/Downloads/skin_disease_images" --check-corrupt --out results\audit_starter.json
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images"
```

### B. On the home PC (RX580, DirectML)

See `SETUP_HOME.md`. Short version:

```powershell
cd F:\cse475_skin
py -3.11 -m venv .venv-home
.\.venv-home\Scripts\Activate.ps1
pip install torch-directml
pip install timm torchvision tqdm pyyaml numpy pillow matplotlib scikit-learn imagehash huggingface_hub
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images"
```

### C. On Kaggle / Colab

Same script. Profile auto-detection picks `kaggle_t4` or `colab_t4`; checkpoint dir is set to `/kaggle/working/results` or `/content/drive/MyDrive/cse475/results` respectively.

---

## Portable training (the important feature)

One command, four machines, one checkpoint, seamless resume:

```powershell
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "<dataset root>" --resume auto
```

The script:
1. Auto-detects platform (lab / Kaggle / Colab / home DirectML / home CPU)
2. Loads `results/<run>_last.pt` if present and resumes at the exact epoch
3. Saves model + optimizer + scheduler + RNG + history after every epoch
4. Handles Ctrl+C / power loss via a signal handler (never loses an epoch)
5. Optionally syncs to Hugging Face with `--hub hf --hf-repo USER/REPO`

Full details: `AGENT.md` Section 11.

---

## Key CLI flags for `train_resumable.py`

| Flag | Purpose |
|---|---|
| `--config PATH` | Training config (required) — e.g. `configs/baseline.yaml` |
| `--profiles PATH` | Profile file (default `configs/profiles.yaml`) |
| `--profile NAME` | Force a profile: `auto` / `lab_a4000` / `kaggle_t4` / `colab_t4` / `home_rx580_dml` / `home_rx580_cpu` |
| `--device` | `auto` / `cuda` / `dml` / `cpu` |
| `--data PATH` | Override dataset root |
| `--resume PATH` | Checkpoint path, or `auto` to use `results/<run>_last.pt` |
| `--hub hf` | Enable Hugging Face download/upload |
| `--hf-repo USER/REPO` | HF model repo for checkpoints |
| `--epochs N` | Override epoch count |

---

## Cross-machine accuracy — short answer

No meaningful drift. Save model + optimizer + scheduler + RNG; keep batch size identical across machines; use AMP on CUDA only. Expected difference is only epoch time. Full explanation in `AGENT.md` Section 11.

---

## Environment traps (do not forget)

- `python` = 3.14.5 → **no PyTorch wheels**. Always use `py -3.11`.
- Lab venv (CUDA torch) and home venv (DirectML torch) are **separate**. Do not mix.
- Do not commit `data/`, `results/`, or `*.pt` (already in `.gitignore`).

---

## Phase 1 command (the immediate goal)

```powershell
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images"
```

Success = `results/phase1_baseline.json` contains a numeric `test_acc`.

Until that file exists, do not touch SkinDiseaseBD, SCIN, Fitzpatrick17k, or 32 Curated Categories.