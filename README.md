# CSE475 — Skin Disease Classification (Portable Training Harness)

East West University · CSE475 Machine Learning · Ratul (nirjon001 / HF: Nirob-jon)

Paper target: *Addressing label heterogeneity and data leakage in multi-source dermatological dataset fusion for South Asian / Bangladesh skin disease classification.*

**Read `AGENT.md` before every session — it is the project memory.**

HF checkpoint repo (public): `Nirob-jon/cse475-skin-checkpoints`

---

## Current state

- Phase 1 (baseline) — **DONE**: efficientnet_b0 on the starter dataset, **test_acc 0.9564**, macro_f1 0.9485
- Phase 2 (cross-dataset generalization) — **ACTIVE**:
  - Code written & smoke-tested: `harmonize.py`, `dedupe.py`, `eval_cross.py`, `gradcam_gallery.py`, `configs/phase2_*.yaml`
  - Dedupe verified: **0 cross-dataset near-duplicates** (no leakage)
  - DermNet train/validation split ready (14,314 / 1,120 / 4,002) — validation dedupe-clean vs train
  - Pending: DermNet 23-class training → `eval_cross.py` results table
- Phase 3 (paper) — pending

Direction: **cross-dataset generalization** (train on one dataset, measure accuracy drop on others) — not fusion (the datasets share zero labels). Details: `docs/DEEPSEEK_CONVERSATION_PLAN.md`.

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
    baseline.yaml        <- Phase 1 config (DO NOT EDIT)
    profiles.yaml        <- per-platform training profiles
    phase2_dermnet.yaml  <- Phase 2 DermNet train config
    phase2_eval.yaml     <- Phase 2 eval config (test-set list)
  src\
    audit.py             <- inspect ImageFolder dataset
    train_resumable.py   <- PORTABLE trainer (auto-detect + resume + HF sync)
    harmonize.py         <- build label_map.csv (Phase 2)
    dedupe.py            <- imagehash near-dup report (Phase 2)
    eval_cross.py        <- eval ONE checkpoint on N test sets (Phase 2)
    gradcam_gallery.py   <- Grad-CAM inspection gallery (mark images)
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
python src\audit.py --data "F:/Downloads/skin_disease_images" --check-corrupt --out results\audit_starter.json
python src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images"
```

### B. On the home PC (RX580, DirectML)

See `SETUP_HOME.md`. Short version:

```powershell
cd F:\cse475_skin
py -3.11 -m venv .venv-home
.\.venv-home\Scripts\Activate.ps1
pip install torch-directml
pip install timm torchvision tqdm pyyaml numpy pillow matplotlib scikit-learn imagehash huggingface_hub
python src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images"
```

### C. On Kaggle / Colab

Same script. Profile auto-detection picks `kaggle_t4` or `colab_t4`; checkpoint dir is set to `/kaggle/working/results` or `/content/drive/MyDrive/cse475/results` respectively.

---

## Portable training (the important feature)

One command, four machines, one checkpoint, seamless resume:

```powershell
python src\train_resumable.py --config configs\baseline.yaml --data "<dataset root>" --resume auto
```

(Use your activated venv's `python` — lab `.venv` (CUDA), home `.venv-home` (DirectML) or `.venv-home-cpu` (CPU).)

The script:
1. Auto-detects platform (lab / Kaggle / Colab / home DirectML / home CPU)
2. Loads `results/<run>_last.pt` if present and resumes at the exact epoch
3. Saves model + optimizer + scheduler + RNG + history after every epoch
4. Handles Ctrl+C / power loss via a signal handler (never loses an epoch)
5. Optionally syncs to Hugging Face with `--hub hf --hf-repo USER/REPO`

Full details: `AGENT.md` Section 14.

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

Phase 2 (current) uses `configs\phase2_dermnet.yaml`, run via `.\.venv-home\Scripts\python.exe` on the home RX580.

---

## Cross-machine accuracy — short answer

No meaningful drift. Save model + optimizer + scheduler + RNG; keep batch size identical across machines; use AMP on CUDA only. Expected difference is only epoch time. Full explanation in `AGENT.md` Section 14.

---

## Environment traps (do not forget)

- Global `python` (3.14) has no torch wheels. On home use `.\.venv-home\Scripts\python.exe` (DirectML) or `.\.venv-home-cpu\Scripts\python.exe` (CPU).
- Lab venv (CUDA torch), home venv (DirectML torch) are **separate**. Do not mix.
- Do not commit `data/`, `results/`, or `*.pt` (already in `.gitignore`).
- YAML files carry a UTF-8 BOM — the trainer reads them as `utf-8-sig`. Don't strip it "by cleanup".

---

## Phase 2 command (the active goal)

**Primary path (2026-09-24): Kaggle T4 notebook** — `notebooks/kaggle_phase2.ipynb` trains the DermNet 23-class model
(batch 32, AMP on, ~1.5-4 min/epoch → full 15 epochs in ~30-90 min). The prepared split (train 14,314 / validation 1,120 /
test 4,002) is on HF as `Nirob-jon/cse475-dermnet-split` (`DermNetPrepared.zip`). Checkpoints sync to HF every epoch.

Local home fallback (RX580, DirectML, ~19 min/epoch — no AMP on DirectML):

```powershell
.\.venv-home\Scripts\python.exe src\train_resumable.py --config configs\phase2_dermnet.yaml --profile home_rx580_dml --hub hf --resume auto
```

Then evaluate the trained model on all 5 test sets:

```powershell
.\.venv-home\Scripts\python.exe src\eval_cross.py --checkpoint results\phase2_dermnet_best.pt --config configs\phase2_eval.yaml
```

Success = `results/phase2_dermnet_best.pt` exists, then a per-dataset eval JSON under `results/`.
Success for Phase 1: `results/phase1_baseline.json` contains a numeric `test_acc` (it does — 0.9564).
Kaggle run: use `notebooks/kaggle_phase2.ipynb` (HF login via `HF_TOKEN` secret, auto-internet); the run is resumable from the same HF repo.