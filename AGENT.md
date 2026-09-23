# AGENT.md — CSE475 Skin Disease Classification Project

**This file is the project memory. Read it before every session. Update it when facts change.**

**Owner:** Ratul (nirjon001)
**Course:** CSE475 Machine Learning, East West University
**Deliverable:** Conference-equivalent paper (target: ICCIT / ICIEV / EWU conference)
**Last updated:** Phase 1 prep complete (bugs fixed, audit OK, CPU smoke test passed); baseline training pending on lab A4000

---

## 1. Paper Angle (the contribution)

> *Addressing label heterogeneity and data leakage in multi-source dermatological dataset fusion for South Asian / Bangladesh skin disease classification.*

This is a **methodology paper**, not an architecture paper. The novelty is:
1. Fusing multiple skin datasets with conflicting label spaces
2. Documenting and fixing label heterogeneity
3. Detecting and removing cross-dataset near-duplicates (leakage)
4. Measuring the fusion gain on South Asian / Fitzpatrick IV-VI skin

---

## 2. GOLDEN RULE

**Do NOT open a second dataset until `results/phase1_baseline.json` exists with a `test_acc` value.**

One dataset. One model. One number. Then fusion.

---

## 3. Hardware (confirmed)

### Home PC — do NOT train here
| Spec | Value |
|---|---|
| CPU | AMD Ryzen 5 5600 |
| GPU | AMD RX580, 8GB VRAM, **no CUDA** (ROCm/DirectML only) |
| RAM | 16GB |
| OS | Windows 11 (build 10.0.26200) |
| Hostname | NirjonPC1 |

Use the home PC for: writing code, reading papers, `audit.py`, unzipping, label mapping, writing the paper.
Never use the RX580 for PyTorch CUDA training — it will not work and will waste days.

### University Lab PC — TRAIN HERE
| Spec | Value |
|---|---|
| CPU | Intel i7 12th / 13th / 14th gen |
| GPU | **NVIDIA A4000, 16GB VRAM, CUDA** |
| RAM | 16GB |

All `train.py` runs happen on the lab A4000.

### Fallback if lab access is unavailable
Kaggle Notebooks (free T4 / P100, ~30 hrs/week) or Google Colab free tier. Upload the dataset as a Kaggle Dataset and run the same `train.py` logic in a notebook. Code stays identical.

---

## 4. Python Environment (critical trap)

| Interpreter | Version | Status |
|---|---|---|
| `python` (default) | **3.14.5** | **DO NOT USE** — no PyTorch wheels exist |
| `py -3.11` | **3.11.9** | **USE THIS ONE** |

Already installed on `py -3.11`: numpy, pandas, scikit-learn, pillow, matplotlib.

**Missing on `py -3.11` (must install once):** torch, torchvision, timm, imagehash, tqdm, pyyaml.

### Setup (run once on the lab PC)
```powershell
cd F:\cse475_skin
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Verify GPU
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 5. Local Machine Layout

| Drive | Used | Free | Role |
|---|---|---|---|
| C: | 137GB | 38.7GB | System, user home |
| D: | 33.6GB | 17GB | Misc |
| E: | 95.3GB | 153.4GB | Misc |
| **F:** | 521.6GB | 151.7GB | **Datasets + project live here** |
| G: | 129GB | 129.2GB | Misc / lan share |

---

## 6. Dataset Inventory (verified on disk)

All under `F:\Downloads\` unless noted.

| # | Dataset | Path | Classes | Images | State |
|---|---|---|---|---|---|
| 1 | **Starter: acne/rosacea/normal** (= HuggingFace `Neperl/skin-disease-acne-rosacea-normal`) | `F:\Downloads\skin_disease_images\` | acne, normal, rosacea | 2,439 | Extracted, pre-split |
| 2 | SkinDiseaseBD | `F:\Downloads\SkinDiseaseBD A Dataset of Common Skin Disease Ima.zip` | 5 BD diseases | ~1,612 | **Zip, 978.7 MB, not extracted** |
| 3 | SCIN (Google Research) | `F:\Downloads\SCIN (Google Research)\` | multi-label (CSV) | ~10,000+ PNG | Extracted, flat PNG + CSVs |
| 4 | Fitzpatrick17k black-images | `F:\Downloads\archive\fitzpatrick-black-images\` | 16 derm classes | ~2,000 | Extracted + `Db.csv` |
| 5 | 32 Curated Categories | `F:\Downloads\32 Curated Categories of Skin Disease Images.zip` | 32 | ? | **Zip, 1039.9 MB, not extracted** |
| 6 | HAM10000 | — | 7 | 10,015 | **NOT FOUND on disk** |
| 7 | DermNet | — | 23 | ~23,000 | **NOT FOUND on disk** |

### Starter dataset class counts (verified)
| Split | acne | normal | rosacea | Total |
|---|---|---|---|---|
| train | 460 | 476 | 770 | 1,706 |
| validation | 99 | 102 | 165 | 366 |
| test | 99 | 103 | 165 | 367 |
| **Total** | | | | **2,439** |

All images `.jpg`. Already split 70/15/15. No splitting needed.

### Other SCIN files found
- `F:\Downloads\dataset_scin_cases.csv`
- `F:\Downloads\dataset_scin_labels.csv`
- `F:\Downloads\dataset_scin_label_questions.csv`
- `F:\Downloads\scin_app_questions.csv`
- `F:\Downloads\archive\Db.csv` (Fitzpatrick labels)

---

## 7. Project Folder

```
F:\cse475_skin\
  AGENT.md              <- this file (project memory)
  README.md             <- quick start
  requirements.txt      <- pip list
  .gitignore            <- blocks data/ and *.pt
  configs\
    baseline.yaml       <- Phase 1 config
  src\
    audit.py            <- inspect ImageFolder dataset
    train.py            <- train + save results JSON
    harmonize.py        <- (Phase 2) label mapping
    dedupe.py           <- (Phase 2) imagehash dedupe
  data\                 <- working copies (gitignored)
  results\              <- one JSON per run
  logs\
  notebooks\
```

---

## 8. Phase Plan (STRICT ORDER)

### Phase 1 — Baseline (Week 1-2) — NOT STARTED
**Goal:** one training run, one accuracy number saved.

- [ ] Lab PC: create venv, install torch/timm (Section 4)
- [ ] Run `py -3.11 src\audit.py --data "F:/Downloads/skin_disease_images" --check-corrupt --out results\audit_starter.json`
- [ ] Run `py -3.11 src\train.py --config configs\baseline.yaml`
- [ ] Confirm `results\phase1_baseline.json` has `test_acc`

Model: `efficientnet_b0` (timm, ImageNet pretrained)
Epochs 15 | Batch 32 | Img 224 | LR 1e-3 | AdamW | Cosine | Seed 42
Expected runtime on A4000: ~5-8 min.

### Phase 2 — Add ONE dataset (Week 3)
- [ ] Extract `SkinDiseaseBD ... .zip`
- [ ] Write `src\harmonize.py` — label mapping CSV (`source_class | unified_class`)
- [ ] Write `src\dedupe.py` — `imagehash.phash`, Hamming <= 5
- [ ] Train on fused set -> `results\phase2_fused.json`
- [ ] Compare against Phase 1

### Phase 3 — Full fusion + bias (Week 4-5)
- [ ] Add Fitzpatrick17k-black + SCIN
- [ ] Fitzpatrick-stratified accuracy (bias measurement)
- [ ] Grad-CAM visualizations

### Phase 4 — Write paper (Week 6)
- [ ] IEEE template, 6-8 pages
- [ ] Submit to ICCIT / ICIEV / EWU conference

---

## 9. DO / DON'T

**DO**
- Always `py -3.11`, never `python`
- Always seed 42
- Save every run as JSON in `results/`
- Log dataset + model + hyperparams in JSON
- Document label mapping before any fusion
- Run dedupe before fusion (leakage kills papers)
- Train on lab A4000 only

**DON'T**
- Don't train on all datasets at once (Phase 1 trap)
- Don't use `python` (3.14, no torch wheels)
- Don't train on the home RX580
- Don't skip dedupe in Phase 2
- Don't report accuracy without a confusion matrix
- Don't commit datasets or `.pt` checkpoints to git
- Don't extract zips until Phase 2

---

## 10. Experiment Tracking Format

Every `results/*.json` must contain:
```json
{
  "run_id": "phase1_baseline",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "dataset": "skin_disease_acne_rosacea_normal",
  "dataset_path": "F:/Downloads/skin_disease_images",
  "num_train": 1706, "num_val": 366, "num_test": 367,
  "classes": ["acne", "normal", "rosacea"],
  "model": "efficientnet_b0", "pretrained": "imagenet",
  "epochs": 15, "batch_size": 32, "img_size": 224,
  "lr": 0.001, "weight_decay": 0.0001,
  "optimizer": "adamw", "scheduler": "cosine", "seed": 42,
  "best_val_acc": 0.0, "test_acc": 0.0, "test_macro_f1": 0.0,
  "per_class_f1": {}, "confusion_matrix": [[..]], "history": [..]
}
```

---

## 11. Reproducibility Checklist

- [ ] `seed_everything(42)` at top of `train.py`
- [ ] `cudnn.deterministic = True`
- [ ] Versions pinned in `requirements.txt`
- [ ] Git commit hash in results JSON
- [ ] No leakage (dedupe ran before fusion)

---

## 12. Paper-Writing Rules

- Cite every dataset source (SkinDiseaseBD, SCIN, Fitzpatrick17k, HAM10000)
- Include harmonization table as a paper table
- Report dedupe stats: "removed X near-duplicates across N datasets"
- Ablate: single-dataset vs fused
- Visualize Grad-CAM on 3+ classes
- Never claim SOTA unless compared on the same split

---

## 13. Session Log

| Date | Action | Result |
|---|---|---|
| setup | Created folder + AGENT.md + audit.py + train.py + baseline.yaml + README | Done |
| — | Phase 1 baseline on lab A4000 | PENDING |
| 2026-09-23 | Repo created + first commit, pushed to GitHub (now `multi-source-skin-disease-fusion`, branch master) | Done |
| 2026-09-23 | Fixed 3 bugs: `data_root` key fallback in train_resumable.py, `run_id`/absolute `results_dir` in baseline.yaml | Done |
| 2026-09-23 | Ran `audit.py --check-corrupt` -> `results/audit_starter.json` (1,706/366/367, 0 corrupt, all RGB) | Done |
| 2026-09-23 | CPU smoke test (2 epochs) on home PC -> test_acc 0.8747, macro_f1 0.8617; artifacts deleted | Done |
| 2026-09-23 | HF login (as Nirob-jon), public repo `Nirob-jon/cse475-skin-checkpoints` created, upload/download round-trip verified | Done |
| — | Phase 1 baseline on lab A4000 | PENDING |

---

## 11. Portable Training — Checkpoint Sync Protocol

### Why this exists
Bangladesh load shedding, 30 hr/week Kaggle quota, limited lab A4000 time, and a home RX580 with no CUDA all mean one training run must survive across four different machines. Solution: **one checkpoint file, four machines, seamless resume.**

### The command (same on every machine)
```powershell
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "<dataset root>" --resume auto
```
The script auto-detects platform and selects the matching profile from `configs/profiles.yaml`.

### Platform auto-detection
| Signal | Profile chosen |
|---|---|
| `/kaggle/working` exists | `kaggle_t4` |
| `google.colab` in sys.modules or `/content/drive` exists | `colab_t4` |
| CUDA GPU name contains "a4000" OR env `CSE475_MACHINE=lab` | `lab_a4000` |
| `torch_directml.is_available()` and no CUDA | `home_rx580_dml` |
| otherwise | `home_rx580_cpu` |

Override any time with `--profile <name>`.

### Checkpoint contents (all required for true resume)
- `model_state` — the weights
- `optimizer_state` — AdamW moments (CRITICAL, do not skip)
- `scheduler_state` — cosine LR position
- `epoch`, `best_f1`, `history`
- `rng_torch`, `rng_py`, `rng_np`, `rng_cuda` — random states for reproducible dataloader shuffle
- `config`, `profile`, `torch_version`

Files saved in `results/` (or profile-specific `checkpoint_dir`):
- `phase1_baseline_last.pt` — latest, used for resume
- `phase1_baseline_best.pt` — best val_acc so far
- `phase1_baseline_state.json` — human-readable epoch history
- `phase1_baseline.json` — final test results (written once, at end)

### Cross-machine accuracy — will results drift?
**No, not in any meaningful way.** The math is identical; the machine is just a calculator. Caveats:
- **Batch size must stay identical** across machines. Different batch → different gradient noise → different weights. All CUDA profiles already use 32. RX580-DirectML uses 16 (slightly noisier but still fine for a conference paper).
- **AMP on/off causes ~0.1% numeric difference.** Use AMP on all CUDA machines; disabled on DirectML/CPU. Negligible after 15 epochs.
- **RNG states are saved/restored** so dataloader shuffle order continues correctly across machines.
- **Expected difference:** epoch time (A4000 6-8× faster than RX580 DirectML) and last-digit numerical noise. **Not** final accuracy (within 0.1-0.3%).

### Sync hub: Hugging Face (not Google Drive)
- **HF free:** 100 GB private storage, per-file up to 200 GB, `hf upload-large-folder` supports interrupt/resume.
- **Google Drive free:** 15 GB shared with Gmail/Photos, no upload resume.
- Checkpoint size: ~20-40 MB each; whole project < 1 GB. HF is the clear pick.

Setup once per machine:
```powershell
pip install huggingface_hub
huggingface-cli login   # paste write-scope token
```
Then run with:
```powershell
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "..." \
  --hub hf --hf-repo Nirob-jon/cse475-skin-checkpoints --resume auto
```
On Kaggle: add `HF_TOKEN` as a notebook Secret.
On Colab: add `HF_TOKEN` in the left key panel.

### Sync workflow (per session)
1. Start: script auto-downloads `phase1_baseline_last.pt` from HF (if `--hub hf --resume auto`).
2. Train for as long as time/power/quota allows.
3. End: script uploads the newest `_last.pt` to HF after each epoch.
4. Next machine: repeat — nothing else to copy.

### Training recipe by dataset size
| Images | Model | Img | Batch | Epochs | AMP | Freeze head? | A4000 time |
|---|---|---|---|---|---|---|---|
| <1k | B0 | 224 | 16 | 20 | yes | yes (5 ep) | <5 min |
| 1k-5k | B0 | 224 | 32 | 15 | yes | yes (3 ep) | ~15 min |
| 5k-20k | B0 | 224 | 32 | 15 | yes | no | ~1 hr |
| >20k | B3 | 224 | 64 | 12 | yes | no | 2-4 hr |

### Time per platform (EfficientNet-B0, 224px, 15 epochs)
| Dataset | A4000+AMP | Kaggle T4 | Colab T4 | RX580 DML | RX580 CPU |
|---|---|---|---|---|---|
| 2,439 | ~6 min | ~11 min | ~13 min | ~1 hr | ~3 hr |
| 5,000 | ~15 min | ~25 min | ~28 min | ~2 hr | ~6 hr |
| 10,000 | ~25 min | ~45 min | ~50 min | ~4 hr (2 nights) | ~12 hr (overnight x2) |

### Why your senior needed days (and you won''t)
They likely: trained from scratch, used high resolution (384/512/1024px), used a huge model (B7/ViT-L), disabled AMP, tiny batch, 5-fold CV × 3 models × 3 seeds, 100-200 epochs, or dermoscopic 1024px ISIC images. Multiply those together = 100× wall-clock. Modern recipe (fine-tune + 224px + B0 + AMP + 15 epochs + 1 split) finishes 10k images in under an hour on A4000.

---

## 12. Session Log (append one row per work session)

| Date | Machine | What was done | Result |
|---|---|---|---|
| setup | — | Project folder created, AGENT.md/README/train.py/audit.py written | Phase 1 not started |
| — | — | Portable training harness added (train_resumable.py, profiles.yaml, SETUP_HOME.md, this section) | Phase 1 not started |
| 2026-09-23 | Home | GitHub repo created (`multi-source-skin-disease-fusion`, master), fixes to baseline.yaml + train_resumable.py, audit run, CPU smoke test (test_acc 0.8747 @ 2 ep) | Phase 1 prep done, lab run pending |
