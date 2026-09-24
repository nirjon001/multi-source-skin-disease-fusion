# AGENT.md — CSE475 Skin Disease Classification Project

**This file is the project memory. Read it before every session. Update it when facts change.**

---

## 1. Paper Angle (the contribution)

> *Cross-dataset generalization (and its limits) for South Asian / Bangladesh skin disease classification: train one model on DermNet 23-class, measure the accuracy drop on other datasets.*

This is a **methodology / evaluation paper**, not a fusion paper. The novelty is:
1. TRAIN one EfficientNet-B0 on DermNet (23 classes).
2. TEST on 4-5 different datasets -> a results table showing the generalization gap.
3. Harmonize label spaces across datasets (label_map.csv) so evaluation is meaningful.
4. Dedupe across datasets (imagehash) to prove no train/test leakage.
5. Measure skin-tone bias (DDI Black vs White, Fitzpatrick-black).

> NOTE: "fusion" (merging datasets that share labels) was the original plan but is **NOT being done** — the datasets share zero labels. See `docs/DEEPSEEK_CONVERSATION_PLAN.md` for the full reasoning.

---

## 2. GOLDEN RULE

**Met.** `results/phase1_baseline.json` exists with `test_acc: 0.9564`. Second dataset is confirmed open.

Phase 2 plan (cross-dataset eval) is now the active plan.

---

## 3. Hardware (confirmed)

### Home PC — CAN train via DirectML (proved in Phase 1)
| Spec | Value |
|---|---|
| CPU | AMD Ryzen 5 5600 |
| GPU | AMD RX580, 8GB VRAM, **no CUDA** (DirectML only) |
| RAM | 16GB |
| OS | Windows 11 (build 10.0.26200) |
| Hostname | NirjonPC1 |

**Phase 1 was trained here** on the RX580 via `torch-directml` (`profile: home_rx580_dml`) — test_acc 0.9564. It works but is slow (~4-6x slower than A4000). Use the home PC for: writing code, reading papers, `audit.py`, unzipping, label mapping, dedupe, eval_cross, Grad-CAM gallery, and training when the lab is unavailable.

### University Lab PC — PREFERRED TRAINING
| Spec | Value |
|---|---|
| CPU | Intel i7 12th / 13th / 14th gen |
| GPU | **NVIDIA A4000, 16GB VRAM, CUDA** |
| RAM | 16GB |

All `train_resumable.py` runs are interchangeable (same checkpoint, resume anywhere). Use the lab when possible; RX580 DirectML is the proven fallback.

### Fallback if lab access is unavailable
Kaggle Notebooks (free T4 / P100, ~30 hrs/week) or Google Colab free tier. Upload the dataset as a Kaggle Dataset and run the same `train_resumable.py` logic in a notebook. Code stays identical.

---

## 4. Python Environment (critical)

Three venvs exist on the home PC — **use the right one, never the global Python**:

| Interpreter | Version / torch | Status |
|---|---|---|
| `.\.venv-home\Scripts\python.exe` | **3.11 + torch 2.4.1 + torch-directml** | **GPU/DirectML training & gallery** |
| `.\.venv-home-cpu\Scripts\python.exe` | **3.11 + torch CPU-only** | **CPU checks, smoke tests, gallery on CPU** |
| `py -3.11` (global) | 3.11.9, **no torch** | only for the audit script's plain deps |

Installed in both venvs: numpy, pandas, scikit-learn, pillow, matplotlib, timm, tqdm, pyyaml, imagehash, huggingface_hub.

The lab venv (CUDA torch) is separate — do not mix.

### Home PC venv setup (one time)
```powershell
cd F:\cse475_skin
py -3.11 -m venv .venv-home
.\.venv-home\Scripts\Activate.ps1
pip install torch-directml   # replaces normal torch
pip install timm torchvision tqdm pyyaml numpy pillow matplotlib scikit-learn imagehash huggingface_hub
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
| 1 | **Starter: acne/rosacea/normal** | `F:\Downloads\skin_disease_images\` | acne, normal, rosacea | 2,439 | Phase 1 trained/evaluated |
| 2 | **DermNet 23 (Phase 2 TRAIN + test)** | `F:\Downloads\Kaggle-skin-disease-different-catergory dataset\` | 23 | train 14,314 / val 1,243 / test 4,002 | **Phase 2 training set** |
| 3 | SkinDiseaseBD | `F:\cse475_skin\data\SkinDiseaseBD\Updated Images\` | 5 (Dermatitis, Eczema, Scabies, Tinea, Vitiligo) | 1,612 | Extracted — **external test #2** |
| 4 | Fitzpatrick17k black-images | `F:\Downloads\fitzpatrick-black-images-dataset\fitzpatrick-black-images\` | ~114 | ~2,000 | External test #3 (3 unified) |
| 5 | DDI (Black/White) | `F:\Downloads\skin-disease-dataset\Disease-Dataset\` | binary st | Black 2,155 / White 2,000 | Bias probe only — **no disease labels** |
| 6 | SCIN (Google Research) | `F:\Downloads\SCIN (Google Research)\` | multi-label (CSV) | ~10,000+ PNG | NOT used (wrong modality/policy) |
| 7 | 32 Curated Categories | `F:\Downloads\32 Curated Categories of Skin Disease Images.zip` | 32 | ? | NOT used (dermoscopic) |
| 8 | HAM10000 / ISIC2018 | — | 7 | 10,015 | NOT FOUND / skipped (dermoscopic) |

### DermNet splits (verified, phase 2)
| Split | Images | Notes |
|---|---|---|
| train | 14,314 | after 8% per-class holdout to validation |
| validation | 1,243 | created 2026-09-24 (seed 42, 8%/class) for best-checkpoint selection |
| test | 4,002 | untouched in-domain test split |

### Starter dataset class counts (verified)
| Split | acne | normal | rosacea | Total |
|---|---|---|---|---|
| train | 460 | 476 | 770 | 1,706 |
| validation | 99 | 102 | 165 | 366 |
| test | 99 | 103 | 165 | 367 |
| **Total** | | | | **2,439** |

All images `.jpg`. Already split 70/15/15. No splitting needed.

---

## 7. Project Folder

```
F:\cse475_skin\
  AGENT.md              <- this file (project memory)
  README.md             <- quick start
  requirements.txt      <- pip list
  .gitignore            <- blocks data/, results/, *.pt
  configs\
    baseline.yaml       <- Phase 1 config (DO NOT EDIT — the Phase 1 record)
    smoke_cpu.yaml      <- smoke test config
    tutorial_cpu.yaml   <- tutorial config
    profiles.yaml       <- per-platform training profiles
    phase2_dermnet.yaml <- Phase 2 DermNet 23-class train config
    phase2_eval.yaml    <- Phase 2 eval config (the test-set list)
  src\
    audit.py            <- inspect ImageFolder dataset
    train_resumable.py  <- portable trainer (auto-detect + resume + HF sync)
    harmonize.py        <- build label_map.csv (phase 2)
    dedupe.py           <- imagehash near-duplicate report (phase 2)
    eval_cross.py       <- evaluate ONE checkpoint on N test sets (phase 2)
    gradcam_gallery.py  <- Grad-CAM image inspection gallery (mark images)
  data\                 <- working copies (gitignored)
  results\              <- checkpoints + JSON per run (gitignored)
  logs\
  notebooks\
    kaggle_phase1.ipynb / colab_phase1.ipynb / tutorial_phase1.ipynb
  docs\
    FILE_MAP.md, DEEPSEEK_CONVERSATION_PLAN.md, CONVERSATION_2026-09-23.md
```

---

## 8. Phase Plan (STRICT ORDER)

> Direction changed from fusion to cross-dataset generalization. Phase 1 = done.

### Phase 1 — Baseline — **DONE (2026-09-23)**
- [x] Trained `efficientnet_b0` on phase-1 dataset (home RX580, DirectML, 15 epochs)
- [x] `results/phase1_baseline.json` — **test_acc 0.9564**, macro_f1 0.9485
- [x] Note: actually ran on Colab then resumed on home RX580 (proved resume works)

### Phase 2 — Cross-dataset generalization (ACTIVE)
- [x] Extract SkinDiseaseBD → `data/SkinDiseaseBD/Updated Images` (1,612 imgs, 5 classes)
- [x] `src/harmonize.py` → `results/label_map.csv` (142 rows, DermNet+SkinDiseaseBD+Fitzpatrick+phase1)
- [x] `src/dedupe.py` → `results/dedupe_report_dermnet_sdb.json`: **0 cross-dataset pairs** (no leakage), 865 DermNet-internal, 157 SkinDiseaseBD-internal
- [x] `src/eval_cross.py` written + CPU smoke-tested (skindiseasebd n=1612, skin_disease_images n=264, ddi Black 2155/White 2000)
- [x] `configs/phase2_dermnet.yaml` + `configs/phase2_eval.yaml` written; DermNet validation split created (1,243 imgs)
- [ ] Train DermNet 23-class → `results/phase2_dermnet_best.pt`
- [ ] Run `src/eval_cross.py` → the results table
- [ ] Grad-CAM / inspection gallery (`src/gradcam_gallery.py`) on all phase-1 splits (test done)

### Phase 3 — Write paper (Week 6)
- [ ] IEEE template, 6-8 pages
- [ ] Submit to ICCIT / ICIEV / EWU conference

---

## 9. DO / DON'T

**DO**
- Use `.\.venv-home\Scripts\python.exe` (DirectML) or `.\.venv-home-cpu\Scripts\python.exe` (CPU) on home; lab venv (CUDA) on lab — **never the global `python`**
- Always seed 42
- Save every run as JSON in `results/`
- Log dataset + model + hyperparams in JSON
- Document label mapping before any cross-dataset eval (`results/label_map.csv`)
- Run dedupe before any cross-dataset eval (leakage kills papers)
- Keep `src/train_resumable.py` as the ONLY trainer
- Train on lab A4000 when available; RX580 DirectML is the proven fallback

**DON'T**
- Don't try "fusion" — the datasets share zero labels (see plan doc)
- Don't modify `configs/baseline.yaml` — Phase 1 record
- Don't use `Images_176x176_v1.zip` / `Images_512x512_v2.zip` — pre-augmented, leaks; use `Raw_Images.zip`
- Don't skip dedupe before cross-eval
- Don't report accuracy without a confusion matrix
- Don't commit datasets or `.pt` checkpoints to git
- Don't use HAM10000 / ISIC2018 / 32 Curated — dermoscopic, wrong modality
- Don't claim SOTA. This paper measures a gap; honest numbers beat high numbers.

---

## 10. Experiment Tracking Format

Every `results/*.json` (trainer output) must contain:
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

`results/phase2_dermnet.json` follows this schema (23 classes). `eval_cross.py` produces a per-test-set JSON keyed by `test_sets.<name>` with `n`, per-class F1 / accuracy, and (DDI) per-group stats.

---

## 11. Reproducibility Checklist

- [x] `seed_everything(42)` at top of `train_resumable.py`
- [ ] `cudnn.deterministic = True`
- [ ] Versions pinned in `requirements.txt`
- [ ] Git commit hash in results JSON
- [x] No leakage (dedupe ran before cross-dataset eval)
- [x] Same batch size across machines; AMP on CUDA only

---

## 12. Paper-Writing Rules

- Cite every dataset source (DermNet, SkinDiseaseBD, Fitzpatrick17k, DDI)
- Include harmonization table as a paper table (`results/label_map.csv`)
- Report dedupe stats: "0 cross-dataset near-duplicates across N datasets" (no leakage)
- Report the cross-dataset accuracy DROP (in-domain vs external) — this is the finding
- Report DDI Black vs White top-5 distribution + mean confidence (bias probe)
- Visualize Grad-CAM on 3+ classes
- Never claim SOTA unless compared on the same split

---

## 13. Session Log (legacy — pre-direction-change notes)

| Date | Action | Result |
|---|---|---|
| setup | Created folder + AGENT.md + audit.py + train.py + baseline.yaml + README | Done |
| 2026-09-23 | Repo created + first commit, pushed to GitHub (now `multi-source-skin-disease-fusion`, branch master) | Done |
| 2026-09-23 | Fixed 3 bugs: `data_root` key fallback in train_resumable.py, `run_id`/absolute `results_dir` in baseline.yaml | Done |
| 2026-09-23 | Ran `audit.py --check-corrupt` -> `results/audit_starter.json` (1,706/366/367, 0 corrupt, all RGB) | Done |
| 2026-09-23 | CPU smoke test (2 epochs) on home PC -> test_acc 0.8747, macro_f1 0.8617; artifacts deleted | Done |
| 2026-09-23 | HF login (as Nirob-jon), public repo `Nirob-jon/cse475-skin-checkpoints` created, upload/download round-trip verified | Done |
| 2026-09-23 | Deleted unused `src/train.py`; added `docs/FILE_MAP.md`; fixed stale `train.py` refs | Done |
| 2026-09-23 | **Phase 1 baseline DONE**: home RX580-DirectML run (resumed from Colab checkpoint), test_acc **0.9564**, macro_f1 0.9485 | Done |
| 2026-09-23 | Direction change decided: cross-dataset generalization, NOT fusion; plan doc `docs/DEEPSEEK_CONVERSATION_PLAN.md` | Done |
| 2026-09-23 | `src/harmonize.py` + `results/label_map.csv` (142 rows, verified: 0 missing) | Done |
| 2026-09-24 | `src/dedupe.py` + `results/dedupe_report_dermnet_sdb.json`: **0 cross-dataset pairs**, 865 DermNet-internal, 157 SDB-internal (dry-run) | Done |
| 2026-09-24 | `src/eval_cross.py` + `configs/phase2_eval.yaml` + `configs/phase2_dermnet.yaml`; CPU smoke test passed | Done |
| 2026-09-24 | `src/gradcam_gallery.py` — Grad-CAM image inspection gallery; smoke + phase-1 test-set run (acc 0.9428 with _best.pt) | Done |
| 2026-09-24 | Fixed phase2_dermnet.yaml data_root (parent dir); created DermNet `validation/` split (8%/class, 1,243 imgs, seed 42) | Done |
| 2026-09-24 | Committed + pushed Phase 2 pipeline (eval_cross, dedupe, gradcam_gallery, configs, docs) `e3febe4` | Done |
| — | Phase 2 DermNet 23-class training (RX580 or A4000) | PENDING |
| — | `src/eval_cross.py` full run → results table | PENDING |

---

## 14. Portable Training — Checkpoint Sync Protocol

### Why this exists
Bangladesh load shedding, 30 hr/week Kaggle quota, limited lab A4000 time, and a home RX580 with no CUDA all mean one training run must survive across four different machines. Solution: **one checkpoint file, four machines, seamless resume.**

### The command (same on every machine)
```powershell
python src\train_resumable.py --config configs\baseline.yaml --data "<dataset root>" --resume auto
```
Use the activated venv's `python` on each machine (lab `.venv` = CUDA, home `.venv-home` = DirectML, home `.venv-home-cpu` = CPU). The script auto-detects platform and selects the matching profile from `configs/profiles.yaml`.

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
python src\train_resumable.py --config configs\baseline.yaml --data "..." \
  --hub hf --hf-repo Nirob-jon/cse475-skin-checkpoints --resume auto
```
On Kaggle: add `HF_TOKEN` as a notebook Secret.
On Colab: add `HF_TOKEN` in the left key panel.

### Sync workflow (per session)
1. Start: script auto-downloads `<run>_last.pt` from HF (if `--hub hf --resume auto`).
2. Train for as long as time/power/quota allows.
3. End: script uploads the newest `_last.pt` (and `_best.pt`) to HF after each epoch.
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

## 15. Session Log (append one row per work session)

| Date | Machine | What was done | Result |
|---|---|---|---|
| setup | — | Project folder created, AGENT.md/README/train.py/audit.py written | Phase 1 not started |
| — | — | Portable training harness added (train_resumable.py, profiles.yaml, SETUP_HOME.md, this section) | Phase 1 not started |
| 2026-09-23 | Home | GitHub repo created (`multi-source-skin-disease-fusion`, master), fixes to baseline.yaml + train_resumable.py, audit run, CPU smoke test (test_acc 0.8747 @ 2 ep) | Phase 1 prep done, lab run pending |
| 2026-09-23 | Colab→Home | **Phase 1 baseline** (efficientnet_b0, 15 ep): started on Colab T4, resumed on home RX580-DirectML, test_acc **0.9564**, macro_f1 0.9485 | Phase 1 DONE; golden rule met |
| 2026-09-23 | Home | Direction change: documented in `docs/DEEPSEEK_CONVERSATION_PLAN.md` (cross-dataset generalization, not fusion) | Plan finalised |
| 2026-09-23 | Home | Harmonization: `src/harmonize.py`, `results/label_map.csv` (142 rows, 0 missing) | Done |
| 2026-09-24 | Home | Dedupe: `src/dedupe.py` → report with 0 cross-dataset pairs (no leakage), 865 DermNet / 157 SDB internal | Done |
| 2026-09-24 | Home | Phase 2 eval pipeline: `src/eval_cross.py` (5 test sets, CPU smoke-tested), `configs/phase2_dermnet.yaml`, `configs/phase2_eval.yaml` | Done |
| 2026-09-24 | Home | Grad-CAM gallery tool `src/gradcam_gallery.py`; run on phase-1 test split (367 img, acc 0.9428 @ _best.pt) → browseable HTML | Done |
| 2026-09-24 | Home | Prepared DermNet for training: fixed `data_root` (parent dir), created `validation/` split (1,243 imgs, 8%/class, seed 42) | Done |
| 2026-09-24 | Home | Committed + pushed Phase 2 pipeline `e3febe4`; updated all Markdown docs | Done |
| — | — | Phase 2 DermNet 23-class training → `results/phase2_dermnet_best.pt` | PENDING |
