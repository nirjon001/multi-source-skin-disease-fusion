# DEEPSEEK CONVERSATION -> PLAN FOR OPENCODE

> Audience: opencode (the coding agent).
> Purpose: Single handoff. What changed, what was decided, what code to write.
> Date: 2026-09-23 (updated 2026-09-24)
> Status: Phase 1 done. Phase 2 plan finalised. **Phase 2 code WRITTEN and smoke-tested** (harmonize, dedupe, eval_cross, configs, gradcam_gallery). DermNet validation split ready. Pending: DermNet training + eval_cross full run.

---

## 0. TL;DR FOR OPENCODE

The direction changed. We are no longer doing "dataset fusion". We are doing cross-dataset generalization testing:

1. TRAIN one model on DermNet 23-class (already on disk, ~17k train images).
2. TEST it on four different test sets -> a 4-row results table.

Code to write (in order):

1. src/harmonize.py - label mapping table (DermNet class -> unified class)
2. src/dedupe.py - phash near-duplicate removal across datasets (leakage guard)
3. src/eval_cross.py - load one checkpoint, evaluate on N test sets, dump JSON
4. configs/phase2_dermnet.yaml - 23-class training config
5. configs/phase2_eval.yaml - test-set list + class mappings

Do NOT modify results/ or data/. Do NOT delete or rename existing files.

---

## 1. WHERE WE WERE (Phase 1 - DONE)

results/phase1_baseline.json
  run_id      : phase1_baseline
  dataset     : skin_disease_images  (Neperl acne/normal/rosacea)
  model       : efficientnet_b0
  profile     : home_rx580_dml   (DirectML on the home RX580)
  epochs      : 15
  test_acc    : 0.9564
  macro_f1    : 0.9485

Phase 1 was a learning run. It proved the pipeline works end to end:
audit -> train -> resume -> checkpoint -> HF sync -> results JSON.
Nothing about it needs to be redone.

Golden rule ("no second dataset until results/phase1_baseline.json has a test_acc") is now satisfied.

---

## 2. THE DIRECTION CHANGE

What the user originally thought:
"Combine multiple datasets into one big dataset to get better accuracy."
That is fusion - merging datasets that share labels so each class gets more images.

Why fusion does not work here:

| Dataset                          | Classes                                               | Overlap with Phase 1 (acne/normal/rosacea) |
| -------------------------------- | ----------------------------------------------------- | ------------------------------------------ |
| skin_disease_images (Phase 1)    | acne, normal, rosacea                                 | - (anchor)                                 |
| SkinDiseaseBD                    | Dermatitis, Eczema, Scabies, Tinea Ringworm, Vitiligo | ZERO                                       |
| fitzpatrick-black-images-dataset | ~114 classes, 8,220 images                            | ~3 (acne, eczema, dermatitis)              |

Merging SkinDiseaseBD with Phase 1 would be concatenation, not fusion - no shared labels, no per-class gain.

What the user actually wants (his own words):
"I wanted a model to be able to predict different kinds of disease, and another dataset to test it out whether it does work or not."

Two distinct jobs:

1. Broad training - train one model on many diseases.
2. Cross-dataset testing - test it on a different dataset and measure the drop.

| Term                         | Meaning                                                   | In this project |
| ---------------------------- | --------------------------------------------------------- | --------------- |
| Fusion                       | merge datasets that share labels -> more images per class | NOT doing       |
| Multi-class / broad training | train on ONE dataset with many diseases                   | job 1           |
| Cross-dataset testing        | test on a DIFFERENT dataset -> measure generalization     | job 2           |

---

## 3. FINAL DATASET DECISION (verified on disk 2026-09-23)

### 3.1 TRAIN set

| Field        | Value                                                                               |
| ------------ | ----------------------------------------------------------------------------------- |
| Path         | F:\Downloads\Kaggle-skin-disease-different-catergory dataset                        |
| Real name    | DermNet 23-class                                                                    |
| Structure    | train/ and test/ subfolders, each with 23 named class folders                       |
| Train images | ~17,000 total                                                                       |
| Modality     | Clinical photos                                                                     |
| Why chosen   | Large, balanced, 23 real diseases, already split, overlaps with every other dataset |

Verified DermNet train class counts:

Psoriasis pictures Lichen Planus and related diseases               1405
Seborrheic Keratoses and other Benign Tumors                        1371
Tinea Ringworm Candidiasis and other Fungal Infections              1300
Eczema Photos                                                       1235
Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions  1149
Warts Molluscum and other Viral Infections                          1086
Nail Fungus and other Nail Disease                                  1040
Acne and Rosacea Photos                                              840
Systemic Disease                                                     606
Light Diseases and Disorders of Pigmentation                        568
Atopic Dermatitis Photos                                             489
Vascular Tumors                                                      482
Melanoma Skin Cancer Nevi and Moles                                  463
Bullous Disease Photos                                               448
Scabies Lyme Disease and other Infestations and Bites                431
Lupus and other Connective Tissue diseases                           420
Vasculitis Photos                                                    416
Herpes HPV and other STDs Photos                                     405
Exanthems and Drug Eruptions                                         404
Cellulitis Impetigo and other Bacterial Infections                   288
Poison Ivy Photos and other Contact Dermatitis                       260
Hair Loss Photos Alopecia and other Hair Diseases                    239
Urticaria Hives                                                      212

### 3.2 TEST sets (four)

| # | Dataset                | Path                                                                                      | Classes used                                     | Purpose                                     |
| - | ---------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------- |
| 1 | DermNet own test split | .../Kaggle-skin-disease-different-catergory dataset/test                                  | 23                                               | in-domain upper bound                       |
| 2 | SkinDiseaseBD          | F:\Downloads\SkinDiseaseBD A Dataset of Common Skin Disease Ima -> extract Raw_Images.zip | 5 (Eczema, Scabies, Tinea, Dermatitis, Vitiligo) | CROSS-DATASET DOMAIN SHIFT - paper headline |
| 3 | Fitzpatrick17k black   | F:\Downloads\fitzpatrick-black-images-dataset\fitzpatrick-black-images\                   | 3 (acne, eczema, allergic_contact_dermatitis)    | cross-dataset + dark skin                   |
| 4 | DDI                    | F:\Downloads\skin-disease-dataset\Disease-Dataset\Black and \White                        | binary (Black vs White)                          | skin-tone bias gap                          |

### 3.3 Datasets to SKIP

| Dataset                               | Reason                                                 |
| ------------------------------------- | ------------------------------------------------------ |
| HAM10000 / ISIC2018 (dataverse_files) | Dermoscopic - wrong modality                           |
| 32 Curated Categories (32_CSD.rar)    | ISIC-derived, dermoscopic - wrong modality             |
| SCIN                                  | Multi-label, needs CSV join; optional extra point only |

---

## 4. THE 5-CLASS HARMONIZATION MAP (the paper real contribution)

| Unified class | DermNet folder(s)                                                         | SkinDiseaseBD folder(s) | Fitzpatrick17k folder(s)    |
| ------------- | ------------------------------------------------------------------------- | ----------------------- | --------------------------- |
| eczema        | Eczema Photos                                                             | Eczema                  | eczema, dyshidrotic_eczema  |
| scabies       | Scabies Lyme Disease and other Infestations and Bites                     | Scabies                 | -                           |
| tinea         | Tinea Ringworm Candidiasis and other Fungal Infections                    | Tinea Ringworm          | -                           |
| dermatitis    | Atopic Dermatitis Photos + Poison Ivy Photos and other Contact Dermatitis | Dermatitis              | allergic_contact_dermatitis |
| vitiligo      | Light Diseases and Disorders of Pigmentation                              | Vitiligo                | -                           |
| acne (extra)  | Acne and Rosacea Photos                                                   | -                       | acne, acne_vulgaris         |

NOTE: dermatitis maps to TWO DermNet folders - the merge happens at dataset-load time, not by folder. Same for Fitzpatrick eczema (2 folders) and acne (2 folders).

---

## 5. THE 4-ROW RESULTS TABLE

| Row | Train        | Test                 | Classes | What it proves                  |
| --- | ------------ | -------------------- | ------- | ------------------------------- |
| 1   | DermNet (23) | DermNet test/        | 23      | in-domain upper bound (~85-90%) |
| 2   | DermNet (23) | SkinDiseaseBD        | 5       | CROSS-DATASET DOMAIN SHIFT      |
| 3   | DermNet (23) | Fitzpatrick17k black | 3       | cross-dataset + dark skin       |
| 4   | DermNet (23) | DDI Black vs White   | binary  | SKIN-TONE BIAS GAP              |

One trained model. Four tests. That is the paper.

---

## 6. WHAT OPENCODE NEEDS TO WRITE

### 6.1 src/harmonize.py

Build a CSV mapping every source class -> unified class.

Inputs:

- --dermnet-root "F:/Downloads/Kaggle-skin-disease-different-catergory dataset"
- --skindiseasebd-root "F:/cse475_skin/data/SkinDiseaseBD_raw"
- --fitzpatrick-root "F:/Downloads/fitzpatrick-black-images-dataset/fitzpatrick-black-images"
- --out results/label_map.csv

Output CSV columns:
source_dataset,source_class,unified_class

Behavior:

- Walk each dataset folder names.
- Apply the mapping in section 4.
- Skip folders not in the mapping (the 18 non-overlapping DermNet classes - fine, they stay 23-class for training; only TEST uses the 5).
- Exit non-zero with a clear error if any expected folder is missing.

### 6.2 src/dedupe.py

Remove near-duplicate images WITHIN and ACROSS datasets. This is the leakage guard - a contribution, not a utility.

Method: imagehash.phash, Hamming distance <= 5.

Inputs:

- --roots (comma-separated dataset roots)
- --threshold 5
- --out results/dedupe_report.json
- --dry-run (default true on first pass, report only)

Output JSON:
{
  "per_dataset": {"dermnet": 17000, "skindiseasebd": 1612, "fitzpatrick": 8220},
  "within_dataset_dupes_removed": 0,
  "cross_dataset_dupes_removed": 0,
  "dupes_removed": [{"kept": "path/a.jpg", "dropped": "path/b.jpg", "distance": 3}]
}

Critical: run BEFORE any cross-dataset evaluation. If the same photo appears in DermNet AND SkinDiseaseBD, accuracy is inflated and the paper is invalid.

### 6.3 src/eval_cross.py

Load ONE trained checkpoint, evaluate on N test sets, dump one JSON.

**DESIGN DECISION (locked 2026-09-24):** use the model's **raw 23-class argmax, then map the predicted class → unified** via `results/label_map.csv`. Do NOT use max-logit reduction — it is degenerate when a test set has a single unified class (e.g. the phase-1 acne+rosacea→acne row). `label_map.csv` is keyed by (source_dataset, source_class) → unified_class.

Inputs:

- --checkpoint results/phase2_dermnet_best.pt
- --config configs/phase2_eval.yaml
- --device auto|dml|cpu

configs/phase2_eval.yaml sketch:

test_sets:

- name: dermnet_test
  root: F:/Downloads/Kaggle-skin-disease-different-catergory dataset/test
  mode: folder_per_class
  classes: all_23
- name: skindiseasebd
  root: F:/cse475_skin/data/SkinDiseaseBD_raw
  mode: folder_per_class
  classes: [eczema, scabies, tinea, dermatitis, vitiligo]
  label_map: results/label_map.csv
- name: fitzpatrick_black
  root: F:/Downloads/fitzpatrick-black-images-dataset/fitzpatrick-black-images
  mode: folder_per_class
  classes: [acne, eczema, dermatitis]
  label_map: results/label_map.csv
- name: ddi
  root: F:/Downloads/skin-disease-dataset/Disease-Dataset
  mode: subfolder_binary
  groups: [Black, White]

Output JSON schema:
{
  "checkpoint": "results/phase2_dermnet_best.pt",
  "model": "efficientnet_b0",
  "test_sets": {
    "dermnet_test": {"accuracy": 0.0, "macro_f1": 0.0, "confusion_matrix": [[0]], "n": 0},
    "skindiseasebd": {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}, "n": 0},
    "fitzpatrick_black": {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}, "n": 0},
    "ddi": {"accuracy_black": 0.0, "accuracy_white": 0.0, "gap_pp": 0.0, "n_black": 0, "n_white": 0}
  }
}

### 6.4 configs/phase2_dermnet.yaml

Copy the shape of configs/baseline.yaml but change:

- data_root: F:/Downloads/Kaggle-skin-disease-different-catergory dataset/train
- run_id: phase2_dermnet
- keep model: efficientnet_b0, epochs: 15, img_size: 224, seed: 42
- drop batch_size to 16 if 23 classes does not fit - test-fit and pick.

DermNet is 17k images vs Phase 1 1.7k -> ~10x longer per epoch.
Budget ~1 hr on A4000, ~3 hr on RX580 DirectML.

### 6.5 Extract task - SkinDiseaseBD (one-time command)

& "C:\Program Files\7-Zip\7z.exe" x "F:\Downloads\SkinDiseaseBD A Dataset of Common Skin Disease Ima\SkinDiseaseBD A Dataset of Common Skin Disease Ima\Raw_Images.zip" -o"F:\cse475_skin\data\SkinDiseaseBD_raw" -y

WARNING: use Raw_Images.zip ONLY.
Images_176x176_v1.zip contains pre-augmented images (aug_ prefix) - the same photo appears multiple times, which would leak between train and test. That leakage is itself part of the paper story, so it must be measured, not accidental.

---

## 7. RULES FOR OPENCODE

DO:

- Read docs/FILE_MAP.md and AGENT.md first.
- Use the activated venv's python: home `.venv-home` (DirectML) / `.venv-home-cpu` (CPU), lab `.venv` (CUDA). The global `python` (3.14.5) and `py -3.11` have no torch wheels.
- Keep src/train_resumable.py as the ONLY trainer. Do not re-add src/train.py.
- Save every run as JSON in results/ with the schema in AGENT.md section 10.
- Pin seed 42 in every training and evaluation run.
- Add a session-log row to AGENT.md section 15 at session end.

DO NOT:

- Do not delete, rename, or move anything in data/, results/, or any dataset folder.
- Do not modify configs/baseline.yaml - it is the Phase 1 record. Make new files instead.
- Do not skip dedupe.py. It is not optional.
- Do not use Images_176x176_v1.zip or Images_512x512_v2.zip - pre-augmented, leaks.
- Do not touch HAM10000, ISIC2018, or 32 Curated Categories - dermoscopic, wrong modality.
- Do not claim SOTA. This paper measures a gap; it does not beat a benchmark.
- Do not chase accuracy. Honest numbers beat high numbers.

---

## 8. WHAT TO DO RIGHT NOW (first three steps) — DONE 2026-09-24

1. Extract SkinDiseaseBD → `data/SkinDiseaseBD/Updated Images` (1,612 imgs, 5 classes: Dermatitis 302 · Eczema 381 · Scabies 301 · Tinea 316 · Vitiligo 312). NOTE: source is the `Images_512x512_v2.zip\Updated Images\<class>\` split; `aug_N` prefix = class index, NOT augmentation; `Raw_Images.zip` (flat unlabeled `ResearchImage\*.jpg`, 197 files) is unusable. — DONE.
2. `src/harmonize.py` → `results/label_map.csv` (142 rows, 0 missing) — DONE.
3. `src/dedupe.py` → `results/dedupe_report_dermnet_sdb.json`: **0 cross-dataset pairs** (no leakage), 865 DermNet-internal, 157 SkinDiseaseBD-internal. — DONE.

Also done: `src/eval_cross.py` + `configs/phase2_eval.yaml` + `configs/phase2_dermnet.yaml` (CPU smoke test passed), and the DermNet `validation/` split (1,120 imgs after removing 124 DermNet-internal exact dups that straddled train↔val — re-dedupe confirms **0 cross-split pairs**).

NEXT (in order) — **2026-09-24 update: training switched to Kaggle T4**:

1. Train DermNet 23-class on **Kaggle** via `notebooks/kaggle_phase2.ipynb` (T4, batch 32, AMP on, ~30-90 min; resumable via HF). Prepared split is on HF as `Nirob-jon/cse475-dermnet-split`. Home fallback (slower, DirectML no AMP): `.\.venv-home\Scripts\python.exe src\train_resumable.py --config configs\phase2_dermnet.yaml --profile home_rx580_dml --hub hf --resume auto`
2. Evaluate: `.\.venv-home\Scripts\python.exe src\eval_cross.py --checkpoint results\phase2_dermnet_best.pt --config configs\phase2_eval.yaml`
3. Phase-2 Colab/kaggle *eval* notebook (cloud dataset slugs TBD).

---

## 9. OPEN QUESTIONS (ask the user - do not resolve alone)

1. SCIN handling - multi-label; collapse to dominant label or handle explicitly? Default recommendation: collapse to dominant label and state it in one sentence in Methods.
2. DDI as test set - DDI Black/White folders are flat (no per-disease subfolders). So DDI measures skin-tone bias, not disease accuracy. Confirm with the user that this is the intended use.
3. Training location for Phase 2 - 17k images -> ~1 hr on lab A4000 vs ~3 hr on home RX580 DirectML. Both work (Phase 1 proved the RX580 path). User call.

---

## 10. REFERENCE - PROJECT FILES AS OF 2026-09-23

F:\cse475_skin
  AGENT.md                  project memory (rules, session log section 15)
  README.md                 quick start
  SETUP_HOME.md             home PC (RX580 DirectML/CPU) setup
  requirements.txt          shared deps
  .gitignore
  configs
    baseline.yaml           Phase 1 config  (DO NOT EDIT)
    profiles.yaml           per-machine settings
    smoke_cpu.yaml          smoke test config
    tutorial_cpu.yaml       tutorial config
  src
    audit.py                dataset inspection
    train_resumable.py      THE trainer
  scripts
    lab_phase1.ps1          lab runbook
  notebooks
    colab_phase1.ipynb
    kaggle_phase1.ipynb
    tutorial_phase1.ipynb
  docs
    CONVERSATION_2026-09-23.md      session log (Phase 1)
    FILE_MAP.md                     what to run/read/ignore
    DEEPSEEK_CONVERSATION_PLAN.md   THIS FILE
  results\                  Phase 1 outputs (gitignored)
  logs
  data\                     (empty until SkinDiseaseBD extraction)

---

End of handoff. Questions -> ask the user, do not guess.

---

## 11. NEW FINDINGS 2026-09-24 (DeepSeek verification pass)

### 11.1 CORRECTION — DermNet HAS a `validation/` split

An earlier DeepSeek reply claimed the DermNet dataset had only `train/` and `test/`,
with NO validation folder, and recommended carving 15% out of `train/`.

**That claim was WRONG.** The dataset has THREE splits:

```
F:\Downloads\Kaggle-skin-disease-different-catergory dataset\
    train\        23 class folders
    validation\   23 class folders
    test\         23 class folders
```

`src/train_resumable.py` line 169 does
`find_split("validation", "val", "Val", "valid")` and WILL find it.
No data-prep script was needed. The error came from inferring off a truncated
directory listing instead of re-listing the parent folder directly.

### 11.2 NEW FINDING — DermNet splits are NOT random; cross-split duplicates exist

Inspection of actual filenames inside `Acne and Rosacea Photos` across all three splits
shows the splits are grouped by filename family, and the SAME PHOTO appears in more
than one split.

**Proof 1 — identical filenames in two splits:**

| filename                    | in train/ | in test/ |
| --------------------------- | :-------: | :------: |
| `07PerioralDermEye.jpg`   |    yes    |   yes   |
| `07Rhinophyma1.jpg`       |    yes    |   yes   |
| `07rhnophymas0321051.jpg` |    yes    |   yes   |
| `07RosaceaMilia0120.jpg`  |    yes    |   yes   |

**Proof 2 — filename families cluster into one split:**

- `train/` holds the whole series `07RosaceaK0216`, `K02161`...`K02166` together
- `validation/` holds `acne-cystic-4`, `-11`, `-14`, `-17`, `-36`, `-94`, `-105`, `-110`, `-114`, `-119`, `-128` as a block
- `test/` holds `acne-closed-comedo-1`, `-2`, `-3`, `-13`, `-15`, `-16`, `-21`, `-22`, `-24`, `-25`, `-26`, `-28`, `-29` as a block

**Proof 3 — split ratios are inconsistent with a random split:**
`validation/` is consistently ~7-9% of each class, while `test/` is ~25-30%.
A uniform random 70/15/15 would not produce that pattern.

**This is genuine train/test leakage** — the model can memorise a photo seen during
training and appear correct when that same photo reappears in the test set.
It is a defect of the DermNet Kaggle dataset, not of this project pipeline.

### 11.3 What this means for the dedupe report

The current `results/dedupe_report_dermnet_sdb.json` reported
**0 cross-dataset pairs** between DermNet and SkinDiseaseBD, plus 865 DermNet-internal
and 157 SkinDiseaseBD-internal near-duplicates.

That report treated the DermNet `train/` folder as a single unit.
It did **not** compare `train/` against `validation/` or `test/`.
Given finding 11.2, a cross-split dedupe pass is **RESOLVED** (2026-09-24):

- Ran `dedupe.py --roots train;validation` → `results/dedupe_report_dermnet_trainval.json`.
- Found **120 train↔validation near-dups (98 pixel-identical)** — the 8% holdout had landed on DermNet's internal same-photo-multiple-classes pairs.
- **Fix applied:** removed the 124 validation-side copies (train kept intact).
- Re-ran dedupe: **741 DermNet-internal pairs, 0 train↔validation cross-split pairs → CLEAN**.
- Final split counts: train 14,314 / validation 1,120 / test 4,002.

---

## 12. WHAT TO DO NEXT (revised, in order)

### Step 1 — extend `src/dedupe.py` for cross-split checking

Add a mode that walks **all three DermNet splits together** and flags any image whose
phash matches an image in a DIFFERENT split (Hamming <= 5).

Produce `results/dedupe_cross_split_dermnet.json` with at minimum:

```json
{
  "method": "imagehash.phash",
  "threshold": 5,
  "train_vs_validation": <count>,
  "train_vs_test": <count>,
  "validation_vs_test": <count>,
  "examples": [{"kept": "...", "dropped": "...", "splits": "train/test", "distance": 0}]
}
```

Do this **before** training. The numbers go in the paper as the leakage quantification.

### Step 2 — decide the fix strategy (ask the user; two valid paths)

**Path A — dedupe across splits, keep DermNet native splits (recommended).**
Drop any `validation/` or `test/` image that matches a `train/` image. Report surviving
counts. Keeps comparability with published DermNet numbers and turns the leakage into
a measured, documented fix — which matches the paper's stated angle.

**Path B — re-split from scratch.**
Pool all three splits, dedupe globally, then random 70/15/15 with seed 42.
Cleaner but loses comparability with published DermNet numbers.

Path A is the stronger paper choice.

### Step 3 — train DermNet 23-class

Only after Step 1-2 are settled:

```
.\.venv-home\Scripts\python.exe src\train_resumable.py --config configs\phase2_dermnet.yaml --profile home_rx580_dml --resume auto
```

### Step 4 — run cross-dataset evaluation

```
.\.venv-home\Scripts\python.exe src\eval_cross.py --checkpoint results\phase2_dermnet_best.pt --config configs\phase2_eval.yaml
```

### Step 5 — write the paper

Four results rows (in-domain DermNet, SkinDiseaseBD, Fitzpatrick black, DDI bias probe),
plus two Limitations items that are actually findings:

- cross-class duplicates: same image, two different disease labels
- cross-split duplicates: same image in train and test

---

## 13. VERIFIED SPLIT COUNTS (2026-09-24)

Total 19,559 `.jpg` files. Per-class counts (train / validation / test):

| Class                                          | train | val | test |
| ---------------------------------------------- | ----: | --: | ---: |
| Acne and Rosacea Photos                        |   773 |  67 |  312 |
| Actinic Keratosis BCC and other Malignant      |  1057 |  92 |  288 |
| Atopic Dermatitis Photos                       |   450 |  39 |  123 |
| Bullous Disease Photos                         |   412 |  36 |  113 |
| Cellulitis Impetigo and other Bacterial        |   265 |  23 |   73 |
| Eczema Photos                                  |  1136 |  99 |  309 |
| Exanthems and Drug Eruptions                   |   372 |  32 |  101 |
| Hair Loss Photos Alopecia                      |   220 |  19 |   60 |
| Herpes HPV and other STDs Photos               |   373 |  32 |  102 |
| Light Diseases and Disorders of Pigmentation   |   523 |  45 |  143 |
| Lupus and other Connective Tissue diseases     |   386 |  34 |  105 |
| Melanoma Skin Cancer Nevi and Moles            |   426 |  37 |  116 |
| Nail Fungus and other Nail Disease             |   957 |  83 |  261 |
| Poison Ivy Photos and other Contact Dermatitis |   239 |  21 |   65 |
| Psoriasis pictures Lichen Planus               |  1293 | 112 |  352 |
| Scabies Lyme Disease and other Infestations    |   397 |  34 |  108 |
| Seborrheic Keratoses and other Benign Tumors   |  1261 | 110 |  343 |
| Systemic Disease                               |   558 |  48 |  152 |
| Tinea Ringworm Candidiasis and other Fungal    |  1196 | 104 |  325 |
| Urticaria Hives                                |   195 |  17 |   53 |
| Vascular Tumors                                |   443 |  39 |  121 |
| Vasculitis Photos                              |   383 |  33 |  105 |
| Warts Molluscum and other Viral Infections     |   999 |  87 |  272 |

**Totals:** train 14,314 · validation ~1,240 · test 4,002 · grand total 19,559.

Note: the validation column is consistently ~7-9% of each class, not 15%.
That is the signature of the non-random split described in 11.2.

---

## 14. CORRECTION TO SECTION 3.1

Section 3.1 above says: "Structure | train/ and test/ subfolders, each with 23 named class folders".

That is WRONG. Correct to: "Structure | train/, validation/, and test/ subfolders, each with 23 named class folders".

---

End of update 2026-09-24. Questions -> ask the user, do not guess.
