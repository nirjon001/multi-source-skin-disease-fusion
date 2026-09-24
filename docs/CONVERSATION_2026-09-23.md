# CSE475 Session Log — 2026-09-23 (+ 2026-09-24 updates)

> Project: **Cross-dataset generalization for South Asian / Bangladesh skin disease classification** (formerly "fusion" — direction changed, see `DEEPSEEK_CONVERSATION_PLAN.md`)
> Folder: `F:\cse475_skin`
> This file is a saved transcript/summary of the working session (exports of the chat conversation). Key results and decisions are captured inline.

---

## 0. Starting point (from memory of the session)

The session resumed mid-way through a **de-cluttering** step: `src/train.py` (a legacy duplicate trainer) had just been deleted, and stale references to it still existed in `README.md`, `AGENT.md`, `configs/baseline.yaml`, and `notebooks/tutorial_phase1.ipynb`. A new `docs/FILE_MAP.md` also still needed to be created.

---

## 1. "What did we do so far?"

Summarised the whole project history: Phase 1 baseline (EfficientNet-B0, acne/normal/rosacea, 2,439 images), audit done, smoke test passed on CPU (test_acc 0.8747), GitHub repo `nirjon001/multi-source-skin-disease-fusion` created, Hugging Face repo `Nirob-jon/cse475-skin-checkpoints` created and verified, notebooks made (Kaggle/Colab/tutorial), and the in-progress de-clutter.

### Then executed the de-clutter plan
- Fixed stale `train.py` references in `README.md`, `AGENT.md` (lines 53/56/145/163/234), `configs/baseline.yaml`, and the tutorial notebook's file-role table + Part 6 (rewritten as "train.py — removed, single trainer is `train_resumable.py`").
- Created `docs/FILE_MAP.md` — one-page "run / read / ignore" file map.
- Appended a session-log row to `AGENT.md`.
- Committed and pushed: `d2c4111  Remove unused train.py trainer; add docs/FILE_MAP.md; update references`.

---

## 2. "okay now how to start training? baseline right?"

- Confirmed baseline = one command: `train_resumable.py + configs/baseline.yaml` (EfficientNet-B0, 15 epochs, batch 32, lr 1e-3, seed 42).
- While checking machine state, **discovered a real surprise**: the Hugging Face repo `Nirob-jon/cse475-skin-checkpoints` already contained `phase1_baseline_last.pt` — a **Colab run had already started the real baseline** (`profile: colab_t4`, **epoch 1/15 done, train 87.3%, val 96.2%**, then the session died).
- Cleaned up leftover `smoke_test*` files in `results/`.
- Inspected the checkpoint: it was a **complete resumable tape** (model_state, optimizer_state, scheduler, all 5 RNGs).
- Found **two real bugs** in `train_resumable.py`:
  1. `_last.pt` was saved with `best_f1` **before** the best-update ran → HF checkpoint carried `best_f1=0.0` while history showed 0.9617. Corrupts best-model tracking on resume.
  2. Only `_last.pt` was uploaded to HF; the `_best.pt` (paper model) was never synced — lost if best epoch ≠ last epoch and you resume elsewhere.
- Fixed both: reordered so best-update happens **before** the `_last.pt` save; added `_best.pt` upload to HF; added a **repair** on resume (`best_f1 = max(best_f1, max val_acc in history)`). Verified: repair 0.0 → 0.9617, resume would start at epoch index 1. Compiled OK.
- Committed/pushed: `632f873  Fix best_f1 ordering in _last checkpoint; sync _best to HF; repair best_f1 on resume`.

---

## 3. User: "will it not use gpu?" / dependent command issue

- User first tried with `py -3.11 ...` inside the activated `.venv-home` and hit:
  ```
  ModuleNotFoundError: No module named 'torch'
  ```
- Root cause: **`py -3.11` resolves to the GLOBAL Python 3.11** (`C:\Users\nirjo\AppData\Local\Programs\Python\Python311\python.exe`), which has no torch. The venv **does** have torch (`2.4.1`, DirectML-aware) but is reached via the plain `python` command.
- Verified DML GPU works on the RX 580 (`has_dml=True`, `is_home=True` → profile `home_rx580_dml`). The trainer auto-detects it.

---

## 4. User ran the baseline on the home RX580

One more blocker: `yaml.parser.ParserError: expected '<document start>' ... line 4, column 1`.

- Root cause: `configs/baseline.yaml` began with a **UTF-8 BOM** (`\xef\xbb\xbf`), which PyYAML cannot parse.
- Fixed: stripped the BOM from `baseline.yaml` and made the trainer BOM-proof (read YAML as `utf-8-sig`). Committed/pushed.

---

## 5. "okay i have done train test acc is 0.9564 what now?"

- Verified `results/phase1_baseline.json`:
  - **test_acc 0.9564**, **macro_f1 0.9485**
  - per-class F1: acne 0.926 · normal 0.925 · rosacea **0.994**
  - model efficientnet_b0 · profile home_rx580_dml · 15 epochs · seed 42 · test 367 imgs
- Best val 0.9727 (epochs 7 & 15). History recorded the full 15-epoch run.
- This **satisfies the golden rule**: `results/phase1_baseline.json` exists with `test_acc` → **Phase 2 is unlocked**.

### Phase 2 dataset reconnaissance
- Inspected `SkinDiseaseBD A Dataset of Common Skin Disease Ima.zip` (978.7 MB): it contains 3 inner zips (`Images_176x176_v1.zip`, `Images_512x512_v2.zip`, `Raw_Images.zip`).
- `Images_512x512_v2.zip` (checked) = messy layout:
  - flat `aug_0..aug_4_*.jpeg` files (augmentation variants of the same photo)
  - `Updated Images/<class>/` folder + raw class folders
  - classes: **Dermatitis 302 · Eczema 381 · Scabies 301 · Tinea Ringworm 316 · Vitiligo 312** (1,612 usable)
- 5 SkinDiseaseBD classes **do not overlap** the starter's acne/normal/rosacea.

### Options offered for Phase 2
- **A — 8-class unified task** (matches the paper thesis): harmonize label map, dedupe cross-source, train one B0 on the fused set, compare via per-source F1.
- B — rosétta-mapping framing (weaker).
- C — 5-class SkinDiseaseBD-only baseline for a clean single-vs-single comparison first.

### User asked: "what's the plan after that? if i chose option A"
- Laid out the full roadmap: extract+curate → `data/phase2_fused/` (8 classes) → `harmonize.py` (label_map.csv) → **`dedupe.py` (phash, Hamming ≤ 5, BEFORE the split — the leakage story)** → `audit.py` → train → per-source F1 comparison vs Phase 1 → Phase 3 (add Fitzpatrick17k-black + SCIN, bias measurement by skin type, Grad-CAM) → Phase 4 (IEEE 6–8 pages, ICCIT/ICIEV/EWU).
- Flagged **open questions**: (1) SCIN is multi-label — collapse to dominant label or explicit handling? (2) where to train Phase 2 (home RX580 ~1–1.5 hr vs Colab/lab 5–10× faster).

---

## 6. Current status / pending decisions

| Item | State |
|---|---|
| Phase 1 baseline (`results/phase1_baseline.json`) | **DONE** — test_acc 0.9564 |
| Golden rule | **Met** (second dataset may now be opened) |
| Direction | Cross-dataset generalization (NOT fusion) — `DEEPSEEK_CONVERSATION_PLAN.md` |
| Harmonization | **DONE** — `results/label_map.csv` (142 rows, 0 missing) |
| Dedupe | **DONE** — 0 cross-dataset pairs; 865 DermNet / 157 SkinDiseaseBD internal |
| eval_cross.py + configs/phase2_*.yaml | **DONE** — CPU smoke-tested |
| DermNet validation split | **DONE** — 1,243 imgs (8%/class, seed 42) |
| gradcam_gallery.py | **DONE** — phase-1 test split browsed (367 imgs) |
| Phase 2 DermNet training | **PENDING** (RX580-DirectML or lab A4000) |
| eval_cross full run | **PENDING** |
| Phase 2 Kaggle/Colab notebooks | **PENDING** |
| AGENT.md / docs | Updated 2026-09-24 |

---

### Reference: phase1_baseline.json highlights
| key | value |
|---|---|
| run_id | phase1_baseline |
| model | efficientnet_b0 |
| profile | home_rx580_dml |
| device | privateuseone:0 |
| epochs | 15 |
| test_acc | 0.9564 |
| macro_f1 | 0.9485 |

---

## 7. Session 2026-09-24 — Phase 2 pipeline

**Direction confirmed:** the user wanted "a model that predicts many diseases, tested on another dataset". Fusion is impossible (zero shared labels). Plan: TRAIN on DermNet 23-class, TEST on 4-5 external sets.

**Completed:**
- Extracted SkinDiseaseBD test images → `data/SkinDiseaseBD/Updated Images` (5 classes, 1,612 imgs, 512×512) from **`Images_512x512_v2.zip\Updated Images\<class>\`**. NOTE (verified 2026-09-24): the `aug_N` filename prefix is the **class index** (aug_0=Dermatitis … aug_4=Vitiligo), NOT augmentation; `Raw_Images.zip` is flat `ResearchImage\*.jpg` (197 unlabeled files, no per-file labels) and was unusable; all 1,611/1,612 base IDs unique (only 2 exact-dups). —— Methods disclosure: "test set uses the 512×512 Updated split; `aug_N` = class index, not augmentation; Raw_Images.zip not used."
- `src/harmonize.py` → `results/label_map.csv` (142 rows across DermNet / SkinDiseaseBD / Fitzpatrick-black / phase-1). Unified classes: eczema, scabies, tinea, dermatitis, vitiligo, acne.
- `src/dedupe.py` (imagehash.phash, Hamming ≤ 5) → `results/dedupe_report_dermnet_sdb.json`: **0 cross-dataset pairs** = no leakage. 865 within-DermNet (benign, train-internal), 157 within-SkinDiseaseBD (test-internal incl. distance-0 exact dups from `aug_*` variants — a paper honesty note, does NOT inflate cross-split accuracy).
- `src/eval_cross.py` — one checkpoint, five test sets, dumps one JSON. **Locked design decision:** raw 23-class argmax then map predicted→unified via label_map.csv (NOT max-logit reduction, which is degenerate for single-unified-class test sets). Supports `folder_per_class` (with inline_map or label_map.csv) and `subfolder_binary` (DDI bias probe). CPU smoke test passed: skindiseasebd n=1612, skin_disease_images n=264 (99 acne + 165 rosacea), ddi_bias Black 2155/White 2000 (mean_conf + top5 distributions).
- `configs/phase2_dermnet.yaml` + `configs/phase2_eval.yaml` (5 test sets: dermnet_test 23-class, skindiseasebd 5-class, fitzpatrick_black 3-unified, skin_disease_images 2→acne via inline_map, ddi_bias subfolder_binary).
- `src/gradcam_gallery.py` — Grad-CAM inspection gallery: original thumb + CAM overlay (manual Grad-CAM on timm `conv_head`, jet colormap in numpy, percentile-clipped alpha so the hotspot shows), GT vs pred + confidence, per-image marking (✓/✗/? + note) saved to browser localStorage, export/import marks JSON, filters, `file:///` open-original. Phase-1 test split run: acc 0.9428 (346/367) using `_best.pt` (slightly below the JSON's 0.9564 because that used the last-epoch checkpoint).
- **Prepared DermNet for training:** fixed `phase2_dermnet.yaml` data_root → parent dir (trainer looks for train/ inside data_root); created `validation/` split (8%/class, seed 42, **1,243 imgs**). Verified trainer resolves train 14,314 / val 1,243 / test 4,002, 23 classes each, consistent class order.
- Committed + pushed `e3febe4` "Add Phase 2 cross-dataset eval pipeline" (7 files).
- Updated all Markdown docs (AGENT.md, README.md, SETUP_HOME.md, FILE_MAP.md, DEEPSEEK_CONVERSATION_PLAN.md, this file).

## 8. Session 2026-09-24 (evening) — Dedupe cleanup + switch training to Kaggle T4

**Prep on the DermNet split (before training):**
- Re-ran `dedupe.py` over `train;validation` → found **120 train↔validation near-dups (98 pixel-identical)**. Root cause: DermNet ships the *same photo in multiple class folders* (e.g. `Acne…/acne-cystic-2.jpg ≡ Eczema…/eczema-hand-16.jpg`), so the 8% holdout landed copies on both sides of the split.
- **Fixed:** removed the 124 validation-side copies (train kept intact) → re-dedupe: **741 DermNet-internal pairs, 0 train↔validation cross-split pairs = CLEAN**. Final split: **train 14,314 / validation 1,120 / test 4,002**.
- Verified SkinDiseaseBD label CSV + provenance disclosure (section 7).

**Training direction change (user decision):**
- Local RX580 epochs were ~18-19 min on 14,314 images (batch 16, DirectML, no AMP). Root cause investigation: `use_amp = amp && str(device).startswith("cuda")` (`train_resumable.py:399`) — AMP requires CUDA; DirectML device is `privateuseone:0`, so AMP cannot run on the home GPU. (My earlier "~2× with AMP" suggestion was wrong.)
- Full 15 epochs at home = ~4.5-5 hr. **User chose to switch to Kaggle T4** where `kaggle_t4` profile (cuda, batch 32, AMP on) gives ~1.5-4 min/epoch → full run ~30-90 min.
- Prepared split (19,436 imgs, 1.71 GB) zipped → uploaded to HF dataset `Nirob-jon/cse475-dermnet-split` (`DermNetPrepared.zip`, commit 850fcd6). Old local phase-2 checkpoints moved to `results/backup_phase2_local/` (NOT deleted).
- Wrote `notebooks/kaggle_phase2.ipynb` (8 cells: HF login → clone harness → snapshot_download zip→extract → pip → data sanity → TRAIN `--profile kaggle_t4 --hub hf --resume auto --data $DATA --epochs 15` → results note).

**What changed vs earlier numbers:** validation split is now 1,120 (dedupe-clean), not 1,243; the trainer reads split counts dynamically so everything still resolves. HF checkpoints repo has no phase2 files yet → Kaggle run starts from scratch.

**Pending (next session):**
1. User imports `notebooks/kaggle_phase2.ipynb` on Kaggle, adds `HF_TOKEN` secret, runs all (~30-90 min; resumable).
2. After training: `src/eval_cross.py --checkpoint results/phase2_dermnet_best.pt --config configs/phase2_eval.yaml` → results table (best.pt downloaded from HF or already pushed there).
3. Full phase-1 gallery (`results/gallery_phase1_all/gallery.html`, 2,439 imgs) — DONE at 5:47 PM; mark-audit of test/val wrongs optional (43 `normal→acne` incl. 16 test).
4. Phase-2 Colab/kaggle eval notebook + cloud dataset slugs (still TBD).