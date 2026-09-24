# FILE_MAP.md — what to run, what to read, what to ignore

The only file that **trains** anything is `src/train_resumable.py`. Everything else is a helper, a setting, or a lesson.

## Run these

| File | When | Command |
|---|---|---|
| `src/audit.py` | Once per dataset, before training | `.\.venv-home-cpu\Scripts\python.exe src\audit.py --data "<root>" --check-corrupt --out results\audit_starter.json` |
| `src/train_resumable.py` | THE trainer (any machine) | **Kaggle:** `notebooks/kaggle_phase2.ipynb` · **Home fallback:** `.\.venv-home\Scripts\python.exe src\train_resumable.py --config configs\phase2_dermnet.yaml --profile home_rx580_dml --hub hf --resume auto` |
| `src/harmonize.py` | Build the cross-dataset label map (Phase 2) | `.\.venv-home-cpu\Scripts\python.exe src\harmonize.py --dermnet-root ".../Kaggle-skin-disease-different-catergory dataset" --skindiseasebd-root ".../SkinDiseaseBD/Updated Images" --fitzpatrick-root ".../fitzpatrick-black-images" --out results\label_map.csv` |
| `src/dedupe.py` | Leakage guard — near-dup report BEFORE any cross-eval (dry-run, no deletes) | `.\.venv-home-cpu\Scripts\python.exe src\dedupe.py --roots "<root1>;<root2>" --out results\dedupe_report.json` |
| `src/eval_cross.py` | Evaluate ONE checkpoint on N test sets → results table | `.\.venv-home\Scripts\python.exe src\eval_cross.py --checkpoint results\phase2_dermnet_best.pt --config configs\phase2_eval.yaml` |
| `src/gradcam_gallery.py` | Grad-CAM inspection gallery (mark good/bad-label images) | `.\.venv-home\Scripts\python.exe src\gradcam_gallery.py --checkpoint results\phase1_baseline_best.pt --data "F:/Downloads/skin_disease_images" --splits train,validation,test --out results\gallery_phase1_all` |
| `scripts/lab_phase1.ps1` | One-shot runbook on the lab A4000 | `pwsh scripts\lab_phase1.ps1` |
| `notebooks/kaggle_phase1.ipynb` | Same trainer, running ON Kaggle | import into Kaggle, add `HF_TOKEN`, run all |
| `notebooks/kaggle_phase2.ipynb` | **Phase 2** DermNet 23-class on Kaggle T4 (batch 32, AMP on, HF-synced) | import into Kaggle, add `HF_TOKEN` secret, run all |
| `notebooks/colab_phase1.ipynb` | Same trainer, running ON Colab | import into Colab, add `HF_TOKEN`, run all |

## Read these

| File | Why |
|---|---|
| `notebooks/tutorial_phase1.ipynb` | Textbook — explains every file in plain English. Not a tool. |
| `README.md` / `AGENT.md` | Quick start + project memory (rules, session log). |
| `configs/baseline.yaml` | Phase 1 record — **DO NOT EDIT**. |
| `configs/phase2_dermnet.yaml` | Phase 2 DermNet training settings (data_root = parent of train/val/test). |
| `configs/phase2_eval.yaml` | Phase 2 test-set list + class mappings. |
| `configs/profiles.yaml` | Per-machine settings (batch size, AMP, checkpoint dir). |
| `docs/DEEPSEEK_CONVERSATION_PLAN.md` | The design doc / handoff — why cross-dataset generalization, not fusion. |

## Ignore (generated / never committed)

| Path | What |
|---|---|
| `data/`, `results/`, `logs/` | Working files, gitignored |
| `*.pt`, `*.pth`, `*.ckpt` | Checkpoints |
| `*.json` under `results/` | Run scores |
| `.venv*` | Local virtualenvs |

## Removed

- `src/train.py` — old first-draft trainer, deleted (2026-09-23). It's the historical mention you'll sometimes see in AGENT.md session logs; translate it mentally to `train_resumable.py`.