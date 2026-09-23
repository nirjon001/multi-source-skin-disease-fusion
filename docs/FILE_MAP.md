# FILE_MAP.md — what to run, what to read, what to ignore

The only file that **trains** anything is `src/train_resumable.py`. Everything else is a helper, a setting, or a lesson.

## Run these

| File | When | Command |
|---|---|---|
| `src/audit.py` | Once per dataset, before training | `py -3.11 src\audit.py --data "<root>" --check-corrupt --out results\audit_starter.json` |
| `src/train_resumable.py` | THE trainer (any machine) | `py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "<root>"` |
| `scripts/lab_phase1.ps1` | One-shot runbook on the lab A4000 | `pwsh scripts\lab_phase1.ps1` |
| `notebooks/kaggle_phase1.ipynb` | Same trainer, running ON Kaggle | import into Kaggle, add `HF_TOKEN`, run all |
| `notebooks/colab_phase1.ipynb` | Same trainer, running ON Colab | import into Colab, add `HF_TOKEN`, run all |

## Read these

| File | Why |
|---|---|
| `notebooks/tutorial_phase1.ipynb` | Textbook — explains every file in plain English. Not a tool. |
| `README.md` / `AGENT.md` | Quick start + project memory (rules, session log). |
| `configs/baseline.yaml` | Infrequently edited settings (epochs, lr, seed…). |
| `configs/profiles.yaml` | Per-machine settings (batch size, AMP, checkpoint dir). |

## Ignore (generated / never committed)

| Path | What |
|---|---|
| `data/`, `results/`, `logs/` | Working files, gitignored |
| `*.pt`, `*.pth`, `*.ckpt` | Checkpoints |
| `*.json` under `results/` | Run scores |
| `.venv*` | Local virtualenvs |

## Removed

- `src/train.py` — old first-draft trainer, deleted (2026-09-23). It's the historical mention you'll sometimes see in AGENT.md session logs; translate it mentally to `train_resumable.py`.