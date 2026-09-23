# SETUP_HOME.md — Home PC Setup (RX580, DirectML / CPU)

This file covers **only the home PC** (NirjonPC1, Ryzen 5 5600, RX580 8GB, Windows 11).
For lab / Kaggle / Colab setup, see `AGENT.md` Section 4.

Purpose: run `src/train_resumable.py` on the home PC when the lab A4000 is not available, and resume the same checkpoint later on any other machine.

---

## 1. Which Python to Use

**Never use `python` (3.14.5) — no PyTorch wheels exist for it.**

Use `py -3.11` (Python 3.11.9) for everything.

Verify:

```powershell
py -3.11 --version
# Python 3.11.9
```

---

## 2. Create the Home Virtual Environment (one time)

Keep the home venv **separate** from the lab venv, because the lab uses CUDA torch and the home uses DirectML/CPU torch. They are not interchangeable.

```powershell
cd F:\cse475_skin
py -3.11 -m venv .venv-home
.\.venv-home\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 3. Install DirectML PyTorch (recommended)

```powershell
pip install torch-directml
pip install timm torchvision tqdm pyyaml numpy pillow matplotlib scikit-learn imagehash
```

Notes:

- `torch-directml` **replaces** the normal `torch` package. Do NOT `pip install torch` in this venv.
- DirectML has **no AMP** (mixed precision). `profiles.yaml` already sets `amp: false` for `home_rx580_dml`.
- DirectML has no CUDA API; `torch.cuda.is_available()` will be False.

Verify DirectML:

```powershell
python -c "import torch, torch_directml; print('torch', torch.__version__); print('dml available:', torch_directml.is_available()); print('device:', torch_directml.device())"
```

---

## 4. Install CPU-Only PyTorch (bulletproof fallback)

If DirectML gives too many unsupported-op errors, use this venv instead:

```powershell
py -3.11 -m venv .venv-home-cpu
.\.venv-home-cpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm tqdm pyyaml numpy pillow matplotlib scikit-learn imagehash
```

CPU is ~10-20x slower than CUDA but always works.

---

## 5. Run Training on the Home PC

Activate the DirectML venv and run:

```powershell
cd F:\cse475_skin
.\.venv-home\Scripts\Activate.ps1
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images"
```

The script auto-detects the home PC and picks `home_rx580_dml`. To force CPU:

```powershell
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images" --profile home_rx580_cpu
```

---

## 6. Expected Timing (EfficientNet-B0, 224px, 15 epochs)

| Dataset size    | A4000 + AMP | RX580 DirectML   | RX580 CPU             |
| --------------- | ----------- | ---------------- | --------------------- |
| 2,439 (starter) | ~6 min      | ~1 hr            | ~3 hr                 |
| 5,000           | ~15 min     | ~2 hr            | ~6 hr                 |
| 10,000          | ~25 min     | ~4 hr (2 nights) | ~12 hr (overnight x2) |

---

## 7. DirectML Auto-Fallback to CPU

`train_resumable.py` wraps every forward/backward in a try/except. If DirectML raises "not implemented" or "could not run" for an operator, the script:

1. Prints `[WARN] DirectML op unsupported -> switching to CPU for rest of run`
2. Moves that batch to CPU
3. Continues training (does not crash)

If you see this warning constantly, switch to `--profile home_rx580_cpu` for the rest of the run.

---

## 8. Hugging Face Sync (optional)

One-time setup on the home PC:

```powershell
pip install huggingface_hub
huggingface-cli login
# paste token (write scope) from https://huggingface.co/settings/tokens
```

Then run with:

```powershell
py -3.11 src\train_resumable.py --config configs\baseline.yaml --data "F:/Downloads/skin_disease_images" --hub hf --hf-repo Nirob-jon/cse475-skin-checkpoints --resume auto
```

This will:

1. Auto-download `phase1_baseline_last.pt` from the HF repo (if present)
2. Resume from the last epoch
3. Upload the new `_last.pt` after each epoch

---

## 9. Do / Don''t on the Home PC

**DO**

- Use `py -3.11` inside `.venv-home`
- Use `--profile home_rx580_dml` or let auto-detect pick it
- Let training run in the background; checkpoint every epoch protects you
- Keep `--resume auto` on so a crash resumes cleanly

**DON''T**

- Don''t install normal `torch` into the DirectML venv
- Don''t enable AMP (`amp: false` is correct for DirectML)
- Don''t expect CUDA-style speedups; DirectML is ~4-6x slower than A4000
- Don''t mix the lab venv and home venv — they are separate

---

## 10. Troubleshooting

| Symptom                          | Fix                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `ImportError: torch_directml`  | `pip install torch-directml` inside `.venv-home`                         |
| `Could not run op ...` in loop | Auto-fallback handles it; if constant, switch to`--profile home_rx580_cpu` |
| GPU out of memory (8GB)          | Lower`batch_size` in `profiles.yaml` to 8                                |
| Extremely slow first epoch       | Normal — Windows paging, shader compilation. Epoch 2+ is faster             |
| `python` (3.14) launches       | You are not in the venv. Run`.venv-home\Scripts\Activate.ps1` first        |
