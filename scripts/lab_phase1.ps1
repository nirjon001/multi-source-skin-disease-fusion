<#
CSE475 Phase 1 - lab A4000 runbook (idempotent)

Run on the lab PC:
    pwsh scripts/lab_phase1.ps1
    pwsh scripts/lab_phase1.ps1 -DataRoot "D:/skin_disease_images"
    pwsh scripts/lab_phase1.ps1 -SkipSetup          # rerun after venv+deps already installed
    pwsh scripts/lab_phase1.ps1 -Epochs 15          # override epoch count

What it does:
  1. Verifies `py -3.11` and the dataset root
  2. Creates .venv + installs cu121 torch and requirements (skipped if present, or with -SkipSetup)
  3. Runs `train_resumable.py --hub hf --resume auto --hf-repo nirjon001/cse475-skin-checkpoints`
  4. Prints the test_acc from results/phase1_baseline.json

HF note: on the lab, authenticate once with `hf auth login` (or set HF_TOKEN env var).
#>
[CmdletBinding()]
param(
    [string]$DataRoot = "F:/Downloads/skin_disease_images",
    [string]$HfRepo  = "nirjon001/cse475-skin-checkpoints",
    [int]$Epochs     = 15,
    [switch]$SkipSetup,
    [switch]$ResetCheckpoint
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

function Write-Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

# 1. Python sanity
Write-Step "Python check"
$py = (Get-Command "py" -ErrorAction SilentlyContinue) -or (Get-Command "py.exe" -ErrorAction SilentlyContinue)
if (-not (Test-Path "$env:LOCALAPPDATA\Programs\Python" -ErrorAction SilentlyContinue) -and -not $py) {
    Write-Host "py launcher not found - install Python 3.11 first." -ForegroundColor Yellow
}
& py -3.11 --version

# 2. Dataset check
Write-Step "Dataset check"
if (-not (Test-Path "$DataRoot/train")) {
    Write-Warning "NOT FOUND: $DataRoot/train"
    Write-Host "Expected train/validation/test subfolders. Use -DataRoot <path> to override."
    exit 1
}
Write-Host "dataset OK: $DataRoot"

# 3. Venv + deps (idempotent)
if (-not $SkipSetup) {
    Write-Step "Venv + deps"
    if (-not (Test-Path ".venv/Scripts/python.exe")) {
        & py -3.11 -m venv .venv
    }
    $venvPy = ".venv/Scripts/python.exe"
    & $venvPy -m pip install --upgrade pip --quiet
    if (-not (& $venvPy -c "import torch" 2>$null)) {
        & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    }
    & $venvPy -m pip install -r requirements.txt --quiet
    & $venvPy -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

# 4. Run training
Write-Step "Training (phase1_baseline, $Epochs epochs)"
$venvPy = ".venv/Scripts/python.exe"
if ($ResetCheckpoint) {
    Remove-Item "results/phase1_baseline_last.pt", "results/phase1_baseline_best.pt" -ErrorAction SilentlyContinue
}
& $venvPy src/train_resumable.py --config configs/baseline.yaml --data $DataRoot --hub hf --resume auto --hf-repo $HfRepo --epochs $Epochs
if ($LASTEXITCODE -ne 0) { Write-Error "training failed"; exit $LASTEXITCODE }

# 5. Report
Write-Step "Result"
if (Test-Path "results/phase1_baseline.json") {
    $r = Get-Content "results/phase1_baseline.json" -Raw | ConvertFrom-Json
    Write-Host "test_acc   = $($r.test_acc)"
    Write-Host "macro_f1   = $($r.macro_f1)"
    Write-Host "best val   = $((($r.history | Select-Object -Last 1).val_acc))"
    Write-Host "GOLDEN RULE: $(if ($null -ne $r.test_acc) { 'SATISFIED - Phase 2 can begin' } else { 'NOT satisfied yet' })" -ForegroundColor Green
} else {
    Write-Warning "results/phase1_baseline.json not found - something went wrong"
}