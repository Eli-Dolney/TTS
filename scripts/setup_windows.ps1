Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Set-Location (Join-Path $PSScriptRoot '..')

py -3.12 -m venv .venv
if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }

if (-not (Test-Path .venv)) { throw 'Failed to create venv with Python 3.12/3.11. Install from python.org' }

& .\.venv\Scripts\python -m pip install -U pip setuptools wheel
& .\.venv\Scripts\python -m pip install -e .
& .\.venv\Scripts\python -m pip install gradio pyloudnorm

& .\.venv\Scripts\python - << 'PY'
import torch, sys
print(sys.version)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
PY

Write-Host "Done. Activate with: .\.venv\Scripts\activate"

