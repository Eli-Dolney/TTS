#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYBIN=$(command -v python3.12 || command -v python3.11 || true)
if [[ -z "${PYBIN}" ]]; then
  echo "Python 3.11+ not found. Install with: brew install python@3.12" >&2
  exit 1
fi

rm -rf .venv
"${PYBIN}" -m venv .venv
./.venv/bin/python -m pip install -U pip setuptools wheel
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m pip install gradio pyloudnorm

./.venv/bin/python - <<'PY'
import torch, sys
print(sys.version)
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
PY

echo "Done. Activate with: source .venv/bin/activate"

