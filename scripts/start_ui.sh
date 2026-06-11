#!/usr/bin/env bash
# Start the TTS Production UI (kills any old instance on port 7861 first)
set -euo pipefail
cd "$(dirname "$0")/.."

if lsof -t -i:7861 >/dev/null 2>&1; then
  echo "Stopping existing UI on port 7861..."
  kill $(lsof -t -i:7861) 2>/dev/null || true
  sleep 1
fi

source .venv/bin/activate
python gradio_production_ui.py
