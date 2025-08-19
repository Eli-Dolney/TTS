### LTW: Local TTS Workflow (RTX 4080 + Mac M4)

This fork streamlines Chatterbox TTS for two daily-driver setups:
- RTX 4080 (Windows/Linux, CUDA)
- MacBook Air/Pro (M3/M4, Apple Silicon MPS)

Use this doc as a fast path to reliable installs, quick runs, and optimal settings for long-form and shorts video creation with your own voice prompts.

---

## 1) Environments

### macOS (Apple Silicon)
- Python: 3.11 or 3.12
- Install once:
```bash
brew install python@3.12
cd /Users/<you>/path/to/TTS
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```
- Verify MPS:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
Quick bootstrap:
```bash
bash scripts/setup_mac.sh
```

### Windows/Linux (RTX 4080)
- Python: 3.11 or 3.12
- NVIDIA driver + CUDA toolkit matching PyTorch build (PyTorch wheels already include CUDA runtime).
```powershell
cd D:\path\to\TTS
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```
Quick bootstrap (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Notes:
- The model will download ~3.2 GB of weights on first run and cache them.
- On macOS, first run can be slower due to JIT/warmup.

---

## 2) Quick runs

### Single line (ad‑hoc)
```bash
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows
python mac_tts_sample.py --text "Hello from my setup" --prompt prompt.wav --out outputs/hello.wav --device mps  # or cuda/cpu
```

### Batch (script file)
Option A: Plain text file `script file.txt` (one line per utterance). Then run:
```bash
python example_tts.py --prompt prompt.wav --script "script file.txt" --output-dir outputs --device mps --overwrite
# On 4080, use --device cuda
```

Option B: CSV/TSV with per-line voices and params. Example `scripts/my_lines.csv`:
```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Hey! Welcome back — let's kick this off with a smile.,eli_funny_001,eli-funny,,,
Quick question: are you ready to level up today?,eli_question_001,eli-question,,,
Let's get to work. Clear, steady, and confident.,eli_neutral_001,eli-neutral,,,
```
Run with presets and post-processing:
```bash
python example_tts.py \
  --script scripts/my_lines.csv \
  --presets-file voices/presets.json \
  --voice eli-neutral \
  --to-48k --lufs-target -16 \
  --output-dir outputs --device mps --overwrite
```

---

## 3) Recommended settings

- **exaggeration**: 0.5 neutral; raise to 0.7–1.0 for more energy. Extreme values may speed up speech.
- **cfg_weight**: 0.3–0.6 controls pacing/stability. Lower values slow speech and stabilize when exaggeration is high.
- **temperature**: 0.6–0.9 for most use cases.

Examples:
```bash
python example_tts.py --prompt prompt.wav --script "script file.txt" \
  --exaggeration 0.7 --cfg-weight 0.35 --temperature 0.8 \
  --device cuda --output-dir outputs --overwrite
```

Presets override defaults when provided; empty fields in CSV fall back to preset values, then CLI defaults.

---

## 4) Voice cloning tips

- Prompt length: 5–15 seconds, clean, dry voice. Avoid reverb, music, noise.
- Match target style: If you want calm delivery, use a calm prompt; for energetic delivery, use an energetic prompt.
- For consistent multi-clip sessions, keep the same prompt and parameters across takes.
- If pace drifts fast with high exaggeration, reduce `cfg_weight` (e.g., 0.25–0.35).

---

## 5) Performance notes

### RTX 4080 (CUDA)
- Use `--device cuda`.
- Batch multiple lines for better amortization (script mode).
- Close heavy GPU apps (browsers with WebGPU, stable diffusion) during long runs.

### Mac M4 (MPS)
- Use `--device mps`.
- First pass is slower due to warm-up; subsequent runs improve.
- Keep other Rosetta-heavy apps closed. Prefer ARM-native Python and wheels.

---

## 6) Long-form and shorts workflow

1. Write content in `script file.txt` (short, natural lines for best alignment).
2. Record `prompt.wav` in the target voice (clean 5–15s).
3. Generate:
```bash
python example_tts.py --prompt prompt.wav --script "script file.txt" --output-dir outputs --device mps --overwrite
```
4. Import WAVs into your editor (Premiere/Resolve/CapCut) for timing and mixing.
5. Add BGM/SFX; export.

For quick auditions of a single line, use `mac_tts_sample.py`.

---

## 7) Gradio UI (interactive)

```bash
python gradio_tts_app.py
# or
python gradio_vc_app.py
```
These auto-select `cuda` on 4080, `mps` on Apple Silicon, else fallback to `cpu`.

If your shell says "No module named gradio":
```bash
source .venv/bin/activate
pip install gradio
```

---

## 8) Troubleshooting

- Python 3.9 TypeError in examples: use Python 3.11/3.12 (union types).
- MPS reported False on macOS: ensure you installed ARM-native Python, and use PyTorch arm64 wheels.
- First run downloads are slow: they cache; subsequent runs are fast.
- Memory errors on CUDA: close other GPU apps, lower concurrent jobs, or run lines sequentially.

Security & privacy checklist:
- Do not commit audio: `outputs/`, `*.wav`, and `voices/**/*.wav` are ignored by `.gitignore`.
- Keep `.env` or private keys out of the repo (ignored by default).
- If sharing a repo publicly, scrub `voices/presets.json` paths and remove any private file names.
- When filing issues or sharing logs, redact local paths and personal info.

---

## 9) License & watermark

- MIT licensed (see `LICENSE`).
- Outputs include imperceptible Perth watermarks for responsible AI usage.


