# Getting Started — Mac & Windows

This guide covers how to **download, install, and run** the TTS Production Manager on macOS and Windows.

The app uses [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) under the hood. On first run it downloads ~3 GB of model weights (cached locally after that).

---

## Requirements

| | macOS | Windows |
|---|--------|---------|
| **OS** | macOS 12+ (Apple Silicon recommended) | Windows 10/11 |
| **Python** | 3.11 or 3.12 | 3.11 or 3.12 |
| **GPU** | Apple Silicon MPS (optional; CPU works) | NVIDIA GPU + CUDA (recommended) |
| **RAM** | 16 GB+ recommended | 16 GB+ recommended |
| **Disk** | ~5 GB free (venv + model cache) | ~5 GB free |

You also need **Git** to clone the repo (or download a ZIP from GitHub).

---

## 1. Download the project

### Option A — Git clone (recommended)

```bash
git clone <your-repo-url> TTS-MAc
cd TTS-MAc/TTS
```

Replace `<your-repo-url>` with your fork or this repository’s URL.

### Option B — ZIP download

1. Open the repo on GitHub → **Code** → **Download ZIP**
2. Unzip and open a terminal in the `TTS` folder inside the archive

All commands below assume your working directory is the **`TTS`** folder (where `pyproject.toml` and `gradio_production_ui.py` live).

---

## 2. Install — macOS

### Quick install (recommended)

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
```

The script creates `.venv`, installs the project, Gradio, and pyloudnorm, then prints whether MPS is available.

### Manual install

```bash
# Install Python if needed (Homebrew)
brew install python@3.12

cd /path/to/TTS-MAc/TTS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[production]"
```

Verify Apple Silicon acceleration:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## 3. Install — Windows

### Quick install (recommended)

Open **PowerShell** in the `TTS` folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

### Manual install

```powershell
cd D:\path\to\TTS-MAc\TTS
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install -e ".[production]"
```

Verify CUDA (if you have an NVIDIA GPU):

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> **Note:** PyTorch wheels bundled with this project include CUDA runtime libraries. You mainly need up-to-date NVIDIA drivers — a separate CUDA toolkit install is usually not required.

---

## 4. Run the Production UI

This is the main interface for scripts, voices, channels, rendering, and re-rendering individual lines.

### macOS / Linux

```bash
cd /path/to/TTS-MAc/TTS
source .venv/bin/activate
python gradio_production_ui.py
```

### Windows

```powershell
cd D:\path\to\TTS-MAc\TTS
.\.venv\Scripts\Activate.ps1
python gradio_production_ui.py
```

Open in your browser:

**http://localhost:7861**

The first launch downloads model weights and may take several minutes. Later launches are much faster.

### What you can do in the UI

| Tab | Purpose |
|-----|---------|
| **Voice Management** | Upload voice samples, import from YouTube, create presets, test generation |
| **Channel Management** | Group voices by YouTube channel / project |
| **Script Editor** | Paste full scripts (auto-split), edit lines, set emphasis per line |
| **Render & Output** | Render full scripts, re-render selected lines, concatenate to `combined.wav` |

---

## 5. Quick CLI workflow (no UI)

Useful for automation or batch jobs.

### Test that voices work

```bash
# macOS/Linux
./.venv/bin/python tools/test_channels.py

# Windows
.\.venv\Scripts\python.exe tools\test_channels.py
```

### Render a script

```bash
# macOS/Linux
./.venv/bin/python tools/lesson.py \
  --script scripts/Demo/demo.csv \
  --channel Demo

# Windows
.\.venv\Scripts\python.exe tools\lesson.py `
  --script scripts\Demo\demo.csv `
  --channel Demo
```

Output lands in `outputs/<Channel>/<subfolder>/` with a `combined.wav` when concatenation runs.

### Single test line

```bash
python example_tts.py \
  --prompt voices/YourVoice.wav \
  --script scripts/example.txt \
  --output-dir outputs/test \
  --overwrite
```

On Mac with Apple Silicon, the UI auto-selects **MPS**. On Windows with NVIDIA, it uses **CUDA**. Otherwise it falls back to **CPU** (slower).

---

## 6. Voice & script setup (first time)

1. **Add a voice sample** — 5–15 seconds, clean WAV, no background music  
   Place it in `voices/` (e.g. `voices/my-narrator.wav`)

2. **Create a preset** — in the UI (Voice Management) or edit `voices/presets.json`:

```json
{
  "my-narrator": {
    "prompt": "voices/my-narrator.wav",
    "exaggeration": 0.45,
    "cfg_weight": 0.55,
    "temperature": 0.75
  }
}
```

3. **Create a channel** — map a project name to preset(s) in `voices/channels.json`:

```json
{
  "MyChannel": ["my-narrator"]
}
```

4. **Write or paste a script** — CSV format (one row per clip):

```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Hello and welcome.,001-intro,my-narrator,0.45,0.55,0.75
```

See `CHANNEL_SETUP_GUIDE.md` and `QUICK_REFERENCE.md` for more detail.

---

## 7. Recommended settings

**General narration**

- Exaggeration: `0.45`–`0.55`
- CFG weight: `0.50`–`0.65`
- Temperature: `0.70`–`0.80`

**More dramatic / emphasis**

- Exaggeration: `0.65`–`0.85`
- CFG weight: `0.40`–`0.50` (lower = slower pacing)

**Mac memory (MPS OOM)**

- Split long paragraphs into smaller clips (~320 characters max)
- Use **Sentence** or **Character limit** split in the Script Editor
- Close other heavy apps while rendering

---

## 8. Troubleshooting

### `ModuleNotFoundError: chatterbox`

The virtual environment is missing or broken. Recreate it:

```bash
# macOS
bash scripts/setup_mac.sh

# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

### `PerthImplicitWatermarker` / watermarker is `None`

Install a compatible setuptools version:

```bash
pip install "setuptools<81"
```

### `No module named gradio`

```bash
pip install -e ".[production]"
```

### MPS out-of-memory on Mac

- Shorten script lines (use auto-split with ~320 char limit)
- Render in smaller batches
- Restart the UI to free GPU memory

### Broken venv pointing at old path

If you moved the project folder, delete `.venv` and run the setup script again.

### UI won’t open / port in use

Change the port in `gradio_production_ui.py` (bottom of file) or stop whatever is using port `7861`.

### Windows: script execution disabled

Run PowerShell as Administrator once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 9. Project layout

```
TTS/
├── gradio_production_ui.py   # Main production UI (port 7861)
├── example_tts.py            # CLI batch renderer
├── voices/
│   ├── presets.json          # Voice presets
│   ├── channels.json         # Channel → voice mapping
│   └── *.wav                 # Voice reference clips
├── scripts/                  # CSV/TXT scripts per channel
├── outputs/                  # Rendered WAV files
└── tools/
    ├── lesson.py             # Full render + concat pipeline
    ├── channel_manager.py    # Channel CLI
    └── script_render.py      # Selective re-render helpers
```

---

## 10. More documentation

| File | Contents |
|------|----------|
| `QUICK_REFERENCE.md` | Common commands cheat sheet |
| `CHANNEL_SETUP_GUIDE.md` | Multi-channel production setup |
| `LESSON_PIPELINE_GUIDE.md` | End-to-end lesson workflow |
| `voices/README.md` | Voice samples and preset setup |
| `README.md` | Upstream Chatterbox TTS reference |

---

## 11. Updating

```bash
git pull
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e ".[production]"
```

If dependencies changed significantly, recreate the venv with the setup script.
