# Multi-Channel Setup Guide

Use channels to organize scripts and voices for different YouTube projects (or any narration workflow).

## Quick start

### 1. Add a voice

- Upload a 5–15s WAV in the Production UI, **or**
- Use **Import from YouTube** in Voice Management, **or**
- Place `voices/your-voice.wav` and edit `voices/presets.json`

Starter preset (update the path to your file):

```json
{
  "demo": {
    "prompt": "voices/your-voice.wav",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8
  }
}
```

### 2. Create a channel

```json
{
  "Demo": ["demo"]
}
```

Copy from `voices/channels.example.json` for multi-channel setups.

### 3. Add scripts

```
scripts/
  Demo/
    demo.csv
  MyChannel/
    episode-01.csv
```

### 4. Render

```bash
./.venv/bin/python tools/channel_manager.py render Demo \
  --script scripts/Demo/demo.csv
```

Output goes to `outputs/Demo/`.

---

## Folder structure

```
TTS/
├── voices/
│   ├── presets.json       # your voice presets
│   ├── channels.json      # channel → preset mapping
│   └── *.wav              # your samples (gitignored)
├── scripts/
│   └── Demo/
│       └── demo.csv
└── outputs/               # rendered audio (gitignored)
    └── Demo/
```

---

## Production UI

Open `http://127.0.0.1:7861` after running `bash scripts/start_ui.sh`.

| Tab | Purpose |
|-----|---------|
| Voice Management | Upload / YouTube import, create presets |
| Channel Management | Map channels to voices |
| Script Editor | Write and split narration scripts |
| Render & Output | Batch render, QC, concat |

---

## Tips

- One WAV can power multiple presets with different exaggeration/CFG/temperature.
- Use subfolders for series: `--subfolder Season01/Episode03`
- Re-run renders safely — existing clips are skipped unless you enable overwrite.
