# Feature Overview

Chatterbox TTS extended with a production UI and automation tools for local voice cloning and YouTube-style narration workflows.

## Core features

### Production UI (`gradio_production_ui.py`)
- Voice Management — upload WAVs, **Import from YouTube**, create presets
- Channel Management — organize projects and voice assignments
- Script Editor — paste, auto-split, edit CSV scripts
- Render & Output — batch render, QC + auto-retry, ETA, concat

### Voice cloning
- Zero-shot cloning via short reference WAV (Chatterbox)
- YouTube URL import (`tools/youtube_voice.py`) — yt-dlp + ffmpeg
- Presets with per-voice exaggeration / CFG / temperature

### Render pipeline
- Shared renderer (`tools/script_render.py`)
- Audio QC (`tools/audio_qc.py`) — duration, clipping, silence checks
- Auto-retry with seed re-roll on bad clips
- Progress + ETA in UI and CLI

### Automation
- `tools/channel_manager.py` — list, render, concat per channel
- `tools/lesson.py` — one-command render → concat → metadata
- `example_tts.py` — CLI batch TTS

## Quick start

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
bash scripts/start_ui.sh
```

1. Add `voices/your-voice.wav` or import from YouTube
2. Update `voices/presets.json` (starter `demo` preset included)
3. Render `scripts/Demo/demo.csv` from the UI or CLI

## Docs

| File | Contents |
|------|----------|
| `GETTING_STARTED.md` | Install Mac/Windows |
| `QUICK_REFERENCE.md` | Common commands |
| `CHANNEL_SETUP_GUIDE.md` | Multi-channel workflow |
| `LESSON_PIPELINE_GUIDE.md` | Automated pipeline |
| `voices/README.md` | Voice samples & presets |
