# Quick Reference

## Install & run

```bash
bash scripts/setup_mac.sh          # macOS
source .venv/bin/activate
bash scripts/start_ui.sh           # Production UI at http://127.0.0.1:7861
```

## First-time setup

1. Add a voice WAV to `voices/` (or use **Import from YouTube** in the UI)
2. Create a preset in **Voice Management** or edit `voices/presets.json`
3. Add a channel in **Channel Management** or edit `voices/channels.json`
4. Write a script CSV under `scripts/YourChannel/`

See `voices/README.md` and `scripts/README.md` for details.

## Common commands

```bash
# List channels / voices
./.venv/bin/python tools/channel_manager.py list channels
./.venv/bin/python tools/channel_manager.py list voices

# Render demo script (after adding voices/your-voice.wav)
./.venv/bin/python tools/channel_manager.py render Demo \
  --script scripts/Demo/demo.csv

# Full lesson pipeline (render + concat + metadata)
./.venv/bin/python tools/lesson.py \
  --script scripts/Demo/demo.csv \
  --channel Demo

# Concatenate clips
./.venv/bin/python tools/channel_manager.py concat Demo
```

## Clone from YouTube

**UI:** Voice Management → Import from YouTube

**CLI:**
```bash
./.venv/bin/python -c "
from pathlib import Path
from tools.youtube_voice import download_voice_sample
path, msg = download_voice_sample(
    'https://www.youtube.com/watch?v=VIDEO_ID',
    Path('voices'), 'my-voice', start='1:30', duration=15
)
print(path, msg)
"
```

## Render quality controls

QC and auto-retry are on by default:

```bash
./.venv/bin/python tools/channel_manager.py render Demo \
  --script scripts/Demo/demo.csv --no-validate

./.venv/bin/python tools/channel_manager.py render Demo \
  --script scripts/Demo/demo.csv --max-retries 3
```

## Script CSV format

```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Hello world.,001-intro,demo,0.5,0.5,0.8
```

## Docs

- `GETTING_STARTED.md` — install on Mac & Windows
- `CHANNEL_SETUP_GUIDE.md` — multi-channel workflow
- `LESSON_PIPELINE_GUIDE.md` — automated render + concat
- `UPGRADE_SUMMARY.md` — feature overview
