# Automated Lesson Pipeline

`tools/lesson.py` runs the full workflow in one command:

1. Render all clips from a CSV script
2. Concatenate with gaps
3. Write `metadata.json`

## Quick start

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/Demo/demo.csv \
  --channel Demo
```

Output: `outputs/Demo/combined.wav` plus per-line clips.

## Options

```bash
# Custom subfolder
./.venv/bin/python tools/lesson.py \
  --script scripts/MyChannel/ep01.csv \
  --channel MyChannel \
  --subfolder Season01/Ep01

# Skip render (concat only)
./.venv/bin/python tools/lesson.py \
  --script scripts/Demo/demo.csv \
  --channel Demo \
  --skip-render

# Overwrite existing clips
./.venv/bin/python tools/lesson.py \
  --script scripts/Demo/demo.csv \
  --channel Demo \
  --overwrite

# Disable QC / change retries
./.venv/bin/python tools/lesson.py \
  --script scripts/Demo/demo.csv \
  --channel Demo \
  --no-validate --max-retries 0
```

## Resume

Re-run the same command after an interruption. Existing clips are skipped; only missing lines are rendered, then everything is concatenated again.

## Output layout

```
outputs/Demo/
  001-intro.wav
  002-setup.wav
  003-youtube.wav
  combined.wav
  metadata.json
```
