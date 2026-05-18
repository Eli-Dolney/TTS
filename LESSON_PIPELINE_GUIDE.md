# Automated Lesson Pipeline Guide

## 🎯 Overview

The `tools/lesson.py` tool automates your entire lesson workflow in **one command**:
1. ✅ Renders all clips from CSV script
2. ✅ Organizes files into proper folder structure
3. ✅ Concatenates clips with gaps
4. ✅ Generates metadata JSON

**Before**: 4 separate commands, manual file moving  
**After**: 1 command, fully automated

---

## 🚀 Quick Start

### Basic Usage

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --channel WiredWorkshop
```

That's it! The tool will:
- Auto-detect subfolder from script path (`CCNA/NetworkTopologies01`)
- Render all clips to `outputs/WiredWorkshop/CCNA/NetworkTopologies01/`
- Concatenate into `combined.wav`
- Generate `metadata.json`

### With Explicit Subfolder

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --channel WiredWorkshop \
  --subfolder CCNA/NetworkTopologies01
```

### Resume After Interruption

If rendering is interrupted, just run the same command again. It will:
- Skip already-rendered clips (existing files are preserved)
- Only render missing clips
- Re-concatenate everything

```bash
# First run (gets interrupted after 10 clips)
./.venv/bin/python tools/lesson.py --script script.csv --channel WiredWorkshop

# Resume (only renders remaining 9 clips)
./.venv/bin/python tools/lesson.py --script script.csv --channel WiredWorkshop
```

---

## 📋 Command Options

### Required
- `--script` - Path to CSV script file

### Optional
- `--channel` - Channel name (auto-detected from script path if not provided)
- `--subfolder` - Subfolder path (auto-detected from script path if not provided)
- `--voice` - Voice preset name (default: channel's first voice)
- `--device` - Device: `auto`, `mps`, `cuda`, `cpu` (default: `auto`)
- `--gap-seconds` - Gap between clips in seconds (default: `0.5`)
- `--lufs-target` - LUFS normalization target (default: `-16.0`)
- `--to-48k` / `--no-48k` - Resample to 48kHz (default: `True`)
- `--overwrite` - Overwrite existing files (default: skip existing)

### Advanced
- `--skip-render` - Skip rendering, only concatenate existing files
- `--skip-concat` - Skip concatenation, only render
- `--presets-file` - Path to presets.json (default: `voices/presets.json`)
- `--channels-file` - Path to channels.json (default: `voices/channels.json`)

---

## 📁 Output Structure

After running, you'll get:

```
outputs/WiredWorkshop/CCNA/NetworkTopologies01/
├── 001-intro.wav
├── 002-main-content.wav
├── 003-outro.wav
├── ... (all individual clips)
├── combined.wav          # Final concatenated audio
└── metadata.json         # Lesson metadata
```

---

## 💡 Common Use Cases

### 1. Process a New Lesson

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/CiscoIOS01.csv \
  --channel WiredWorkshop
```

### 2. Re-render Everything (Overwrite)

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --channel WiredWorkshop \
  --overwrite
```

### 3. Only Concatenate (Skip Rendering)

Useful if you've already rendered clips and just want to re-concatenate:

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --channel WiredWorkshop \
  --skip-render
```

### 4. Only Render (Skip Concatenation)

Useful if you want to render now, concatenate later:

```bash
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --channel WiredWorkshop \
  --skip-concat
```

### 5. Batch Process Multiple Lessons

```bash
# Process all CCNA lessons
for script in scripts/WiredWorkshop/CCNA/*.csv; do
  echo "Processing: $script"
  ./.venv/bin/python tools/lesson.py \
    --script "$script" \
    --channel WiredWorkshop \
    --device mps
done
```

---

## 🔄 Resume Functionality

The tool automatically supports resuming:

1. **First run**: Renders all clips
2. **Interrupted**: Some clips are rendered, some are not
3. **Resume**: Run the same command again
   - Existing clips are **skipped** (not re-rendered)
   - Missing clips are **rendered**
   - Everything is **re-concatenated**

**No special flags needed** - just run the same command again!

---

## 📊 Progress Tracking

The tool shows:
- ✅ Which clips are being rendered
- ✅ Which clips are skipped (already exist)
- ✅ Concatenation progress
- ✅ Final summary with file sizes

Example output:
```
============================================================
🎬 RENDERING LESSON
============================================================
Script: scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv
Channel: WiredWorkshop
Output: outputs/WiredWorkshop/CCNA/NetworkTopologies01
============================================================

Saved: outputs/.../001-intro.wav
Skip (exists): outputs/.../002-main.wav
Saved: outputs/.../003-outro.wav
...

============================================================
🔗 CONCATENATING AUDIO
============================================================
Combined 19 files -> combined.wav

============================================================
✅ LESSON COMPLETE
============================================================
Combined Audio: combined.wav (21.73 MB)
============================================================
```

---

## 🎯 Auto-Detection

The tool automatically detects:

### Channel Name
From script path: `scripts/WiredWorkshop/...` → `WiredWorkshop`

### Subfolder
From script path: `scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv` → `CCNA/NetworkTopologies01`

You can override either with explicit flags if needed.

---

## ⚡ Performance Tips

1. **Use `--device mps`** on Mac for faster rendering
2. **Use `--device cuda`** on Windows/Linux with GPU
3. **Resume is automatic** - don't worry about interruptions
4. **Batch processing** - Process multiple lessons in a loop

---

## 🐛 Troubleshooting

### Error: "Could not detect channel"
**Solution**: Specify `--channel` explicitly

### Error: "Script file not found"
**Solution**: Check the script path is correct

### Files in wrong location
**Solution**: The tool auto-organizes files. If issues persist, check `--subfolder` path

### Want to re-render everything
**Solution**: Use `--overwrite` flag

---

## 📈 What's Next?

After using the lesson pipeline, you can:
1. ✅ Use the `combined.wav` file in your video editor
2. ✅ Check `metadata.json` for lesson information
3. ✅ Process the next lesson with the same command

---

## 🎬 Example: Complete Workflow

```bash
# 1. Process lesson
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --channel WiredWorkshop

# 2. Check output
ls -lh outputs/WiredWorkshop/CCNA/NetworkTopologies01/combined.wav

# 3. Process next lesson
./.venv/bin/python tools/lesson.py \
  --script scripts/WiredWorkshop/CCNA/NetworkTypes01.csv \
  --channel WiredWorkshop
```

**That's it!** No manual file moving, no separate concatenation step. Everything is automated. 🚀

