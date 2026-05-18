# Multi-Channel Setup Guide

## 🎯 Quick Start

### 1. Test All Channels & Voices
Verify everything works before production:

```bash
# Test all channels and voices
./.venv/bin/python tools/test_channels.py

# Test a specific channel
./.venv/bin/python tools/test_channels.py --channel WiredWorkshop
```

### 2. List Your Channels & Voices
```bash
# List all channels
./.venv/bin/python tools/channel_manager.py list channels

# List all voice presets
./.venv/bin/python tools/channel_manager.py list voices
```

### 3. Render Audio for a Channel
```bash
# Basic render (uses channel's default voice)
./.venv/bin/python tools/channel_manager.py render WiredWorkshop \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv

# With subfolder organization
./.venv/bin/python tools/channel_manager.py render WiredWorkshop \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --subfolder CCNA/NetworkTopologies01

# Specify a different voice
./.venv/bin/python tools/channel_manager.py render TinyTalesTV \
  --script scripts/TinyTalesTV/BirthdayParty.csv \
  --voice tinytales-female2
```

### 4. Concatenate Audio
```bash
# Concatenate all clips in a channel folder
./.venv/bin/python tools/channel_manager.py concat WiredWorkshop \
  --subfolder CCNA/NetworkTopologies01
```

---

## 📁 Folder Structure

```
TTS/
├── voices/
│   ├── channels.json          # Maps channels to voice presets
│   ├── presets.json           # Voice preset configurations
│   └── *.wav                  # Voice prompt files
├── scripts/
│   ├── WiredWorkshop/         # Scripts for WiredWorkshop channel
│   ├── TinyTalesTV/           # Scripts for TinyTalesTV channel
│   ├── WiredToWork/           # Scripts for WiredToWork channel
│   └── [Other Channels]/      # Scripts for other channels
└── outputs/
    ├── WiredWorkshop/         # Rendered audio for WiredWorkshop
    ├── TinyTalesTV/           # Rendered audio for TinyTalesTV
    └── [Other Channels]/     # Rendered audio for other channels
```

---

## 🎤 Your Channels

### Currently Configured:
- **WiredWorkshop** - Uses `wired-eliv3`
- **TinyTalesTV** - Uses `tinytales-female`, `tinytales-female2`, `tinytales-oldman`
- **WiredToWork** - Uses `wired-businessgirl`
- **LearningTheWires** - Uses `wired-eliv3`
- **ViceCityVault** - Uses `other-man1`
- **NeuralWires** - Uses `wired-eliv3`
- **EliDolney** - Uses `wired-eliv3`
- **LotsOfErrors** - Uses `other-man2`
- **FomoFactory** - Uses `tinytales-female`

### Adding a New Channel:

1. **Add to `voices/channels.json`**:
```json
{
  "YourNewChannel": [
    "voice-preset-name"
  ]
}
```

2. **Create voice preset in `voices/presets.json`** (if needed):
```json
{
  "voice-preset-name": {
    "prompt": "voices/your-voice.wav",
    "exaggeration": 0.50,
    "cfg_weight": 0.65,
    "temperature": 0.8
  }
}
```

3. **Create script folder**:
```bash
mkdir -p scripts/YourNewChannel
```

4. **Test it**:
```bash
./.venv/bin/python tools/test_channels.py --channel YourNewChannel
```

---

## 🛠️ Common Workflows

### Complete Lesson Workflow (WiredWorkshop CCNA Example)

```bash
# 1. Render all clips
./.venv/bin/python tools/channel_manager.py render WiredWorkshop \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --subfolder CCNA/NetworkTopologies01 \
  --to-48k --lufs-target -16

# 2. Concatenate into final audio
./.venv/bin/python tools/channel_manager.py concat WiredWorkshop \
  --subfolder CCNA/NetworkTopologies01
```

### Testing a New Voice

```bash
# 1. Add voice to presets.json
# 2. Test it
./.venv/bin/python tools/test_channels.py \
  --channel YourChannel \
  --test-text "This is a test of the new voice preset."
```

### Batch Processing Multiple Scripts

```bash
# Process all CCNA lessons
for script in scripts/WiredWorkshop/CCNA/*.csv; do
  lesson_name=$(basename "$script" .csv)
  echo "Processing $lesson_name..."
  
  # Render
  ./.venv/bin/python tools/channel_manager.py render WiredWorkshop \
    --script "$script" \
    --subfolder "CCNA/$lesson_name" \
    --to-48k --lufs-target -16
  
  # Concatenate
  ./.venv/bin/python tools/channel_manager.py concat WiredWorkshop \
    --subfolder "CCNA/$lesson_name"
done
```

---

## 📝 Script Format

### CSV Format (Recommended)
```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Hello and welcome to my channel.,001-intro,wired-eliv3,0.58,0.45,0.75
Today we're learning about networks.,002-main,wired-eliv3,0.58,0.45,0.75
Thanks for watching!,003-outro,wired-eliv3,0.58,0.45,0.75
```

### Plain Text Format
```txt
Hello and welcome to my channel.
Today we're learning about networks.
Thanks for watching!
```

---

## 🔧 Troubleshooting

### Voice not generating?
1. Check if prompt file exists:
   ```bash
   ls -la voices/*.wav
   ```
2. Test the voice:
   ```bash
   ./.venv/bin/python tools/test_channels.py --channel YourChannel
   ```

### Wrong voice being used?
- Check `voices/channels.json` - make sure channel is mapped correctly
- Check `voices/presets.json` - make sure preset exists
- Use `--voice` flag to override: `--voice preset-name`

### Audio quality issues?
- Ensure `--to-48k` flag is set for video production
- Use `--lufs-target -16` for consistent loudness
- Check LUFS normalization is working (requires `pyloudnorm`)

---

## 🚀 Next Steps

1. **Test all channels**: Run `tools/test_channels.py` to verify setup
2. **Organize scripts**: Create folders for each channel in `scripts/`
3. **Start rendering**: Use `channel_manager.py render` for production
4. **Scale up**: Use batch scripts for multiple lessons

---

## 💡 Tips

- **Always test first**: Use `test_channels.py` before production renders
- **Use subfolders**: Keep outputs organized with `--subfolder`
- **Version control**: Commit your `channels.json` and `presets.json`
- **Backup voices**: Keep your `.wav` voice files backed up
- **Document changes**: Note when you tune voice parameters

---

## 📞 Need Help?

- Check `YOUTUBE_EMPIRE_ROADMAP.md` for future enhancements
- Review `README.md` for general TTS usage
- Test individual components with `tools/test_channels.py`

