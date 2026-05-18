# Quick Reference Card

## 🎯 Most Common Commands

### Test Everything
```bash
./.venv/bin/python tools/test_channels.py
```

### List Channels
```bash
./.venv/bin/python tools/channel_manager.py list channels
```

### Render a Lesson
```bash
./.venv/bin/python tools/channel_manager.py render WiredWorkshop \
  --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
  --subfolder CCNA/NetworkTopologies01
```

### Concatenate Audio
```bash
./.venv/bin/python tools/channel_manager.py concat WiredWorkshop \
  --subfolder CCNA/NetworkTopologies01
```

---

## 📺 Your Channels

| Channel | Default Voice | Scripts Folder |
|---------|--------------|----------------|
| WiredWorkshop | wired-eliv3 | `scripts/WiredWorkshop/` |
| TinyTalesTV | tinytales-female | `scripts/TinyTalesTV/` |
| WiredToWork | wired-businessgirl | `scripts/WiredToWork/` |
| LearningTheWires | wired-eliv3 | `scripts/LearningTheWires/` |
| ViceCityVault | other-man1 | `scripts/ViceCityVault/` |
| NeuralWires | wired-eliv3 | `scripts/NeuralWires/` |
| EliDolney | wired-eliv3 | `scripts/EliDolney/` |
| LotsOfErrors | other-man2 | `scripts/LotsOfErrors/` |
| FomoFactory | tinytales-female | `scripts/FomoFactory/` |

---

## 🎤 Voice Presets

All voices are configured in `voices/presets.json`. Each preset has:
- `prompt`: Path to voice sample WAV file
- `exaggeration`: Emotion/intensity (0.0-1.0)
- `cfg_weight`: Pacing/stability (0.0-1.0)
- `temperature`: Sampling randomness (0.0-1.0)

---

## 📝 Script Format

**CSV (Recommended)**:
```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Hello world.,001-intro,wired-eliv3,0.58,0.45,0.75
```

**Plain Text**:
```txt
Hello world.
This is line two.
```

---

## 🔧 Troubleshooting

**Problem**: Voice not generating
**Solution**: Run `tools/test_channels.py` to check setup

**Problem**: Wrong voice used
**Solution**: Check `voices/channels.json` mapping

**Problem**: Audio quality issues
**Solution**: Use `--to-48k --lufs-target -16` flags

---

## 📚 Full Documentation

- `CHANNEL_SETUP_GUIDE.md` - Complete setup guide
- `YOUTUBE_EMPIRE_ROADMAP.md` - Future enhancements
- `README.md` - General TTS documentation

