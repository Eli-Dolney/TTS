# Test Video Scripts

## 📝 Overview

Test scripts have been created for all your channels to showcase each voice. These are short, 2-3 line scripts perfect for:
- Testing voice generation
- Creating demo videos
- Comparing voice quality
- Showcasing your channel voices

## 📂 Test Scripts Created

### Multi-Voice Channels

#### **TinyTalesTV** (`scripts/TinyTalesTV/test_all_voices.csv`)
- **3 voices**: `tinytales-female`, `tinytales-female2`, `tinytales-oldman`
- **7 clips**: Each voice says 2 lines, plus an outro
- **Theme**: Storytelling and magical adventures

#### **Other Channel** (`scripts/Other/test_all_voices.csv`)
- **3 voices**: `other-man1`, `other-man2`, `other-news`
- **6 clips**: Each voice says 2 lines
- **Theme**: General content, news, and educational

### Single-Voice Channels

Each single-voice channel has a `test_voice.csv` with 3 clips (intro, main, outro):

- **WiredWorkshop** - `wired-eliv3` (tuned for CCNA)
- **WiredToWork** - `wired-businessgirl`
- **LearningTheWires** - `wired-eliv3`
- **ViceCityVault** - `other-man1`
- **NeuralWires** - `wired-eliv3`
- **EliDolney** - `wired-eliv3`
- **LotsOfErrors** - `other-man2`
- **FomoFactory** - `tinytales-female`

### Comprehensive Test

#### **ALL_VOICES_TEST.csv** (`scripts/ALL_VOICES_TEST.csv`)
- **All 7 unique voices** in one script
- **17 clips**: 2 lines per voice + summary
- **Perfect for**: Comparing all voices side-by-side

## 🎬 How to Render

### Option 1: Render Individual Channel

```bash
# Example: Render TinyTalesTV test
./.venv/bin/python tools/channel_manager.py render TinyTalesTV \
  --script scripts/TinyTalesTV/test_all_voices.csv \
  --subfolder test_all_voices \
  --to-48k --lufs-target -16

# Then concatenate
./.venv/bin/python tools/channel_manager.py concat TinyTalesTV \
  --subfolder test_all_voices
```

### Option 2: Render All Tests (Batch)

```bash
# Make script executable
chmod +x scripts/render_all_tests.sh

# Run all tests
cd scripts
./render_all_tests.sh
```

### Option 3: Render Comprehensive Test

```bash
# Render all voices in one go (uses Other channel as base)
./.venv/bin/python example_tts.py \
  --script scripts/ALL_VOICES_TEST.csv \
  --presets-file voices/presets.json \
  --output-dir outputs/ALL_VOICES_TEST \
  --to-48k --lufs-target -16

# Concatenate
./.venv/bin/python tools/concat.py \
  --input-dir outputs/ALL_VOICES_TEST \
  --out outputs/ALL_VOICES_TEST/combined.wav \
  --gap-seconds 0.5 --target-sr 48000 --mono --lufs-target -16
```

## 📊 What Each Test Includes

### TinyTalesTV Test
1. Female voice intro
2. Female voice story line
3. Female2 voice intro
4. Female2 voice story line
5. Old man voice intro
6. Old man voice story line
7. Outro (female voice)

### Other Channel Test
1. Man1 intro
2. Man1 content
3. Man2 intro
4. Man2 content
5. News intro
6. News outro

### Single-Voice Tests
Each includes:
1. Channel intro
2. Main content line
3. Outro

## 🎯 Use Cases

1. **Voice Quality Check**: Verify all voices generate correctly
2. **Demo Videos**: Create showcase videos for each channel
3. **A/B Testing**: Compare different voice parameters
4. **Client Presentations**: Show available voices
5. **Training**: Learn how each voice sounds

## 📁 Output Locations

After rendering, find your test videos in:

```
outputs/
├── TinyTalesTV/test_all_voices/     # Multi-voice test
├── Other/test_all_voices/           # Multi-voice test
├── WiredWorkshop/test_voice/        # Single voice test
├── WiredToWork/test_voice/          # Single voice test
├── LearningTheWires/test_voice/     # Single voice test
├── ViceCityVault/test_voice/        # Single voice test
├── NeuralWires/test_voice/          # Single voice test
├── EliDolney/test_voice/            # Single voice test
├── LotsOfErrors/test_voice/          # Single voice test
├── FomoFactory/test_voice/           # Single voice test
└── ALL_VOICES_TEST/                  # Comprehensive test
```

## 🔧 Customization

Want to change the test scripts? Edit the CSV files:

```csv
text,filename,voice,exaggeration,cfg_weight,temperature
Your custom text here.,001-custom-name,voice-preset-name,0.50,0.65,0.8
```

Then re-render using the commands above.

## ✅ Next Steps

1. **Render a test**: Start with one channel to verify it works
2. **Listen to results**: Check audio quality and voice consistency
3. **Adjust parameters**: Tune exaggeration/cfg_weight if needed
4. **Create your content**: Use these as templates for real scripts

---

**Tip**: Start with the comprehensive `ALL_VOICES_TEST.csv` to hear all voices at once, then render individual channel tests for specific use cases.

