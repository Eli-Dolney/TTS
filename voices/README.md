# Voices folder

This folder holds **your** reference audio and preset configuration. Nothing here is shipped with the repo except starter JSON templates.

## Quick start

1. Add a clean 5–15 second WAV of the voice you want to clone (mono, dry, minimal background noise).
2. Save it here, e.g. `voices/your-voice.wav`.
3. Edit `presets.json` so the `prompt` path points at your file.
4. In the Production UI (**Voice Management**), upload a WAV or use **Import from YouTube** to create a preset automatically.

## Files

| File | Purpose |
|------|---------|
| `presets.json` | Voice presets (name → prompt path + generation params) |
| `channels.json` | YouTube/production channels and which presets they use |
| `presets.example.json` | Copy-friendly template for multiple voices |
| `channels.example.json` | Copy-friendly template for multiple channels |
| `*.wav` | Your voice samples (gitignored — not committed) |

## Tips

- Use 5–15 seconds of clear speech for best cloning quality.
- One WAV can back multiple presets with different exaggeration/CFG/temperature.
- Paths in JSON should be relative to the project root, e.g. `voices/my-voice.wav`.
