
<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/XqS7RxUp)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce Chatterbox, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our [Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- SoTA zeroshot TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Tips
- **General Use (TTS and Voice Agents):**
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.


# Installation
```
pip install chatterbox-tts
```


# Quickstart

```bash
pip install chatterbox-tts
```

### Windows (PowerShell) one-time setup for local dev
```powershell
cd H:\python\TTS\chatterbox
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

### Text-to-Speech (TTS)
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")  # or "cpu" / "mps"

text = "Hello world from Chatterbox."
wav = model.generate(text)
ta.save("outputs/test-1.wav", wav, model.sr)

# Synthesize with your own voice prompt (a short, clean WAV)
AUDIO_PROMPT_PATH = "prompt.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("outputs/test-2.wav", wav, model.sr)
```

### Batch generate from a script file
By default `example_tts.py` reads `script file.txt` (one line per utterance, or CSV/TSV with headers `text,filename,exaggeration,cfg_weight,temperature`) and writes WAVs to `outputs/`.

```powershell
cd H:\python\TTS\chatterbox
.\.venv\Scripts\python.exe example_tts.py --prompt "prompt.wav" --script "script file.txt" --output-dir "outputs" --overwrite
```

Optional controls:
- `--exaggeration` (emotion/energy, 0..1)
- `--cfg-weight` (pacing/stability, 0..1)
- `--temperature` (sampling randomness)

Examples:
```powershell
.\.venv\Scripts\python.exe example_tts.py --prompt "prompt.wav" --script "script file.txt" --output-dir "outputs" --exaggeration 0.7 --cfg-weight 0.3 --overwrite
```

### Voice Conversion (VC)
```python
import torchaudio as ta
from chatterbox.vc import ChatterboxVC

model = ChatterboxVC.from_pretrained("cuda")
wav = model.generate(
    audio="input.wav",          # source speech to convert
    target_voice_path="voice.wav"  # target voice reference
)
ta.save("outputs/testvc.wav", wav, model.sr)
```

### YouTube video workflow (example)
1. Write lines in `script file.txt` (one per line). Keep them short and natural.
2. Record or choose your prompt voice as `prompt.wav` (5–15s, clean, dry).
3. Generate audio:
   ```powershell
   .\.venv\Scripts\python.exe example_tts.py --prompt "prompt.wav" --script "script file.txt" --output-dir "outputs" --exaggeration 0.6 --cfg-weight 0.4 --overwrite
   ```
4. Import resulting WAVs from `outputs/` into your editor (Premiere, CapCut, Resolve).
5. Add background music/SFX; render your video for upload.


# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


# Official Discord

👋 Join us on [Discord](https://discord.gg/XqS7RxUp) and let's build something awesome together!

# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
