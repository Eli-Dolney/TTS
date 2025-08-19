from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def select_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Simple TTS sample for macOS (MPS-enabled)")
    parser.add_argument("--text", type=str, default="Hello from Chatterbox on Apple Silicon.", help="Text to synthesize")
    parser.add_argument("--prompt", type=Path, default=None, help="Optional reference voice WAV (5–15s, clean)")
    parser.add_argument("--out", type=Path, default=Path("outputs/mac-sample.wav"), help="Output WAV path")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion/energy [0..1+] (0.5 neutral)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="Pace/stability [0..1]")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Runtime device")
    args = parser.parse_args()

    # Auto-select device with preference for CUDA>MPS>CPU
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Instantiate model (weights will be cached after first download)
    model = ChatterboxTTS.from_pretrained(device=device)

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Generate speech
    wav = model.generate(
        args.text,
        audio_prompt_path=str(args.prompt) if args.prompt else None,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )

    # Save WAV
    ta.save(str(args.out), wav, model.sr)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


