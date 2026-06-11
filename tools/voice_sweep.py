#!/usr/bin/env python3
"""
Voice Parameter Sweep Tool
Generate the same text with different voice parameters to find the best settings.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from itertools import product

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def generate_sweep_csv(
    text: str,
    output_csv: Path,
    exaggeration_range: list[float],
    cfg_weight_range: list[float],
    temperature_range: list[float],
    voice: str = "demo",
) -> None:
    """Generate a CSV with all parameter combinations."""
    rows = [("text", "filename", "voice", "exaggeration", "cfg_weight", "temperature")]
    
    for exag, cfg, temp in product(exaggeration_range, cfg_weight_range, temperature_range):
        filename = f"test-exag{exag:.2f}-cfg{cfg:.2f}-temp{temp:.2f}"
        rows.append((f'"{text}"', filename, voice, exag, cfg, temp))
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Generated {len(rows)-1} parameter combinations in {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice parameter sweep for testing"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello welcome to learning the wires. Today we are testing out my voice clone.",
        help="Text to generate with different parameters",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/Demo/voice_test_sweep.csv"),
        help="Output CSV file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/voice_sweep"),
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="demo",
        help="Voice preset to use",
    )
    parser.add_argument(
        "--exag-range",
        type=str,
        default="0.50,0.55,0.58,0.60",
        help="Comma-separated exaggeration values",
    )
    parser.add_argument(
        "--cfg-range",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated cfg_weight values",
    )
    parser.add_argument(
        "--temp-range",
        type=str,
        default="0.70,0.75,0.80",
        help="Comma-separated temperature values",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--to-48k",
        action="store_true",
        default=True,
        help="Resample to 48kHz",
    )
    parser.add_argument(
        "--lufs-target",
        type=float,
        default=-16.0,
        help="LUFS normalization target",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate CSV, don't render audio",
    )
    
    args = parser.parse_args()
    
    # Parse ranges
    exag_range = [float(x.strip()) for x in args.exag_range.split(",")]
    cfg_range = [float(x.strip()) for x in args.cfg_range.split(",")]
    temp_range = [float(x.strip()) for x in args.temp_range.split(",")]
    
    # Generate CSV
    generate_sweep_csv(
        text=args.text,
        output_csv=args.output_csv,
        exaggeration_range=exag_range,
        cfg_weight_range=cfg_range,
        temperature_range=temp_range,
        voice=args.voice,
    )
    
    if args.generate_only:
        print("CSV generated. Run with example_tts.py to render audio.")
        return
    
    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Generating {len(exag_range) * len(cfg_range) * len(temp_range)} audio files...")
    
    # Load model
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Load presets
    presets_path = Path("voices/presets.json")
    presets = {}
    if presets_path.exists():
        import json
        with presets_path.open("r", encoding="utf-8") as f:
            presets = json.load(f)
    
    # Get voice prompt
    voice_prompt = None
    if args.voice in presets:
        prompt_path = presets[args.voice].get("prompt") or presets[args.voice].get("audio_prompt")
        if prompt_path:
            voice_prompt = Path(prompt_path)
    
    if not voice_prompt or not voice_prompt.exists():
        print(f"Warning: Voice prompt not found for {args.voice}")
        return
    
    # Prepare conditionals once
    model.prepare_conditionals(str(voice_prompt), exaggeration=0.5)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and generate
    with args.output_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip('"')
            filename = row["filename"]
            exag = float(row["exaggeration"])
            cfg = float(row["cfg_weight"])
            temp = float(row["temperature"])
            
            output_path = args.output_dir / f"{filename}.wav"
            
            print(f"Generating: {filename} (exag={exag:.2f}, cfg={cfg:.2f}, temp={temp:.2f})")
            
            # Update exaggeration if needed
            if exag != model.conds.t3.emotion_adv[0, 0, 0]:
                from chatterbox.tts import T3Cond
                _cond = model.conds.t3
                model.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exag * torch.ones(1, 1, 1),
                ).to(device=device)
            
            # Generate
            wav = model.generate(
                text,
                exaggeration=exag,
                cfg_weight=cfg,
                temperature=temp,
            )
            
            # Post-process
            import librosa
            target_sr = model.sr
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            
            if args.to_48k and target_sr != 48000:
                wav_np = librosa.resample(wav_np, orig_sr=target_sr, target_sr=48000)
                target_sr = 48000
            
            if args.lufs_target is not None:
                try:
                    import pyloudnorm as pyln
                    meter = pyln.Meter(target_sr)
                    loud = meter.integrated_loudness(wav_np)
                    gain_db = args.lufs_target - loud
                    factor = 10 ** (gain_db / 20.0)
                    wav_np = wav_np * factor
                except Exception:
                    pass
            
            ta.save(str(output_path), torch.from_numpy(wav_np).unsqueeze(0), target_sr)
            print(f"  Saved: {output_path}")
    
    print(f"\n✅ Generated {len(exag_range) * len(cfg_range) * len(temp_range)} audio files in {args.output_dir}")
    print(f"\nListen to the files and pick your favorite parameters!")


if __name__ == "__main__":
    main()


