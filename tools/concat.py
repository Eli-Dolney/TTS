#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf


def read_wav_mono(path: Path, target_sr: int, mixdown_mono: bool) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1 and mixdown_mono:
        data = data.mean(axis=1)
    # Ensure float32
    data = data.astype(np.float32, copy=False)
    if sr != target_sr:
        import librosa
        if data.ndim > 1:
            # If still stereo (mixdown_mono=False), resample each channel then stack
            channels = []
            for ch in range(data.shape[1]):
                channels.append(librosa.resample(data[:, ch], orig_sr=sr, target_sr=target_sr))
            data = np.stack(channels, axis=1)
        else:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    # Safety: avoid NaNs/Infs
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def lufs_normalize(wave: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln  # type: ignore
    except Exception:
        print("Note: pyloudnorm not available; skipping LUFS normalization.")
        return wave
    # If multi-channel, process combined integrated loudness and scale globally
    meter = pyln.Meter(sr)
    # pyloudnorm expects mono; for stereo, average channels
    mono = wave.mean(axis=1) if wave.ndim > 1 else wave
    loud = meter.integrated_loudness(mono)
    gain_db = target_lufs - loud
    factor = 10 ** (gain_db / 20.0)
    return (wave * factor).astype(np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Concatenate WAV files with a fixed gap and optional LUFS normalization.")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing input WAV files")
    p.add_argument("--pattern", type=str, default="[0-9][0-9][0-9]-*.wav", help="Glob pattern to match files (sorted)")
    p.add_argument("--out", type=Path, required=True, help="Output WAV path")
    p.add_argument("--gap-seconds", type=float, default=0.5, help="Silence gap between clips in seconds")
    p.add_argument("--target-sr", type=int, default=48000, help="Target sample rate for output")
    p.add_argument("--mono", action="store_true", help="Mix down to mono before concat")
    p.add_argument("--lufs-target", type=float, default=None, help="Normalize final file to target LUFS (e.g., -16)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    args.input_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = sorted(args.input_dir.glob(args.pattern))
    if not files:
        print(f"No files match: {args.input_dir}/{args.pattern}")
        return

    silence = np.zeros(int(args.target_sr * max(0.0, args.gap_seconds)), dtype=np.float32)
    parts: List[np.ndarray] = []

    for idx, f in enumerate(files):
        wav = read_wav_mono(f, target_sr=args.target_sr, mixdown_mono=args.mono)
        # Peak safety per clip
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = (wav / peak).astype(np.float32)
        parts.append(wav)
        if idx < len(files) - 1:
            parts.append(silence)

    if parts:
        combined = np.concatenate(parts)
    else:
        combined = np.zeros(1, dtype=np.float32)

    # Final peak safety
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 1.0:
        combined = (combined / peak).astype(np.float32)

    if args.lufs_target is not None:
        combined = lufs_normalize(combined, args.target_sr, args.lufs_target)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), combined, args.target_sr)
    print(f"Combined {len(files)} files -> {args.out}")


if __name__ == "__main__":
    main()


