#!/usr/bin/env python3
"""Download and prepare voice samples from YouTube URLs for Chatterbox cloning."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def parse_time_to_seconds(value: str) -> float:
    """Parse mm:ss, hh:mm:ss, or plain seconds into total seconds."""
    value = (value or "").strip()
    if not value:
        return 0.0

    if ":" not in value:
        return max(0.0, float(value))

    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return max(0.0, int(minutes) * 60 + float(seconds))
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return max(0.0, int(hours) * 3600 + int(minutes) * 60 + float(seconds))

    raise ValueError(f"Invalid time format: {value}")


def slugify_name(name: str) -> str:
    """Turn a voice name into a safe filename stem."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-").lower()
    return slug or "youtube-voice"


def find_ffmpeg() -> str:
    """Locate ffmpeg: system binary first, then static-ffmpeg package."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
    except Exception:
        pass

    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg or run: pip install static-ffmpeg"
    )


def validate_audio_file(file_path: Path) -> Tuple[bool, str]:
    """Validate that an audio file is valid and has reasonable duration."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(file_path))
        duration = waveform.shape[1] / sr
        if duration < 1:
            return False, "Audio too short (< 1 second)"
        if duration > 60:
            return False, "Audio too long (> 60 seconds). Use 5-15 second clips for best results."
        return True, f"Valid audio: {duration:.1f}s at {sr}Hz"
    except Exception as e:
        return False, f"Invalid audio file: {e}"


def _postprocess_voice_clip(
    wav_path: Path,
    target_sr: int = 24000,
    max_duration: float = 30.0,
) -> Path:
    """Trim silence, resample, peak-normalize, and cap duration."""
    audio, sr = librosa.load(str(wav_path), sr=None, mono=True)

    if audio.size == 0:
        raise ValueError("Downloaded audio is empty")

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if audio.size == 0:
        raise ValueError("Audio is silent after trimming")

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    max_samples = int(max_duration * sr)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95

    sf.write(str(wav_path), audio, sr, subtype="PCM_16")
    return wav_path


def download_voice_sample(
    url: str,
    out_dir: Path,
    out_name: str,
    start: str = "0",
    duration: float = 15.0,
) -> Tuple[Path, str]:
    """
    Download a voice sample slice from YouTube and save as a WAV prompt.

    Returns (output_path, status_message).
    """
    if not url or not url.strip():
        raise ValueError("YouTube URL is required")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify_name(out_name)
    out_path = out_dir / f"{stem}.wav"

    start_seconds = parse_time_to_seconds(start)
    duration = max(5.0, min(30.0, float(duration)))
    end_seconds = start_seconds + duration

    ffmpeg_path = find_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="yt-voice-") as tmpdir:
        tmp = Path(tmpdir)
        template = str(tmp / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "download_sections": [f"*{start_seconds}-{end_seconds}"],
            "ffmpeg_location": str(Path(ffmpeg_path).parent),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }

        try:
            import yt_dlp

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "YouTube video") if info else "YouTube video"
        except Exception as e:
            raise RuntimeError(f"YouTube download failed: {e}") from e

        wav_files = sorted(tmp.glob("*.wav"))
        if not wav_files:
            raise RuntimeError("No audio file produced from YouTube download")

        shutil.copy2(wav_files[0], out_path)

    _postprocess_voice_clip(out_path, target_sr=24000, max_duration=duration)

    valid, msg = validate_audio_file(out_path)
    if not valid:
        if out_path.exists():
            out_path.unlink()
        raise ValueError(msg)

    return out_path, f"Downloaded from '{title}': {msg}"
