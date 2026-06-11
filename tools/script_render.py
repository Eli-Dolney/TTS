#!/usr/bin/env python3
"""Shared script loading, line selection, and TTS rendering helpers."""

from __future__ import annotations

import csv
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio as ta

from audio_qc import QCResult, check_clip


@dataclass
class ScriptLine:
    """One row from a script CSV."""

    index: int  # 1-based line number in script
    text: str
    filename: str
    voice: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    prompt: str = ""


EMPHASIS_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "normal": (0.45, 0.55, 0.75),
    "emphasis": (0.65, 0.50, 0.70),
    "dramatic": (0.85, 0.45, 0.65),
    "calm": (0.30, 0.65, 0.85),
    "whisper": (0.25, 0.70, 0.80),
}


def parse_line_selection(spec: str, max_line: int) -> List[int]:
    """
    Parse line selection like '5, 12, 23-26' into sorted unique 1-based indices.

    Raises ValueError on invalid input.
    """
    if not spec or not spec.strip():
        raise ValueError("Enter line numbers (e.g. 5, 12, 23-26)")

    selected: set[int] = set()
    for part in re.split(r"[,;\s]+", spec.strip()):
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
                raise ValueError(f"Invalid range: {part}")
            start, end = int(bounds[0]), int(bounds[1])
            if start > end:
                start, end = end, start
            for n in range(start, end + 1):
                if 1 <= n <= max_line:
                    selected.add(n)
                elif n > max_line:
                    raise ValueError(f"Line {n} out of range (script has {max_line} lines)")
        elif part.isdigit():
            n = int(part)
            if n < 1 or n > max_line:
                raise ValueError(f"Line {n} out of range (script has {max_line} lines)")
            selected.add(n)
        else:
            raise ValueError(f"Invalid token: {part}")

    if not selected:
        raise ValueError("No valid line numbers selected")

    return sorted(selected)


def load_script_lines(script_path: Path) -> List[ScriptLine]:
    """Load script rows from a CSV file."""
    lines: List[ScriptLine] = []
    line_num = 0
    with script_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            line_num += 1
            lines.append(
                ScriptLine(
                    index=line_num,
                    text=text,
                    filename=(row.get("filename") or "").strip(),
                    voice=(row.get("voice") or "").strip(),
                    exaggeration=float(row.get("exaggeration") or 0.5),
                    cfg_weight=float(row.get("cfg_weight") or 0.5),
                    temperature=float(row.get("temperature") or 0.8),
                    prompt=(row.get("prompt") or "").strip(),
                )
            )
    return lines


def slugify_text(text: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    return "-".join(words[:max_words]).lower() or "line"


def output_path_for_line(output_dir: Path, line: ScriptLine) -> Path:
    """Resolve the WAV path for a script line (matches full-render naming)."""
    idx_str = f"{line.index:03d}"
    if line.filename:
        stem = line.filename
    else:
        stem = slugify_text(line.text)

    exact = output_dir / f"{idx_str}-{stem}.wav"
    if exact.exists():
        return exact

    # Overwrite an existing clip at this index even if filename stem changed
    matches = sorted(output_dir.glob(f"{idx_str}-*.wav"))
    if matches:
        return matches[0]

    return exact


def resolve_voice_prompt(
    line: ScriptLine,
    presets: Dict[str, dict],
    voice_override: str = "",
) -> Tuple[Optional[Path], str, float, float, float]:
    """
    Resolve prompt path and generation params for a line.

    Per-line CSV values always win; preset supplies prompt path and fallbacks only.
    """
    voice = line.voice or voice_override or ""
    prompt_path: Optional[Path] = None
    exag = line.exaggeration
    cfg = line.cfg_weight
    temp = line.temperature

    if line.prompt and Path(line.prompt).exists():
        prompt_path = Path(line.prompt)
    elif voice and voice in presets:
        preset = presets[voice]
        ppath = preset.get("prompt") or preset.get("audio_prompt")
        if ppath and Path(ppath).exists():
            prompt_path = Path(ppath)
        if not line.voice:
            exag = float(preset.get("exaggeration", exag))
            cfg = float(preset.get("cfg_weight", cfg))
            temp = float(preset.get("temperature", temp))

    return prompt_path, voice, exag, cfg, temp


def format_eta(seconds: float) -> str:
    """Format ETA seconds as a human-readable string."""
    if seconds <= 0 or not np.isfinite(seconds):
        return "calculating..."
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def postprocess_wav(
    wav,
    model_sr: int,
    to_48k: bool = True,
    lufs_target: float = 0.0,
) -> Tuple[Any, int]:
    """Resample and optionally LUFS-normalize generated audio."""
    target_sr = model_sr
    wav_np = wav.squeeze(0).detach().cpu().numpy()

    if to_48k and target_sr != 48000:
        wav_np = librosa.resample(wav_np, orig_sr=target_sr, target_sr=48000)
        target_sr = 48000

    if lufs_target != 0:
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(target_sr)
            loud = meter.integrated_loudness(wav_np)
            gain_db = lufs_target - loud
            factor = 10 ** (gain_db / 20.0)
            wav_np = wav_np * factor
        except Exception:
            pass

    return wav_np, target_sr


def _generate_with_qc(
    model,
    line: ScriptLine,
    *,
    to_48k: bool,
    lufs_target: float,
    validate: bool,
    max_retries: int,
) -> Tuple[Optional[np.ndarray], int, QCResult, int, Optional[str]]:
    """
    Generate audio for one line with optional QC and retries.

    Returns (wav_np, target_sr, qc_result, retries_used, error_message).
    """
    attempts = max_retries + 1
    best_wav: Optional[np.ndarray] = None
    best_sr = model.sr
    best_qc = QCResult(passed=False, issues=["no attempt"])
    retries_used = 0
    last_error: Optional[str] = None

    for attempt in range(attempts):
        if attempt > 0:
            retries_used = attempt
            torch.manual_seed(random.randint(0, 2**31 - 1))

        try:
            wav = model.generate(
                line.text,
                exaggeration=line.exaggeration,
                cfg_weight=line.cfg_weight,
                temperature=line.temperature,
            )
            wav_np, target_sr = postprocess_wav(wav, model.sr, to_48k, lufs_target)

            if validate:
                qc = check_clip(wav_np, target_sr, line.text)
            else:
                qc = QCResult(passed=True, duration_seconds=float(wav_np.shape[0]) / target_sr)

            if best_wav is None or qc.issue_count < best_qc.issue_count:
                best_wav = wav_np
                best_sr = target_sr
                best_qc = qc

            if not validate or qc.passed:
                return wav_np, target_sr, qc, retries_used, None

        except Exception as e:
            last_error = str(e)
            if attempt >= attempts - 1:
                break

    if best_wav is not None:
        return best_wav, best_sr, best_qc, retries_used, last_error

    return None, model.sr, best_qc, retries_used, last_error or "generation failed"


def render_script_lines(
    model,
    lines: List[ScriptLine],
    output_dir: Path,
    presets: Dict[str, dict],
    *,
    line_indices: Optional[List[int]] = None,
    voice_override: str = "",
    to_48k: bool = True,
    lufs_target: float = 0.0,
    overwrite: bool = True,
    param_overrides: Optional[Dict[str, float]] = None,
    progress_callback=None,
    validate: bool = True,
    max_retries: int = 2,
) -> List[Dict[str, str]]:
    """
    Render one or more script lines to WAV files.

    line_indices: 1-based line numbers to render; None = all lines.
    param_overrides: optional {exaggeration, cfg_weight, temperature} applied to every rendered line.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    index_set = set(line_indices) if line_indices else None
    results: List[Dict[str, str]] = []
    last_voice_key: Optional[Tuple[str, str]] = None

    targets = [ln for ln in lines if index_set is None or ln.index in index_set]
    if not targets:
        return results

    line_times: List[float] = []

    for i, line in enumerate(targets):
        eta_seconds = 0.0
        if line_times:
            avg = sum(line_times) / len(line_times)
            eta_seconds = avg * (len(targets) - i)

        if progress_callback:
            progress_callback(i, len(targets), line, eta_seconds)

        line_start = time.monotonic()

        if param_overrides:
            line = replace(
                line,
                exaggeration=float(param_overrides.get("exaggeration", line.exaggeration)),
                cfg_weight=float(param_overrides.get("cfg_weight", line.cfg_weight)),
                temperature=float(param_overrides.get("temperature", line.temperature)),
            )

        out_path = output_path_for_line(output_dir, line)
        fname = out_path.name

        if out_path.exists() and not overwrite:
            results.append({"file": fname, "line": str(line.index), "status": "skipped (exists)"})
            continue

        prompt_path, voice, exag, cfg, temp = resolve_voice_prompt(
            line, presets, voice_override
        )

        if not prompt_path:
            results.append({
                "file": fname,
                "line": str(line.index),
                "status": f"error: no voice prompt for '{voice or 'default'}'",
            })
            continue

        voice_key = (voice or "__none__", str(prompt_path))
        if voice_key != last_voice_key:
            try:
                model.prepare_conditionals(str(prompt_path), exaggeration=exag)
                last_voice_key = voice_key
            except Exception as e:
                results.append({"file": fname, "line": str(line.index), "status": f"error: {e}"})
                continue

        render_line = replace(line, exaggeration=exag, cfg_weight=cfg, temperature=temp)
        wav_np, target_sr, qc, retries_used, gen_error = _generate_with_qc(
            model,
            render_line,
            to_48k=to_48k,
            lufs_target=lufs_target,
            validate=validate,
            max_retries=max_retries,
        )

        if wav_np is None:
            err = gen_error or qc.summary
            results.append({"file": fname, "line": str(line.index), "status": f"error: {err}"})
            line_times.append(time.monotonic() - line_start)
            continue

        ta.save(str(out_path), torch.from_numpy(wav_np).unsqueeze(0), target_sr)

        status_parts = [f"✓ rendered (exag={exag:.2f}, cfg={cfg:.2f}, temp={temp:.2f})"]
        if retries_used > 0:
            status_parts.append(f"retried {retries_used}x")
        if validate:
            if qc.passed:
                status_parts.append("QC ok")
            else:
                status_parts.append(f"⚠ QC: {qc.summary}")

        results.append({
            "file": fname,
            "line": str(line.index),
            "status": " — ".join(status_parts),
        })
        line_times.append(time.monotonic() - line_start)

    return results


def apply_emphasis_to_rows(
    data: List[List[str]],
    line_spec: str,
    preset_name: str,
) -> Tuple[List[List[str]], str]:
    """Update exaggeration/cfg/temp on selected dataframe rows."""
    if preset_name not in EMPHASIS_PRESETS:
        return data, f"❌ Unknown emphasis preset: {preset_name}"

    if not data:
        return data, "❌ No script rows loaded"

    try:
        indices = parse_line_selection(line_spec, len(data))
    except ValueError as e:
        return data, f"❌ {e}"

    exag, cfg, temp = EMPHASIS_PRESETS[preset_name]
    updated = 0
    new_data = [list(row) for row in data]

    for n in indices:
        row = new_data[n - 1]
        while len(row) < 6:
            row.append("")
        row[3] = str(exag)
        row[4] = str(cfg)
        row[5] = str(temp)
        updated += 1

    label = preset_name.replace("_", " ").title()
    return new_data, f"✅ Applied '{label}' emphasis to {updated} line(s). Save script, then re-render those lines."


def preview_lines(lines: List[ScriptLine], line_spec: str) -> str:
    """Format a preview of selected script lines."""
    try:
        indices = parse_line_selection(line_spec, len(lines))
    except ValueError as e:
        return f"❌ {e}"

    by_index = {ln.index: ln for ln in lines}
    parts = []
    for n in indices:
        ln = by_index.get(n)
        if not ln:
            parts.append(f"Line {n}: (not found)")
            continue
        snippet = ln.text[:120] + ("…" if len(ln.text) > 120 else "")
        parts.append(
            f"**Line {n}** — exag={ln.exaggeration}, cfg={ln.cfg_weight}, temp={ln.temperature}\n"
            f"> {snippet}"
        )
    return "\n\n".join(parts)
