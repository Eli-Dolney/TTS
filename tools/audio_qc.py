#!/usr/bin/env python3
"""Audio quality checks for generated TTS clips."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class QCResult:
    """Outcome of quality checks on a generated clip."""

    passed: bool
    issues: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def summary(self) -> str:
        if self.passed:
            return "ok"
        return "; ".join(self.issues)

    @property
    def issue_count(self) -> int:
        return len(self.issues)


def _estimate_duration_bounds(text: str) -> tuple[float, float]:
    """Estimate reasonable min/max duration from text length."""
    char_count = max(1, len(text.strip()))
    # Typical speech: ~10-22 chars/sec for narration
    min_seconds = max(0.4, char_count / 22.0)
    max_seconds = max(min_seconds + 0.5, char_count / 6.0)
    return min_seconds, max_seconds


def check_clip(wav_np: np.ndarray, sr: int, text: str) -> QCResult:
    """
    Run quality checks on a generated clip.

    Checks: empty/NaN audio, clipping, silence, duration sanity.
    """
    issues: List[str] = []

    if wav_np is None or wav_np.size == 0:
        return QCResult(passed=False, issues=["empty audio"], duration_seconds=0.0)

    audio = np.asarray(wav_np, dtype=np.float64).squeeze()
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    if not np.isfinite(audio).all():
        issues.append("contains NaN or Inf")

    duration = float(audio.shape[0]) / float(sr) if sr > 0 else 0.0

    if duration < 0.15:
        issues.append("too short")

    min_dur, max_dur = _estimate_duration_bounds(text)
    if duration < min_dur * 0.5:
        issues.append("truncated (much shorter than expected)")
    elif duration > max_dur * 2.5:
        issues.append("runaway generation (much longer than expected)")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-4:
        issues.append("mostly silent")
    elif peak >= 0.99:
        issues.append("clipping detected")

    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 0.005 and peak >= 1e-4:
        issues.append("very quiet")

    # Internal silence: long stretch below threshold
    if audio.size > sr and peak > 1e-4:
        threshold = max(0.01, peak * 0.02)
        silent = np.abs(audio) < threshold
        max_run = 0
        current = 0
        for is_silent in silent:
            if is_silent:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        max_silence_seconds = max_run / sr
        if max_silence_seconds > 1.5:
            issues.append("long internal silence")

    return QCResult(
        passed=len(issues) == 0,
        issues=issues,
        duration_seconds=duration,
    )
