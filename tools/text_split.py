"""Split raw script text into TTS-ready chunks."""
from __future__ import annotations

import re
from typing import Literal

SplitMode = Literal["sentence", "paragraph", "char_limit"]

SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]+(?:\s+|$)|[^.!?…]+$")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    raw = text.replace("\r\n", "\n").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return [normalize_whitespace(p) for p in parts]


def split_by_char_limit(text: str, max_chars: int = 320) -> list[str]:
    max_chars = max(80, int(max_chars))
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current and current_len + sent_len + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_script_text(
    text: str,
    mode: SplitMode = "sentence",
    char_limit: int = 320,
) -> list[str]:
    if not text or not text.strip():
        return []

    if mode == "paragraph":
        return split_paragraphs(text)
    if mode == "char_limit":
        return split_by_char_limit(text, char_limit)
    return split_sentences(text)
