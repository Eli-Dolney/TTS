import torchaudio as ta
import torch
from pathlib import Path
import re
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import librosa
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# ------------ User settings (defaults; can be overridden via CLI) ------------
# Where to save the generated files (portable default)
OUTPUT_DIR = Path("outputs")

# Optional: path to a text file containing one line per utterance
# If this file doesn't exist, the fallback single example text will be used.
SCRIPT_PATH = Path("script file.txt")

# Optional: your voice prompt (short clean sample of your voice)
AUDIO_PROMPT_PATH = Path("prompt.wav")

# Optional generation controls
EXAGGERATION = 0.5
CFG_WEIGHT = 0.5
TEMPERATURE = 0.8
# ---------------------------------------


@dataclass
class LineSpec:
    text: str
    exaggeration: float
    cfg_weight: float
    temperature: float
    filename_stem: Optional[str] = None
    voice_name: Optional[str] = None
    prompt_path: Optional[Path] = None


def slugify(text: str, max_words: int = 8) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    words = words[:max_words]
    slug = "-".join(words).lower()
    # Ensure at least something present
    return slug if slug else "line"


def next_index(prefix_dir: Path) -> int:
    """Return the next zero-padded index based on existing files like 001-*.wav."""
    existing = [p.stem for p in prefix_dir.glob("*.wav")]
    max_idx = 0
    for stem in existing:
        m = re.match(r"^(\d{3,})-", stem)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass
    return max_idx + 1


def read_lines(script_path: Path, defaults: LineSpec) -> list[LineSpec]:
    if not script_path.exists():
        return [defaults]

    # CSV/TSV support with headers: text, filename, exaggeration, cfg_weight, temperature
    if script_path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if script_path.suffix.lower() == ".tsv" else ","
        specs: list[LineSpec] = []
        with script_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                stem = (row.get("filename") or row.get("file") or "").strip() or None
                exaggeration = float(row.get("exaggeration") or defaults.exaggeration)
                cfg_weight = float(row.get("cfg_weight") or row.get("cfg") or defaults.cfg_weight)
                temperature = float(row.get("temperature") or defaults.temperature)
                voice_name = (row.get("voice") or "").strip() or None
                prompt_col = (row.get("prompt") or row.get("audio_prompt") or "").strip() or None
                prompt_path = Path(prompt_col) if prompt_col else None
                specs.append(LineSpec(text, exaggeration, cfg_weight, temperature, stem, voice_name, prompt_path))
        return specs

    # Plain text: one line per utterance
    with script_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    specs: list[LineSpec] = []
    for ln in lines:
        if not ln:
            continue
        # Support optional "voice_name: actual text" prefix in plain text files
        m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.+)$", ln)
        if m:
            voice_name = m.group(1)
            text = m.group(2)
            specs.append(
                LineSpec(
                    text=text,
                    exaggeration=defaults.exaggeration,
                    cfg_weight=defaults.cfg_weight,
                    temperature=defaults.temperature,
                    voice_name=voice_name,
                )
            )
        else:
            specs.append(
                LineSpec(
                    text=ln,
                    exaggeration=defaults.exaggeration,
                    cfg_weight=defaults.cfg_weight,
                    temperature=defaults.temperature,
                    voice_name=defaults.voice_name,
                )
            )
    return specs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch TTS generator for Chatterbox")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save wav files")
    parser.add_argument("--script", type=Path, default=SCRIPT_PATH, help="Text/CSV/TSV file with lines to synthesize")
    parser.add_argument("--prompt", type=Path, default=AUDIO_PROMPT_PATH, help="Path to voice prompt wav (optional)")
    parser.add_argument("--exaggeration", type=float, default=EXAGGERATION, help="Emotion/intensity control [0..1]")
    parser.add_argument("--cfg-weight", type=float, default=CFG_WEIGHT, help="Classifier-free guidance weight [0..1]")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Runtime device override")
    parser.add_argument("--presets-file", type=Path, default=Path("voices/presets.json"), help="JSON file of voice presets")
    parser.add_argument("--voice", type=str, default=None, help="Default voice preset name (overridable per line via CSV 'voice' column)")
    parser.add_argument("--to-48k", action="store_true", help="Resample output to 48 kHz for video")
    parser.add_argument("--lufs-target", type=float, default=None, help="Normalize loudness to target LUFS (e.g., -16). Requires pyloudnorm if available.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional starting index for filenames (e.g. 101)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing wav files instead of skipping")
    parser.add_argument("--seed", type=int, default=None, help="Set random seed for reproducibility")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve device
    selected_device = device if args.device == "auto" else args.device

    # Seed for determinism (best-effort; sampling variability may still occur)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metadata.jsonl"

    model = ChatterboxTTS.from_pretrained(device=selected_device)

    # Load presets if present
    presets: dict[str, dict] = {}
    if args.presets_file and args.presets_file.exists():
        try:
            with args.presets_file.open("r", encoding="utf-8") as pf:
                presets = json.load(pf)
        except Exception:
            presets = {}

    default_voice = args.voice
    if default_voice and default_voice not in presets:
        print(f"Warning: voice preset '{default_voice}' not found in {args.presets_file}")

    defaults = LineSpec(
        text="Hello from Chatterbox.",
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        voice_name=default_voice,
        prompt_path=args.prompt if args.prompt and args.prompt.exists() else None,
    )
    specs = read_lines(args.script, defaults)

    start_idx = args.start_index if args.start_index is not None else next_index(args.output_dir)

    with log_path.open("a", encoding="utf-8") as log_file:
        last_voice_key = None
        for offset, spec in enumerate(specs, start=0):
            idx = start_idx + offset
            idx_str = f"{idx:03d}"
            stem = spec.filename_stem or slugify(spec.text)
            fname = f"{idx_str}-{stem}.wav"
            out_path = args.output_dir / fname

            if out_path.exists() and not args.overwrite:
                print(f"Skip (exists): {out_path}")
                continue

            # Resolve voice and prompt
            resolved_voice = spec.voice_name or default_voice
            resolved_prompt: Optional[Path] = spec.prompt_path
            if resolved_voice and resolved_voice in presets:
                preset = presets[resolved_voice]
                ppath = preset.get("prompt") or preset.get("audio_prompt")
                if ppath:
                    resolved_prompt = Path(ppath)
                # Merge defaults if user did not override
                if spec.exaggeration == args.exaggeration:
                    try:
                        spec.exaggeration = float(preset.get("exaggeration", spec.exaggeration))
                    except Exception:
                        pass
                if spec.cfg_weight == args.cfg_weight:
                    try:
                        spec.cfg_weight = float(preset.get("cfg_weight", spec.cfg_weight))
                    except Exception:
                        pass
                if spec.temperature == args.temperature:
                    try:
                        spec.temperature = float(preset.get("temperature", spec.temperature))
                    except Exception:
                        pass

            # Prepare conditionals when voice/prompt changes
            voice_key = (resolved_voice or "__none__", str(resolved_prompt) if resolved_prompt else "__builtin__")
            if resolved_prompt is not None and voice_key != last_voice_key:
                if resolved_prompt.exists():
                    model.prepare_conditionals(str(resolved_prompt), exaggeration=spec.exaggeration)
                    last_voice_key = voice_key
                else:
                    print(f"Warning: prompt not found for voice '{resolved_voice}': {resolved_prompt}")

            wav = model.generate(
                spec.text,
                exaggeration=spec.exaggeration,
                cfg_weight=spec.cfg_weight,
                temperature=spec.temperature,
            )

            # Post-processing: resample to 48k and/or LUFS normalize
            target_sr = model.sr
            wav_np = wav.squeeze(0).detach().cpu().numpy()

            if args.to_48k and target_sr != 48000:
                wav_np = librosa.resample(wav_np, orig_sr=target_sr, target_sr=48000)
                target_sr = 48000

            if args.lufs_target is not None:
                try:
                    import pyloudnorm as pyln  # type: ignore
                    meter = pyln.Meter(target_sr)
                    loud = meter.integrated_loudness(wav_np)
                    gain_db = args.lufs_target - loud
                    factor = 10 ** (gain_db / 20.0)
                    wav_np = wav_np * factor
                except Exception:
                    print("Note: pyloudnorm not available; skipping LUFS normalization.")

            ta.save(str(out_path), torch.from_numpy(wav_np).unsqueeze(0), target_sr)
            print(f"Saved: {out_path}")

            meta = {
                "index": idx,
                "filename": fname,
                "text": spec.text,
                "exaggeration": spec.exaggeration,
                "cfg_weight": spec.cfg_weight,
                "temperature": spec.temperature,
                "prompt_path": str(resolved_prompt) if resolved_prompt else None,
                "voice": resolved_voice,
                "device": selected_device,
                "seed": args.seed,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "sample_rate": target_sr,
            }
            log_file.write(json.dumps(meta, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

