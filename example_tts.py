import torchaudio as ta
import torch
from pathlib import Path
import re
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
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
    filename_stem: str | None = None


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
                specs.append(LineSpec(text, exaggeration, cfg_weight, temperature, stem))
        return specs

    # Plain text: one line per utterance
    with script_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [
        LineSpec(text=ln, exaggeration=defaults.exaggeration, cfg_weight=defaults.cfg_weight, temperature=defaults.temperature)
        for ln in lines if ln
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch TTS generator for Chatterbox")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save wav files")
    parser.add_argument("--script", type=Path, default=SCRIPT_PATH, help="Text/CSV/TSV file with lines to synthesize")
    parser.add_argument("--prompt", type=Path, default=AUDIO_PROMPT_PATH, help="Path to voice prompt wav (optional)")
    parser.add_argument("--exaggeration", type=float, default=EXAGGERATION, help="Emotion/intensity control [0..1]")
    parser.add_argument("--cfg-weight", type=float, default=CFG_WEIGHT, help="Classifier-free guidance weight [0..1]")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Runtime device override")
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

    # Prepare conditionals once if we have a prompt
    use_prompt = args.prompt.exists()
    if use_prompt:
        model.prepare_conditionals(str(args.prompt), exaggeration=args.exaggeration)

    defaults = LineSpec(
        text="Hello from Chatterbox.",
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    specs = read_lines(args.script, defaults)

    start_idx = args.start_index if args.start_index is not None else next_index(args.output_dir)

    with log_path.open("a", encoding="utf-8") as log_file:
        for offset, spec in enumerate(specs, start=0):
            idx = start_idx + offset
            idx_str = f"{idx:03d}"
            stem = spec.filename_stem or slugify(spec.text)
            fname = f"{idx_str}-{stem}.wav"
            out_path = args.output_dir / fname

            if out_path.exists() and not args.overwrite:
                print(f"Skip (exists): {out_path}")
                continue

            wav = model.generate(
                spec.text,
                exaggeration=spec.exaggeration,
                cfg_weight=spec.cfg_weight,
                temperature=spec.temperature,
            )

            ta.save(str(out_path), wav, model.sr)
            print(f"Saved: {out_path}")

            meta = {
                "index": idx,
                "filename": fname,
                "text": spec.text,
                "exaggeration": spec.exaggeration,
                "cfg_weight": spec.cfg_weight,
                "temperature": spec.temperature,
                "prompt_path": str(args.prompt) if use_prompt else None,
                "device": selected_device,
                "seed": args.seed,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "sample_rate": model.sr,
            }
            log_file.write(json.dumps(meta, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

