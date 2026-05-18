from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from itertools import product

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def write_channel_csv(channel: str, out_csv: Path, text: str = "hello Welcome to our Empire") -> int:
    presets_path = Path("voices/presets.json")
    channels_path = Path("voices/channels.json")
    if not presets_path.exists() or not channels_path.exists():
        raise FileNotFoundError("voices/presets.json or voices/channels.json not found")
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    channels = json.loads(channels_path.read_text(encoding="utf-8"))
    if channel not in channels:
        raise KeyError(f"Unknown channel '{channel}'. Available: {', '.join(channels.keys())}")
    voice_keys = channels[channel]
    rows = [("text","filename","voice","prompt","exaggeration","cfg_weight","temperature")]
    for key in voice_keys:
        p = presets.get(key)
        if not p:
            continue
        rows.append((text, key, key, p.get("prompt"), p.get("exaggeration",0.5), p.get("cfg_weight",0.5), p.get("temperature",0.8)))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    return len(rows) - 1


def main():
    p = argparse.ArgumentParser(description="Parameter sweep or channel CSV helper for Chatterbox")
    sub = p.add_subparsers(dest="cmd")

    p_sweep = sub.add_parser("sweep", help="Generate a sweep of exaggeration x cfg")
    p_sweep.add_argument("--text", type=str, required=True)
    p_sweep.add_argument("--prompt", type=Path, default=None)
    p_sweep.add_argument("--exag", type=str, default="0.4,0.5,0.7,0.9")
    p_sweep.add_argument("--cfg", type=str, default="0.3,0.4,0.5,0.6")
    p_sweep.add_argument("--temperature", type=float, default=0.8)
    p_sweep.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p_sweep.add_argument("--outdir", type=Path, default=Path("outputs/sweeps"))

    p_csv = sub.add_parser("channel-csv", help="Write a CSV for a channel's presets")
    p_csv.add_argument("channel", type=str)
    p_csv.add_argument("out_csv", type=Path)
    p_csv.add_argument("--text", type=str, default="hello Welcome to our Empire")

    args = p.parse_args()

    if args.cmd == "channel-csv":
        n = write_channel_csv(args.channel, args.out_csv, text=args.text)
        print(f"WROTE {args.out_csv} rows={n}")
        return

    if args.cmd != "sweep":
        p.print_help()
        return

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    model = ChatterboxTTS.from_pretrained(device=device)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Prepare conditionals once if prompt is provided
    if args.prompt and args.prompt.exists():
        model.prepare_conditionals(str(args.prompt), exaggeration=0.5)

    exags = [float(x.strip()) for x in args.exag.split(",") if x.strip()]
    cfgs = [float(x.strip()) for x in args.cfg.split(",") if x.strip()]

    for e, c in product(exags, cfgs):
        wav = model.generate(
            args.text,
            exaggeration=e,
            cfg_weight=c,
            temperature=args.temperature,
        )
        fname = f"{args.text[:32].strip().replace(' ', '_')}-e{e:.2f}-cfg{c:.2f}.wav"
        ta.save(str(args.outdir / fname), wav, model.sr)
        print(f"Saved: {args.outdir / fname}")


if __name__ == "__main__":
    main()


