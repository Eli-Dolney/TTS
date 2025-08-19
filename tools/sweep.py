from __future__ import annotations

import argparse
from pathlib import Path
from itertools import product

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def main():
    p = argparse.ArgumentParser(description="Parameter sweep for Chatterbox (exaggeration x cfg_weight)")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--prompt", type=Path, default=None)
    p.add_argument("--exag", type=str, default="0.4,0.5,0.7,0.9", help="Comma list of exaggeration values")
    p.add_argument("--cfg", type=str, default="0.3,0.4,0.5,0.6", help="Comma list of cfg_weight values")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--outdir", type=Path, default=Path("outputs/sweeps"))
    args = p.parse_args()

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


