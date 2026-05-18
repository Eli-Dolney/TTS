#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict


def load_channels(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise SystemExit(f"channels file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid channels json: {path}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Channel utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="Render audio for a channel using presets")
    r.add_argument("channel", type=str, help="Channel name (e.g., WiredToWork)")
    r.add_argument("--script", type=Path, required=True, help="Input script (txt/csv/tsv)")
    r.add_argument("--output-dir", type=Path, default=None, help="Output directory (default outputs/<Channel>)")
    r.add_argument("--channels-file", type=Path, default=Path("voices/channels.json"), help="Channels mapping JSON")
    r.add_argument("--presets-file", type=Path, default=Path("voices/presets.json"), help="Voice presets JSON")
    r.add_argument("--voice", type=str, default=None, help="Default preset key to use when not specified in CSV")
    r.add_argument("--exaggeration", type=float, default=None, help="Optional override of exaggeration")
    r.add_argument("--cfg-weight", type=float, default=None, help="Optional override of cfg weight")
    r.add_argument("--temperature", type=float, default=None, help="Optional override of temperature")
    r.add_argument("--to-48k", action="store_true", help="Resample output to 48kHz")
    r.add_argument("--lufs-target", type=float, default=None, help="Target LUFS normalization (e.g., -16)")
    r.add_argument("--overwrite", action="store_true", help="Overwrite existing wavs")
    r.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"], help="Device")

    return p


def render_channel(args: argparse.Namespace) -> None:
    channels = load_channels(args.channels_file)
    if args.channel not in channels:
        raise SystemExit(f"channel not found: {args.channel}")

    output_dir = args.output_dir or Path("outputs") / args.channel
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "example_tts.py",
        "--output-dir", str(output_dir),
        "--script", str(args.script),
        "--presets-file", str(args.presets_file),
        "--device", args.device,
    ]

    if args.voice:
        cmd += ["--voice", args.voice]
    if args.to_48k:
        cmd += ["--to-48k"]
    if args.lufs_target is not None:
        cmd += ["--lufs-target", str(args.lufs_target)]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.exaggeration is not None:
        cmd += ["--exaggeration", str(args.exaggeration)]
    if args.cfg_weight is not None:
        cmd += ["--cfg-weight", str(args.cfg_weight)]
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    if args.cmd == "render":
        render_channel(args)
    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()


