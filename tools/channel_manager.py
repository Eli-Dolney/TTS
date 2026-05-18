#!/usr/bin/env python3
"""
Unified channel management tool for multi-channel YouTube production.
Handles rendering, testing, and organization for all your channels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path) -> dict:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_channels(channels_file: Path) -> None:
    """List all configured channels."""
    channels = load_json(channels_file)
    print("\n📺 Configured Channels:")
    print("=" * 60)
    for channel_name, voice_keys in channels.items():
        print(f"\n{channel_name}:")
        for voice_key in voice_keys:
            print(f"  • {voice_key}")
    print(f"\nTotal: {len(channels)} channels")


def list_voices(presets_file: Path) -> None:
    """List all voice presets."""
    presets = load_json(presets_file)
    print("\n🎤 Available Voice Presets:")
    print("=" * 60)
    for preset_name, preset_config in presets.items():
        prompt = preset_config.get("prompt", "N/A")
        exists = "✓" if Path(prompt).exists() else "✗"
        print(f"{exists} {preset_name}")
        print(f"    Prompt: {prompt}")
        print(f"    Params: exag={preset_config.get('exaggeration', 0.5)}, "
              f"cfg={preset_config.get('cfg_weight', 0.5)}, "
              f"temp={preset_config.get('temperature', 0.8)}")
    print(f"\nTotal: {len(presets)} presets")


def render_channel(
    channel: str,
    script: Path,
    channels_file: Path,
    presets_file: Path,
    output_dir: Optional[Path] = None,
    voice: Optional[str] = None,
    to_48k: bool = True,
    lufs_target: float = -16.0,
    device: str = "auto",
    overwrite: bool = False,
    subfolder: Optional[str] = None,
) -> None:
    """Render audio for a channel."""
    channels = load_json(channels_file)
    
    if channel not in channels:
        print(f"Error: Channel '{channel}' not found.")
        print(f"Available channels: {', '.join(channels.keys())}")
        sys.exit(1)

    # Determine output directory
    if output_dir is None:
        output_dir = Path("outputs") / channel
        if subfolder:
            output_dir = output_dir / subfolder

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use channel's default voice if not specified
    if voice is None:
        voice_keys = channels[channel]
        if voice_keys:
            voice = voice_keys[0]  # Use first voice as default
            print(f"Using default voice for {channel}: {voice}")

    # Build command
    cmd = [
        sys.executable,
        "example_tts.py",
        "--output-dir", str(output_dir),
        "--script", str(script),
        "--presets-file", str(presets_file),
        "--device", device,
    ]

    if voice:
        cmd += ["--voice", voice]
    if to_48k:
        cmd += ["--to-48k"]
    if lufs_target is not None:
        cmd += ["--lufs-target", str(lufs_target)]
    if overwrite:
        cmd += ["--overwrite"]

    print(f"\n🎬 Rendering {channel}")
    print(f"   Script: {script}")
    print(f"   Output: {output_dir}")
    print(f"   Voice: {voice}")
    print(f"\nRunning: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


def concat_channel(
    channel: str,
    input_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    gap_seconds: float = 0.5,
    target_sr: int = 48000,
    lufs_target: float = -16.0,
    subfolder: Optional[str] = None,
) -> None:
    """Concatenate audio files for a channel."""
    if input_dir is None:
        input_dir = Path("outputs") / channel
        if subfolder:
            input_dir = input_dir / subfolder

    if output_file is None:
        output_file = input_dir / "combined.wav"

    cmd = [
        sys.executable,
        "tools/concat.py",
        "--input-dir", str(input_dir),
        "--out", str(output_file),
        "--gap-seconds", str(gap_seconds),
        "--target-sr", str(target_sr),
        "--mono",
        "--lufs-target", str(lufs_target),
    ]

    print(f"\n🔗 Concatenating audio for {channel}")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_file}")
    print(f"\nRunning: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Unified channel management for multi-channel YouTube production"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List all channels or voices")
    list_parser.add_argument("what", choices=["channels", "voices"], help="What to list")
    list_parser.add_argument(
        "--channels-file",
        type=Path,
        default=Path("voices/channels.json"),
        help="Path to channels.json",
    )
    list_parser.add_argument(
        "--presets-file",
        type=Path,
        default=Path("voices/presets.json"),
        help="Path to presets.json",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test all channels/voices")
    test_parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Test only this channel (default: all)",
    )
    test_parser.add_argument(
        "--channels-file",
        type=Path,
        default=Path("voices/channels.json"),
    )
    test_parser.add_argument(
        "--presets-file",
        type=Path,
        default=Path("voices/presets.json"),
    )
    test_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    # Render command
    render_parser = subparsers.add_parser("render", help="Render audio for a channel")
    render_parser.add_argument("channel", type=str, help="Channel name")
    render_parser.add_argument("--script", type=Path, required=True, help="Script file (CSV/TXT)")
    render_parser.add_argument(
        "--channels-file",
        type=Path,
        default=Path("voices/channels.json"),
    )
    render_parser.add_argument(
        "--presets-file",
        type=Path,
        default=Path("voices/presets.json"),
    )
    render_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/<channel>)",
    )
    render_parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within channel output (e.g., 'CCNA/NetworkTopologies01')",
    )
    render_parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice preset to use (default: channel's first voice)",
    )
    render_parser.add_argument(
        "--to-48k",
        action="store_true",
        default=True,
        help="Resample to 48kHz (default: True)",
    )
    render_parser.add_argument(
        "--no-48k",
        action="store_false",
        dest="to_48k",
        help="Don't resample to 48kHz",
    )
    render_parser.add_argument(
        "--lufs-target",
        type=float,
        default=-16.0,
        help="LUFS normalization target (default: -16)",
    )
    render_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    render_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )

    # Concat command
    concat_parser = subparsers.add_parser("concat", help="Concatenate audio for a channel")
    concat_parser.add_argument("channel", type=str, help="Channel name")
    concat_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory (default: outputs/<channel>)",
    )
    concat_parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within channel output",
    )
    concat_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output file (default: <input-dir>/combined.wav)",
    )
    concat_parser.add_argument(
        "--gap-seconds",
        type=float,
        default=0.5,
        help="Gap between clips in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    if args.command == "list":
        if args.what == "channels":
            list_channels(args.channels_file)
        else:
            list_voices(args.presets_file)

    elif args.command == "test":
        cmd = [
            sys.executable,
            "tools/test_channels.py",
            "--channels-file", str(args.channels_file),
            "--presets-file", str(args.presets_file),
            "--device", args.device,
        ]
        if args.channel:
            cmd += ["--channel", args.channel]
        subprocess.run(cmd, check=True)

    elif args.command == "render":
        render_channel(
            channel=args.channel,
            script=args.script,
            channels_file=args.channels_file,
            presets_file=args.presets_file,
            output_dir=args.output_dir,
            voice=args.voice,
            to_48k=args.to_48k,
            lufs_target=args.lufs_target,
            device=args.device,
            overwrite=args.overwrite,
            subfolder=args.subfolder,
        )

    elif args.command == "concat":
        concat_channel(
            channel=args.channel,
            input_dir=args.input_dir,
            output_file=args.output_file,
            gap_seconds=args.gap_seconds,
            subfolder=args.subfolder,
        )


if __name__ == "__main__":
    main()

