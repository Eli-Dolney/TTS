#!/usr/bin/env python3
"""
Automated Lesson Pipeline - One command to process a full lesson from script to final combined audio.

This tool automates the entire workflow:
1. Render all clips from CSV script
2. Organize files into proper folder structure
3. Concatenate clips with gaps
4. Generate metadata JSON

Usage:
    python tools/lesson.py \
      --script scripts/WiredWorkshop/CCNA/NetworkTopologies01.csv \
      --channel WiredWorkshop \
      --subfolder CCNA/NetworkTopologies01
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


def detect_channel_from_path(script_path: Path) -> Optional[str]:
    """Try to detect channel name from script path."""
    parts = script_path.parts
    if "scripts" in parts:
        idx = parts.index("scripts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def render_lesson(
    script: Path,
    channel: str,
    subfolder: Optional[str] = None,
    voice: Optional[str] = None,
    to_48k: bool = True,
    lufs_target: float = -16.0,
    device: str = "auto",
    overwrite: bool = False,
    presets_file: Path = Path("voices/presets.json"),
    channels_file: Path = Path("voices/channels.json"),
) -> Path:
    """Render all clips for a lesson."""
    # Determine output directory
    output_dir = Path("outputs") / channel
    if subfolder:
        output_dir = output_dir / subfolder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎬 RENDERING LESSON")
    print(f"{'='*60}")
    print(f"Script: {script}")
    print(f"Channel: {channel}")
    print(f"Output: {output_dir}")
    if subfolder:
        print(f"Subfolder: {subfolder}")
    if voice:
        print(f"Voice: {voice}")
    print(f"{'='*60}\n")
    
    # Build render command
    cmd = [
        sys.executable,
        "tools/channel_manager.py",
        "render",
        channel,
        "--script", str(script),
        "--output-dir", str(output_dir),
        "--presets-file", str(presets_file),
        "--channels-file", str(channels_file),
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
    
    # Run render
    subprocess.run(cmd, check=True)
    
    return output_dir


def move_files_to_subfolder(output_dir: Path, subfolder: Optional[str] = None) -> Path:
    """Move rendered files to subfolder if needed."""
    if not subfolder:
        return output_dir
    
    target_dir = output_dir / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files in output_dir (not in subfolders)
    wav_files = [f for f in output_dir.glob("*.wav") if f.is_file()]
    
    if wav_files:
        print(f"\n📁 Moving {len(wav_files)} files to {target_dir}")
        for wav_file in wav_files:
            target_path = target_dir / wav_file.name
            if target_path.exists() and not target_path.samefile(wav_file):
                print(f"  ⚠️  Overwriting: {target_path.name}")
            shutil.move(str(wav_file), str(target_path))
            print(f"  ✓ Moved: {wav_file.name}")
    
    return target_dir


def concatenate_lesson(
    input_dir: Path,
    gap_seconds: float = 0.5,
    target_sr: int = 48000,
    lufs_target: float = -16.0,
) -> Path:
    """Concatenate all clips into final audio."""
    output_file = input_dir / "combined.wav"
    
    print(f"\n{'='*60}")
    print(f"🔗 CONCATENATING AUDIO")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_file}")
    print(f"Gap: {gap_seconds}s")
    print(f"{'='*60}\n")
    
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
    
    subprocess.run(cmd, check=True)
    
    return output_file


def generate_metadata(
    script: Path,
    channel: str,
    output_dir: Path,
    combined_file: Path,
    subfolder: Optional[str] = None,
) -> Path:
    """Generate metadata JSON for the lesson."""
    metadata = {
        "channel": channel,
        "script": str(script),
        "output_dir": str(output_dir),
        "combined_file": str(combined_file),
        "subfolder": subfolder,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clips": [],
    }
    
    # Count clips
    wav_files = sorted(output_dir.glob("[0-9][0-9][0-9]-*.wav"))
    metadata["clip_count"] = len(wav_files)
    
    # Get file sizes
    if combined_file.exists():
        metadata["combined_size_bytes"] = combined_file.stat().st_size
        metadata["combined_size_mb"] = round(metadata["combined_size_bytes"] / (1024 * 1024), 2)
    
    metadata_file = output_dir / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Metadata saved: {metadata_file}")
    
    return metadata_file


def main():
    parser = argparse.ArgumentParser(
        description="Automated Lesson Pipeline - Render, organize, and concatenate in one command"
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Script file (CSV/TXT)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Channel name (auto-detected from script path if not provided)",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within channel output (e.g., 'CCNA/NetworkTopologies01')",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice preset to use (default: channel's first voice)",
    )
    parser.add_argument(
        "--to-48k",
        action="store_true",
        default=True,
        help="Resample to 48kHz (default: True)",
    )
    parser.add_argument(
        "--no-48k",
        action="store_false",
        dest="to_48k",
        help="Don't resample to 48kHz",
    )
    parser.add_argument(
        "--lufs-target",
        type=float,
        default=-16.0,
        help="LUFS normalization target (default: -16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=0.5,
        help="Gap between clips in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip rendering (only concatenate existing files)",
    )
    parser.add_argument(
        "--skip-concat",
        action="store_true",
        help="Skip concatenation (only render)",
    )
    parser.add_argument(
        "--presets-file",
        type=Path,
        default=Path("voices/presets.json"),
        help="Path to presets.json",
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=Path("voices/channels.json"),
        help="Path to channels.json",
    )
    
    args = parser.parse_args()
    
    # Validate script exists
    if not args.script.exists():
        print(f"Error: Script file not found: {args.script}")
        sys.exit(1)
    
    # Detect channel if not provided
    channel = args.channel
    if not channel:
        channel = detect_channel_from_path(args.script)
        if not channel:
            print("Error: Could not detect channel from script path.")
            print("Please specify --channel explicitly.")
            sys.exit(1)
        print(f"Auto-detected channel: {channel}")
    
    # Auto-detect subfolder from script name if not provided
    subfolder = args.subfolder
    if not subfolder:
        # Try to extract from script path
        script_stem = args.script.stem
        script_parent = args.script.parent
        
        # If script is in a subfolder like CCNA/, use that
        if "CCNA" in script_parent.parts:
            ccna_idx = script_parent.parts.index("CCNA")
            if ccna_idx + 1 < len(script_parent.parts):
                subfolder = f"CCNA/{script_parent.parts[ccna_idx + 1]}"
            else:
                subfolder = f"CCNA/{script_stem}"
        elif script_parent.name != channel:
            # Use parent folder name if it's not the channel name
            subfolder = f"{script_parent.name}/{script_stem}"
        else:
            subfolder = script_stem
        
        print(f"Auto-detected subfolder: {subfolder}")
    
    # Step 1: Render
    output_dir = None
    if not args.skip_render:
        output_dir = render_lesson(
            script=args.script,
            channel=channel,
            subfolder=subfolder,
            voice=args.voice,
            to_48k=args.to_48k,
            lufs_target=args.lufs_target,
            device=args.device,
            overwrite=args.overwrite,
            presets_file=args.presets_file,
            channels_file=args.channels_file,
        )
        
        # Move files if needed (channel_manager might have put them in wrong place)
        if output_dir and subfolder:
            final_dir = output_dir.parent / subfolder
            if final_dir != output_dir:
                output_dir = move_files_to_subfolder(output_dir.parent, subfolder)
    else:
        # Use existing output directory
        output_dir = Path("outputs") / channel
        if subfolder:
            output_dir = output_dir / subfolder
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}")
            print("Cannot skip render if files don't exist.")
            sys.exit(1)
    
    # Step 2: Concatenate
    combined_file = None
    if not args.skip_concat:
        combined_file = concatenate_lesson(
            input_dir=output_dir,
            gap_seconds=args.gap_seconds,
            lufs_target=args.lufs_target,
        )
    else:
        combined_file = output_dir / "combined.wav"
        if not combined_file.exists():
            print(f"Warning: Combined file not found: {combined_file}")
    
    # Step 3: Generate metadata
    if output_dir and combined_file:
        generate_metadata(
            script=args.script,
            channel=channel,
            output_dir=output_dir,
            combined_file=combined_file,
            subfolder=subfolder,
        )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ LESSON COMPLETE")
    print(f"{'='*60}")
    print(f"Channel: {channel}")
    if subfolder:
        print(f"Subfolder: {subfolder}")
    print(f"Output Directory: {output_dir}")
    if combined_file and combined_file.exists():
        size_mb = combined_file.stat().st_size / (1024 * 1024)
        print(f"Combined Audio: {combined_file} ({size_mb:.2f} MB)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

