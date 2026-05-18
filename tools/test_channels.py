#!/usr/bin/env python3
"""
Test tool to verify all channels and voices generate correctly.
This ensures your multi-channel setup is working before production use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


def load_json(path: Path) -> dict:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_voice(
    model: ChatterboxTTS,
    preset_name: str,
    preset_config: dict,
    test_text: str,
    output_dir: Path,
    device: str,
) -> tuple[bool, str]:
    """
    Test a single voice preset.
    Returns (success: bool, message: str)
    """
    try:
        prompt_path = Path(preset_config.get("prompt") or preset_config.get("audio_prompt", ""))
        if not prompt_path.exists():
            return False, f"Prompt file not found: {prompt_path}"

        # Prepare conditionals
        exaggeration = preset_config.get("exaggeration", 0.5)
        model.prepare_conditionals(str(prompt_path), exaggeration=exaggeration)

        # Generate
        wav = model.generate(
            test_text,
            exaggeration=preset_config.get("exaggeration", 0.5),
            cfg_weight=preset_config.get("cfg_weight", 0.5),
            temperature=preset_config.get("temperature", 0.8),
        )

        # Save test file
        output_file = output_dir / f"test-{preset_name}.wav"
        ta.save(str(output_file), wav, model.sr)

        return True, f"✓ Generated: {output_file.name}"

    except Exception as e:
        return False, f"✗ Error: {str(e)}"


def test_channel(
    model: ChatterboxTTS,
    channel_name: str,
    voice_keys: List[str],
    presets: Dict,
    test_text: str,
    output_dir: Path,
    device: str,
) -> Dict:
    """Test all voices for a channel."""
    results = {
        "channel": channel_name,
        "voices": [],
        "all_passed": True,
    }

    print(f"\n{'='*60}")
    print(f"Testing Channel: {channel_name}")
    print(f"Voices: {', '.join(voice_keys)}")
    print(f"{'='*60}")

    for voice_key in voice_keys:
        if voice_key not in presets:
            print(f"  ✗ {voice_key}: Preset not found in presets.json")
            results["voices"].append({
                "name": voice_key,
                "success": False,
                "message": "Preset not found",
            })
            results["all_passed"] = False
            continue

        preset = presets[voice_key]
        success, message = test_voice(model, voice_key, preset, test_text, output_dir, device)
        
        print(f"  {message}")
        results["voices"].append({
            "name": voice_key,
            "success": success,
            "message": message,
        })
        
        if not success:
            results["all_passed"] = False

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test all channels and voices to verify setup"
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=Path("voices/channels.json"),
        help="Path to channels.json",
    )
    parser.add_argument(
        "--presets-file",
        type=Path,
        default=Path("voices/presets.json"),
        help="Path to presets.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/test_all_channels"),
        help="Directory to save test audio files",
    )
    parser.add_argument(
        "--test-text",
        type=str,
        default="Hello, this is a test of the voice generation system.",
        help="Test text to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Test only this channel (default: test all)",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Test text: {args.test_text}")

    # Load configs
    channels = load_json(args.channels_file)
    presets = load_json(args.presets_file)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (once, reuse for all tests)
    print("\nLoading TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

    # Test channels
    channels_to_test = [args.channel] if args.channel else list(channels.keys())
    
    if args.channel and args.channel not in channels:
        print(f"Error: Channel '{args.channel}' not found in channels.json")
        print(f"Available channels: {', '.join(channels.keys())}")
        sys.exit(1)

    all_results = []
    for channel_name in channels_to_test:
        voice_keys = channels[channel_name]
        results = test_channel(
            model, channel_name, voice_keys, presets, args.test_text, args.output_dir, device
        )
        all_results.append(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_channels = len(all_results)
    passed_channels = sum(1 for r in all_results if r["all_passed"])
    total_voices = sum(len(r["voices"]) for r in all_results)
    passed_voices = sum(
        sum(1 for v in r["voices"] if v["success"])
        for r in all_results
    )

    print(f"Channels: {passed_channels}/{total_channels} passed")
    print(f"Voices: {passed_voices}/{total_voices} passed")
    print(f"\nTest files saved to: {args.output_dir}")

    if passed_channels == total_channels and passed_voices == total_voices:
        print("\n✓ All tests passed! Your setup is ready for production.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

