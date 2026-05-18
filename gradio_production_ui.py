#!/usr/bin/env python3
"""
TTS Production Management UI

A comprehensive Gradio-based interface for managing TTS production:
- Script creation and editing
- Voice management (upload, create presets, edit parameters)
- Channel management
- Rendering and output management
"""

import json
import csv
import shutil
import subprocess
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

# Add src directory to path for local chatterbox module
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torchaudio as ta
import librosa
import gradio as gr

# Import Chatterbox TTS
from chatterbox.tts import ChatterboxTTS

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHANNELS_FILE = VOICES_DIR / "channels.json"
PRESETS_FILE = VOICES_DIR / "presets.json"

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🎙️ TTS Production UI - Using device: {DEVICE}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_json(path: Path) -> dict:
    """Load JSON file, returning empty dict if not found."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def save_json(path: Path, data: dict) -> bool:
    """Save data to JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {path}: {e}")
        return False

def get_channels() -> Dict[str, List[str]]:
    """Load channels configuration."""
    return load_json(CHANNELS_FILE)

def get_presets() -> Dict[str, dict]:
    """Load voice presets configuration."""
    return load_json(PRESETS_FILE)

def get_preset_choices() -> List[str]:
    """Get list of preset names for dropdown."""
    presets = get_presets()
    return list(presets.keys())

def get_channel_choices() -> List[str]:
    """Get list of channel names for dropdown."""
    channels = get_channels()
    return list(channels.keys())

def get_voice_files() -> List[str]:
    """Get list of voice audio files."""
    if not VOICES_DIR.exists():
        return []
    return [f.name for f in VOICES_DIR.glob("*.wav")]

def get_scripts_for_channel(channel: str) -> List[str]:
    """Get list of script files for a channel."""
    channel_scripts_dir = SCRIPTS_DIR / channel
    if not channel_scripts_dir.exists():
        return []
    scripts = []
    for f in channel_scripts_dir.glob("*.csv"):
        scripts.append(f.name)
    for f in channel_scripts_dir.glob("*.txt"):
        scripts.append(f.name)
    return sorted(scripts)

def get_outputs_for_channel(channel: str, subfolder: str = "") -> List[dict]:
    """Get list of output files for a channel."""
    output_dir = OUTPUTS_DIR / channel
    if subfolder:
        output_dir = output_dir / subfolder
    if not output_dir.exists():
        return []
    
    files = []
    for f in sorted(output_dir.glob("*.wav")):
        files.append({
            "name": f.name,
            "path": str(f),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        })
    return files

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def load_model():
    """Load TTS model."""
    print("Loading Chatterbox TTS model...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
    print("Model loaded successfully!")
    return model

# ============================================================================
# VOICE MANAGEMENT TAB
# ============================================================================

def refresh_presets_table() -> List[List[str]]:
    """Get presets data for table display."""
    presets = get_presets()
    rows = []
    for name, config in presets.items():
        prompt = config.get("prompt", "N/A")
        exists = "✓" if Path(prompt).exists() else "✗"
        rows.append([
            name,
            prompt,
            exists,
            str(config.get("exaggeration", 0.5)),
            str(config.get("cfg_weight", 0.5)),
            str(config.get("temperature", 0.8))
        ])
    return rows

def validate_audio_file(file_path: Path) -> Tuple[bool, str]:
    """Validate that an audio file is valid and has reasonable duration."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(file_path))
        duration = waveform.shape[1] / sr
        if duration < 1:
            return False, "Audio too short (< 1 second)"
        if duration > 60:
            return False, "Audio too long (> 60 seconds). Use 5-15 second clips for best results."
        return True, f"Valid audio: {duration:.1f}s at {sr}Hz"
    except Exception as e:
        return False, f"Invalid audio file: {e}"

def upload_voice_file(file) -> Tuple[str, str]:
    """Handle voice file upload."""
    if file is None:
        return "No file selected", ""
    
    try:
        source_path = Path(file.name if hasattr(file, 'name') else file)
        
        # Validate the audio file
        valid, msg = validate_audio_file(source_path)
        if not valid:
            return f"❌ {msg}", ""
        
        dest_path = VOICES_DIR / source_path.name
        
        # Check for overwrite
        if dest_path.exists():
            # Add number suffix
            stem = dest_path.stem
            suffix = dest_path.suffix
            i = 1
            while dest_path.exists():
                dest_path = VOICES_DIR / f"{stem}_{i}{suffix}"
                i += 1
        
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, dest_path)
        
        return f"✅ Uploaded: {dest_path.name} ({msg})", str(dest_path)
    except Exception as e:
        return f"❌ Error uploading: {e}", ""

def create_preset(name: str, prompt_path: str, exag: float, cfg: float, temp: float) -> Tuple[str, List[List[str]]]:
    """Create a new voice preset."""
    if not name or not name.strip():
        return "❌ Please enter a preset name", refresh_presets_table()
    
    name = name.strip()
    presets = get_presets()
    
    if name in presets:
        return f"❌ Preset '{name}' already exists", refresh_presets_table()
    
    # Validate prompt path
    if prompt_path:
        if not Path(prompt_path).exists():
            return f"❌ Prompt file not found: {prompt_path}", refresh_presets_table()
    
    presets[name] = {
        "prompt": prompt_path or "",
        "exaggeration": exag,
        "cfg_weight": cfg,
        "temperature": temp
    }
    
    if save_json(PRESETS_FILE, presets):
        return f"✅ Created preset: {name}", refresh_presets_table()
    else:
        return "❌ Failed to save preset", refresh_presets_table()

def update_preset(name: str, prompt_path: str, exag: float, cfg: float, temp: float) -> Tuple[str, List[List[str]]]:
    """Update an existing voice preset."""
    if not name:
        return "❌ Please select a preset to update", refresh_presets_table()
    
    presets = get_presets()
    
    if name not in presets:
        return f"❌ Preset '{name}' not found", refresh_presets_table()
    
    presets[name] = {
        "prompt": prompt_path or presets[name].get("prompt", ""),
        "exaggeration": exag,
        "cfg_weight": cfg,
        "temperature": temp
    }
    
    if save_json(PRESETS_FILE, presets):
        return f"✅ Updated preset: {name}", refresh_presets_table()
    else:
        return "❌ Failed to save preset", refresh_presets_table()

def delete_preset(name: str) -> Tuple[str, List[List[str]], List[str]]:
    """Delete a voice preset."""
    if not name:
        return "❌ Please select a preset to delete", refresh_presets_table(), get_preset_choices()
    
    presets = get_presets()
    
    if name not in presets:
        return f"❌ Preset '{name}' not found", refresh_presets_table(), get_preset_choices()
    
    del presets[name]
    
    if save_json(PRESETS_FILE, presets):
        return f"✅ Deleted preset: {name}", refresh_presets_table(), get_preset_choices()
    else:
        return "❌ Failed to delete preset", refresh_presets_table(), get_preset_choices()

def load_preset_for_edit(name: str) -> Tuple[str, float, float, float]:
    """Load preset values for editing."""
    if not name:
        return "", 0.5, 0.5, 0.8
    
    presets = get_presets()
    if name not in presets:
        return "", 0.5, 0.5, 0.8
    
    config = presets[name]
    return (
        config.get("prompt", ""),
        config.get("exaggeration", 0.5),
        config.get("cfg_weight", 0.5),
        config.get("temperature", 0.8)
    )

def get_voice_audio_path(name: str) -> Optional[str]:
    """Get audio path for preset preview."""
    if not name:
        return None
    presets = get_presets()
    if name not in presets:
        return None
    prompt = presets[name].get("prompt", "")
    if prompt and Path(prompt).exists():
        return prompt
    return None

def test_voice_generation(
    model,
    preset_name: str,
    test_text: str,
    progress=gr.Progress()
) -> Tuple[Any, str, Optional[str]]:
    """Generate a test audio sample with a voice preset."""
    if model is None:
        return model, "❌ Model not loaded. Please wait for model to load.", None
    
    if not preset_name:
        return model, "❌ Please select a voice preset to test", None
    
    if not test_text or not test_text.strip():
        test_text = "Hello, this is a test of my voice. How does it sound?"
    
    presets = get_presets()
    if preset_name not in presets:
        return model, f"❌ Preset '{preset_name}' not found", None
    
    preset = presets[preset_name]
    prompt_path = preset.get("prompt", "")
    
    if not prompt_path or not Path(prompt_path).exists():
        return model, f"❌ Voice prompt file not found: {prompt_path}", None
    
    try:
        progress(0.2, desc="Preparing voice...")
        model.prepare_conditionals(
            str(prompt_path),
            exaggeration=preset.get("exaggeration", 0.5)
        )
        
        progress(0.5, desc="Generating audio...")
        wav = model.generate(
            test_text.strip(),
            exaggeration=preset.get("exaggeration", 0.5),
            cfg_weight=preset.get("cfg_weight", 0.5),
            temperature=preset.get("temperature", 0.8),
        )
        
        progress(0.9, desc="Saving...")
        # Save to temp file
        test_output_dir = OUTPUTS_DIR / "voice_tests"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_file = test_output_dir / f"test_{preset_name}_{timestamp}.wav"
        
        target_sr = model.sr
        wav_np = wav.squeeze(0).detach().cpu().numpy()
        
        # Resample to 48k
        wav_np = librosa.resample(wav_np, orig_sr=target_sr, target_sr=48000)
        target_sr = 48000
        
        ta.save(str(test_file), torch.from_numpy(wav_np).unsqueeze(0), target_sr)
        
        return model, f"✅ Test generated: {test_file.name}", str(test_file)
    except Exception as e:
        return model, f"❌ Error generating test: {e}", None

# ============================================================================
# CHANNEL MANAGEMENT TAB
# ============================================================================

def refresh_channels_table() -> List[List[str]]:
    """Get channels data for table display."""
    channels = get_channels()
    rows = []
    for name, voices in channels.items():
        voice_list = ", ".join(voices) if voices else "(none)"
        output_dir = OUTPUTS_DIR / name
        has_outputs = "✓" if output_dir.exists() and any(output_dir.glob("*.wav")) else "✗"
        rows.append([name, voice_list, has_outputs])
    return rows

def create_channel(name: str, voices_str: str) -> Tuple[str, List[List[str]], List[str]]:
    """Create a new channel."""
    if not name or not name.strip():
        return "❌ Please enter a channel name", refresh_channels_table(), get_channel_choices()
    
    name = name.strip()
    channels = get_channels()
    
    if name in channels:
        return f"❌ Channel '{name}' already exists", refresh_channels_table(), get_channel_choices()
    
    # Parse voices
    voices = [v.strip() for v in voices_str.split(",") if v.strip()] if voices_str else []
    
    channels[name] = voices
    
    # Create directories
    (SCRIPTS_DIR / name).mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / name).mkdir(parents=True, exist_ok=True)
    
    if save_json(CHANNELS_FILE, channels):
        return f"✅ Created channel: {name}", refresh_channels_table(), get_channel_choices()
    else:
        return "❌ Failed to create channel", refresh_channels_table(), get_channel_choices()

def update_channel_voices(name: str, voices_str: str) -> Tuple[str, List[List[str]]]:
    """Update voices for a channel."""
    if not name:
        return "❌ Please select a channel", refresh_channels_table()
    
    channels = get_channels()
    
    if name not in channels:
        return f"❌ Channel '{name}' not found", refresh_channels_table()
    
    voices = [v.strip() for v in voices_str.split(",") if v.strip()] if voices_str else []
    channels[name] = voices
    
    if save_json(CHANNELS_FILE, channels):
        return f"✅ Updated voices for: {name}", refresh_channels_table()
    else:
        return "❌ Failed to update channel", refresh_channels_table()

def delete_channel(name: str) -> Tuple[str, List[List[str]], List[str]]:
    """Delete a channel (keeps files)."""
    if not name:
        return "❌ Please select a channel to delete", refresh_channels_table(), get_channel_choices()
    
    channels = get_channels()
    
    if name not in channels:
        return f"❌ Channel '{name}' not found", refresh_channels_table(), get_channel_choices()
    
    del channels[name]
    
    if save_json(CHANNELS_FILE, channels):
        return f"✅ Deleted channel: {name} (files preserved)", refresh_channels_table(), get_channel_choices()
    else:
        return "❌ Failed to delete channel", refresh_channels_table(), get_channel_choices()

def load_channel_for_edit(name: str) -> str:
    """Load channel voices for editing."""
    if not name:
        return ""
    channels = get_channels()
    if name not in channels:
        return ""
    return ", ".join(channels[name])

# ============================================================================
# SCRIPT EDITOR TAB
# ============================================================================

def create_empty_script_row() -> List[str]:
    """Create an empty script row."""
    return ["", "", "", "0.5", "0.5", "0.8"]

def refresh_script_data(channel: str, script_name: str) -> Tuple[List[List[str]], str]:
    """Load script data for editing."""
    if not channel or not script_name:
        return [create_empty_script_row()], ""
    
    script_path = SCRIPTS_DIR / channel / script_name
    if not script_path.exists():
        return [create_empty_script_row()], f"Script not found: {script_name}"
    
    try:
        rows = []
        with script_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                if not text:
                    continue
                rows.append([
                    text,
                    row.get("filename", "") or "",
                    row.get("voice", "") or "",
                    row.get("exaggeration", "0.5") or "0.5",
                    row.get("cfg_weight", "0.5") or "0.5",
                    row.get("temperature", "0.8") or "0.8"
                ])
        
        if not rows:
            rows = [create_empty_script_row()]
        
        return rows, f"Loaded: {script_name} ({len(rows)} lines)"
    except Exception as e:
        return [create_empty_script_row()], f"Error loading script: {e}"

def save_script(channel: str, script_name: str, data: List[List[str]]) -> str:
    """Save script data to CSV file."""
    if not channel:
        return "❌ Please select a channel"
    if not script_name:
        return "❌ Please enter a script name"
    
    # Ensure .csv extension
    if not script_name.endswith(".csv"):
        script_name = script_name + ".csv"
    
    script_dir = SCRIPTS_DIR / channel
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / script_name
    
    try:
        # Filter out empty rows
        valid_rows = [row for row in data if row and row[0] and row[0].strip()]
        
        if not valid_rows:
            return "❌ No text lines to save"
        
        with script_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "filename", "voice", "exaggeration", "cfg_weight", "temperature"])
            for row in valid_rows:
                writer.writerow(row)
        
        return f"✅ Saved: {script_path} ({len(valid_rows)} lines)"
    except Exception as e:
        return f"❌ Error saving script: {e}"

def add_script_row(data: List[List[str]]) -> List[List[str]]:
    """Add a new row to script data."""
    if data is None:
        data = []
    data.append(create_empty_script_row())
    return data

def delete_script_row(data: List[List[str]], row_index: int) -> List[List[str]]:
    """Delete a row from script data."""
    if data is None or len(data) == 0:
        return [create_empty_script_row()]
    
    if 0 <= row_index < len(data):
        data.pop(row_index)
    
    if len(data) == 0:
        data = [create_empty_script_row()]
    
    return data

def get_script_choices(channel: str) -> List[str]:
    """Get script choices for a channel."""
    return get_scripts_for_channel(channel) if channel else []

# ============================================================================
# RENDER & OUTPUT TAB
# ============================================================================

def render_script(
    model,
    channel: str,
    script_name: str,
    voice_override: str,
    to_48k: bool,
    lufs_target: float,
    overwrite: bool,
    subfolder: str,
    progress=gr.Progress()
) -> Tuple[Any, str, List[dict]]:
    """Render a script to audio files."""
    if model is None:
        return model, "❌ Model not loaded. Please wait for model to load.", []
    
    if not channel:
        return model, "❌ Please select a channel", []
    if not script_name:
        return model, "❌ Please select a script", []
    
    script_path = SCRIPTS_DIR / channel / script_name
    if not script_path.exists():
        return model, f"❌ Script not found: {script_path}", []
    
    # Determine output directory
    output_dir = OUTPUTS_DIR / channel
    if subfolder and subfolder.strip():
        output_dir = output_dir / subfolder.strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load presets
    presets = get_presets()
    
    # Read script
    try:
        lines = []
        with script_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                lines.append({
                    "text": text,
                    "filename": row.get("filename", "") or "",
                    "voice": row.get("voice", "") or voice_override or "",
                    "exaggeration": float(row.get("exaggeration") or 0.5),
                    "cfg_weight": float(row.get("cfg_weight") or 0.5),
                    "temperature": float(row.get("temperature") or 0.8),
                    "prompt": row.get("prompt", "") or ""
                })
    except Exception as e:
        return model, f"❌ Error reading script: {e}", []
    
    if not lines:
        return model, "❌ No text lines found in script", []
    
    # Find starting index
    existing = [p.stem for p in output_dir.glob("*.wav")]
    max_idx = 0
    for stem in existing:
        import re
        m = re.match(r"^(\d{3,})-", stem)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass
    start_idx = max_idx + 1 if not overwrite else 1
    
    # Render each line
    results = []
    last_voice_key = None
    
    progress(0, desc="Starting render...")
    
    for i, line in enumerate(lines):
        progress((i + 1) / len(lines), desc=f"Rendering line {i + 1}/{len(lines)}")
        
        idx = start_idx + i
        idx_str = f"{idx:03d}"
        
        # Generate filename
        if line["filename"]:
            fname = f"{idx_str}-{line['filename']}.wav"
        else:
            # Slugify text
            import re
            words = re.findall(r"[A-Za-z0-9']+", line["text"])
            slug = "-".join(words[:8]).lower() or "line"
            fname = f"{idx_str}-{slug}.wav"
        
        out_path = output_dir / fname
        
        if out_path.exists() and not overwrite:
            results.append({"file": fname, "status": "skipped (exists)"})
            continue
        
        # Resolve voice/prompt
        voice = line["voice"]
        prompt_path = None
        exag = line["exaggeration"]
        cfg = line["cfg_weight"]
        temp = line["temperature"]
        
        if line["prompt"] and Path(line["prompt"]).exists():
            prompt_path = Path(line["prompt"])
        elif voice and voice in presets:
            preset = presets[voice]
            ppath = preset.get("prompt")
            if ppath and Path(ppath).exists():
                prompt_path = Path(ppath)
            # Use preset defaults if not overridden
            exag = preset.get("exaggeration", exag)
            cfg = preset.get("cfg_weight", cfg)
            temp = preset.get("temperature", temp)
        
        # Prepare conditionals if voice changed
        voice_key = (voice or "__none__", str(prompt_path) if prompt_path else "__builtin__")
        if prompt_path and voice_key != last_voice_key:
            try:
                model.prepare_conditionals(str(prompt_path), exaggeration=exag)
                last_voice_key = voice_key
            except Exception as e:
                results.append({"file": fname, "status": f"error: {e}"})
                continue
        
        # Generate audio
        try:
            wav = model.generate(
                line["text"],
                exaggeration=exag,
                cfg_weight=cfg,
                temperature=temp,
            )
            
            # Post-processing
            target_sr = model.sr
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            
            if to_48k and target_sr != 48000:
                wav_np = librosa.resample(wav_np, orig_sr=target_sr, target_sr=48000)
                target_sr = 48000
            
            if lufs_target != 0:
                try:
                    import pyloudnorm as pyln
                    meter = pyln.Meter(target_sr)
                    loud = meter.integrated_loudness(wav_np)
                    gain_db = lufs_target - loud
                    factor = 10 ** (gain_db / 20.0)
                    wav_np = wav_np * factor
                except Exception:
                    pass
            
            ta.save(str(out_path), torch.from_numpy(wav_np).unsqueeze(0), target_sr)
            results.append({"file": fname, "status": "✓ rendered"})
            
        except Exception as e:
            results.append({"file": fname, "status": f"error: {e}"})
    
    # Refresh outputs
    outputs = get_outputs_for_channel(channel, subfolder)
    
    success_count = sum(1 for r in results if "rendered" in r["status"])
    status = f"✅ Rendered {success_count}/{len(lines)} files to {output_dir}"
    
    return model, status, outputs

def quick_generate(
    model,
    text: str,
    voice: str,
    channel: str,
    progress=gr.Progress()
) -> Tuple[Any, str, Optional[str]]:
    """Quick generate a single line of audio."""
    if model is None:
        return model, "❌ Model not loaded. Please wait.", None
    
    if not text or not text.strip():
        return model, "❌ Please enter text to generate", None
    
    if not voice:
        return model, "❌ Please select a voice", None
    
    presets = get_presets()
    if voice not in presets:
        return model, f"❌ Voice preset '{voice}' not found", None
    
    preset = presets[voice]
    prompt_path = preset.get("prompt", "")
    
    if not prompt_path or not Path(prompt_path).exists():
        return model, f"❌ Voice prompt file not found: {prompt_path}", None
    
    try:
        progress(0.2, desc="Preparing voice...")
        model.prepare_conditionals(
            str(prompt_path),
            exaggeration=preset.get("exaggeration", 0.5)
        )
        
        progress(0.5, desc="Generating audio...")
        wav = model.generate(
            text.strip(),
            exaggeration=preset.get("exaggeration", 0.5),
            cfg_weight=preset.get("cfg_weight", 0.5),
            temperature=preset.get("temperature", 0.8),
        )
        
        progress(0.9, desc="Saving...")
        
        # Determine output directory
        if channel:
            output_dir = OUTPUTS_DIR / channel
        else:
            output_dir = OUTPUTS_DIR / "quick_generate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        import re
        words = re.findall(r"[A-Za-z0-9']+", text)
        slug = "-".join(words[:6]).lower() or "quick"
        
        # Find next index
        existing = [p.stem for p in output_dir.glob("*.wav")]
        max_idx = 0
        for stem in existing:
            m = re.match(r"^(\d{3,})-", stem)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(1)))
                except ValueError:
                    pass
        idx = max_idx + 1
        
        output_file = output_dir / f"{idx:03d}-{slug}.wav"
        
        target_sr = model.sr
        wav_np = wav.squeeze(0).detach().cpu().numpy()
        
        # Resample to 48k
        wav_np = librosa.resample(wav_np, orig_sr=target_sr, target_sr=48000)
        target_sr = 48000
        
        # LUFS normalization
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(target_sr)
            loud = meter.integrated_loudness(wav_np)
            gain_db = -16 - loud
            factor = 10 ** (gain_db / 20.0)
            wav_np = wav_np * factor
        except Exception:
            pass
        
        ta.save(str(output_file), torch.from_numpy(wav_np).unsqueeze(0), target_sr)
        
        return model, f"✅ Generated: {output_file}", str(output_file)
    except Exception as e:
        return model, f"❌ Error: {e}", None

def concatenate_audio(
    channel: str,
    subfolder: str,
    gap_seconds: float,
    lufs_target: float
) -> Tuple[str, Optional[str]]:
    """Concatenate audio files for a channel."""
    if not channel:
        return "❌ Please select a channel", None
    
    input_dir = OUTPUTS_DIR / channel
    if subfolder and subfolder.strip():
        input_dir = input_dir / subfolder.strip()
    
    if not input_dir.exists():
        return f"❌ Output directory not found: {input_dir}", None
    
    output_file = input_dir / "combined.wav"
    
    # Use concat.py via subprocess
    cmd = [
        sys.executable,
        str(BASE_DIR / "tools" / "concat.py"),
        "--input-dir", str(input_dir),
        "--out", str(output_file),
        "--gap-seconds", str(gap_seconds),
        "--target-sr", "48000",
        "--mono",
    ]
    
    if lufs_target != 0:
        cmd += ["--lufs-target", str(lufs_target)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        if result.returncode != 0:
            return f"❌ Concatenation failed: {result.stderr}", None
        
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            return f"✅ Created: {output_file} ({size_mb:.2f} MB)", str(output_file)
        else:
            return "❌ Concatenation completed but output file not found", None
    except Exception as e:
        return f"❌ Error: {e}", None

def refresh_outputs_table(channel: str, subfolder: str) -> List[List[str]]:
    """Refresh outputs table."""
    outputs = get_outputs_for_channel(channel, subfolder)
    return [[o["name"], o["size_kb"], o["modified"]] for o in outputs]

def get_audio_path(channel: str, subfolder: str, filename: str) -> Optional[str]:
    """Get full path to audio file."""
    if not channel or not filename:
        return None
    
    output_dir = OUTPUTS_DIR / channel
    if subfolder and subfolder.strip():
        output_dir = output_dir / subfolder.strip()
    
    audio_path = output_dir / filename
    if audio_path.exists():
        return str(audio_path)
    return None

# ============================================================================
# UI DEFINITION
# ============================================================================

def create_ui():
    """Create the Gradio UI."""
    
    with gr.Blocks(
        title="TTS Production Manager",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            font=gr.themes.GoogleFont("JetBrains Mono")
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .header { 
            text-align: center; 
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        .header h1 { color: #e0e7ff; margin: 0; font-size: 1.8rem; }
        .header p { color: #a5b4fc; margin: 0.5rem 0 0 0; }
        .tab-content { min-height: 500px; }
        """
    ) as demo:
        
        # Model state
        model_state = gr.State(None)
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎙️ TTS Production Manager</h1>
            <p>Create scripts, manage voices, render audio for your YouTube channels & stories</p>
        </div>
        """)
        
        # Status bar
        with gr.Row():
            status_text = gr.Textbox(
                label="Status",
                value="Loading model...",
                interactive=False,
                show_label=False
            )
        
        with gr.Tabs():
            # ================================================================
            # TAB 1: VOICE MANAGEMENT
            # ================================================================
            with gr.TabItem("🎤 Voice Management", elem_classes="tab-content"):
                gr.Markdown("### Manage Voice Presets")
                gr.Markdown("Upload voice audio files and create presets with custom parameters.")
                
                with gr.Row():
                    # Left column: Presets table
                    with gr.Column(scale=2):
                        presets_table = gr.Dataframe(
                            headers=["Name", "Prompt File", "Exists", "Exaggeration", "CFG Weight", "Temperature"],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            value=refresh_presets_table(),
                            interactive=False,
                            label="Voice Presets",
                            wrap=True
                        )
                        refresh_presets_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    # Right column: Upload & Create
                    with gr.Column(scale=1):
                        gr.Markdown("#### Upload Voice File")
                        voice_upload = gr.File(
                            label="Upload .wav file",
                            file_types=[".wav"],
                            type="filepath"
                        )
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        uploaded_path = gr.Textbox(label="Uploaded Path", visible=False)
                        
                        gr.Markdown("#### Create/Edit Preset")
                        preset_name_input = gr.Textbox(label="Preset Name", placeholder="e.g., my-narrator")
                        preset_prompt_input = gr.Textbox(label="Prompt File Path", placeholder="voices/MyVoice.wav")
                        preset_exag_input = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Exaggeration")
                        preset_cfg_input = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="CFG Weight")
                        preset_temp_input = gr.Slider(0.05, 2.0, value=0.8, step=0.05, label="Temperature")
                        
                        with gr.Row():
                            create_preset_btn = gr.Button("➕ Create Preset", variant="primary")
                            update_preset_btn = gr.Button("✏️ Update Preset")
                        
                        preset_action_status = gr.Textbox(label="Action Status", interactive=False)
                        
                        gr.Markdown("#### Delete Preset")
                        delete_preset_dropdown = gr.Dropdown(
                            choices=get_preset_choices(),
                            label="Select Preset to Delete",
                            interactive=True
                        )
                        delete_preset_btn = gr.Button("🗑️ Delete Preset", variant="stop")
                
                # Preview and Test section
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Preview Voice Prompt")
                        preview_preset_dropdown = gr.Dropdown(
                            choices=get_preset_choices(),
                            label="Select Preset",
                            interactive=True
                        )
                        preview_audio = gr.Audio(label="Original Voice Sample", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("#### Test Voice Generation")
                        test_preset_dropdown = gr.Dropdown(
                            choices=get_preset_choices(),
                            label="Select Preset to Test",
                            interactive=True
                        )
                        test_text_input = gr.Textbox(
                            label="Test Text",
                            placeholder="Hello, this is a test of my voice. How does it sound?",
                            lines=2
                        )
                        test_generate_btn = gr.Button("🎙️ Generate Test", variant="primary")
                        test_status = gr.Textbox(label="Test Status", interactive=False)
                        test_audio = gr.Audio(label="Generated Test Audio", interactive=False)
                
                # Event handlers for Voice Management
                voice_upload.change(
                    upload_voice_file,
                    inputs=[voice_upload],
                    outputs=[upload_status, uploaded_path]
                )
                
                uploaded_path.change(
                    lambda x: x,
                    inputs=[uploaded_path],
                    outputs=[preset_prompt_input]
                )
                
                create_preset_btn.click(
                    create_preset,
                    inputs=[preset_name_input, preset_prompt_input, preset_exag_input, preset_cfg_input, preset_temp_input],
                    outputs=[preset_action_status, presets_table]
                ).then(
                    lambda: get_preset_choices(),
                    outputs=[delete_preset_dropdown]
                ).then(
                    lambda: get_preset_choices(),
                    outputs=[preview_preset_dropdown]
                ).then(
                    lambda: get_preset_choices(),
                    outputs=[test_preset_dropdown]
                )
                
                update_preset_btn.click(
                    update_preset,
                    inputs=[preset_name_input, preset_prompt_input, preset_exag_input, preset_cfg_input, preset_temp_input],
                    outputs=[preset_action_status, presets_table]
                )
                
                delete_preset_btn.click(
                    delete_preset,
                    inputs=[delete_preset_dropdown],
                    outputs=[preset_action_status, presets_table, delete_preset_dropdown]
                ).then(
                    lambda: get_preset_choices(),
                    outputs=[preview_preset_dropdown]
                ).then(
                    lambda: get_preset_choices(),
                    outputs=[test_preset_dropdown]
                )
                
                refresh_presets_btn.click(
                    refresh_presets_table,
                    outputs=[presets_table]
                )
                
                preview_preset_dropdown.change(
                    get_voice_audio_path,
                    inputs=[preview_preset_dropdown],
                    outputs=[preview_audio]
                )
                
                test_generate_btn.click(
                    test_voice_generation,
                    inputs=[model_state, test_preset_dropdown, test_text_input],
                    outputs=[model_state, test_status, test_audio]
                )
            
            # ================================================================
            # TAB 2: CHANNEL MANAGEMENT
            # ================================================================
            with gr.TabItem("📺 Channel Management", elem_classes="tab-content"):
                gr.Markdown("### Manage Channels")
                gr.Markdown("Create channels and assign voice presets to them.")
                
                with gr.Row():
                    # Left column: Channels table
                    with gr.Column(scale=2):
                        channels_table = gr.Dataframe(
                            headers=["Channel", "Assigned Voices", "Has Outputs"],
                            datatype=["str", "str", "str"],
                            value=refresh_channels_table(),
                            interactive=False,
                            label="Channels",
                            wrap=True
                        )
                        refresh_channels_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    # Right column: Create & Edit
                    with gr.Column(scale=1):
                        gr.Markdown("#### Create New Channel")
                        channel_name_input = gr.Textbox(label="Channel Name", placeholder="e.g., MyNewChannel")
                        channel_voices_input = gr.Textbox(
                            label="Assigned Voices (comma-separated)",
                            placeholder="wired-eliv3, tinytales-female"
                        )
                        create_channel_btn = gr.Button("➕ Create Channel", variant="primary")
                        channel_action_status = gr.Textbox(label="Action Status", interactive=False)
                        
                        gr.Markdown("#### Edit Channel Voices")
                        edit_channel_dropdown = gr.Dropdown(
                            choices=get_channel_choices(),
                            label="Select Channel",
                            interactive=True
                        )
                        edit_channel_voices = gr.Textbox(
                            label="Voices (comma-separated)",
                            placeholder="voice1, voice2"
                        )
                        update_channel_btn = gr.Button("✏️ Update Voices")
                        
                        gr.Markdown("#### Delete Channel")
                        delete_channel_dropdown = gr.Dropdown(
                            choices=get_channel_choices(),
                            label="Select Channel to Delete",
                            interactive=True
                        )
                        delete_channel_btn = gr.Button("🗑️ Delete Channel", variant="stop")
                
                # Event handlers for Channel Management
                create_channel_btn.click(
                    create_channel,
                    inputs=[channel_name_input, channel_voices_input],
                    outputs=[channel_action_status, channels_table, edit_channel_dropdown]
                ).then(
                    lambda: get_channel_choices(),
                    outputs=[delete_channel_dropdown]
                )
                
                edit_channel_dropdown.change(
                    load_channel_for_edit,
                    inputs=[edit_channel_dropdown],
                    outputs=[edit_channel_voices]
                )
                
                update_channel_btn.click(
                    update_channel_voices,
                    inputs=[edit_channel_dropdown, edit_channel_voices],
                    outputs=[channel_action_status, channels_table]
                )
                
                delete_channel_btn.click(
                    delete_channel,
                    inputs=[delete_channel_dropdown],
                    outputs=[channel_action_status, channels_table, delete_channel_dropdown]
                ).then(
                    lambda: get_channel_choices(),
                    outputs=[edit_channel_dropdown]
                )
                
                refresh_channels_btn.click(
                    refresh_channels_table,
                    outputs=[channels_table]
                )
            
            # ================================================================
            # TAB 3: SCRIPT EDITOR
            # ================================================================
            with gr.TabItem("📝 Script Editor", elem_classes="tab-content"):
                gr.Markdown("### Create & Edit Scripts")
                gr.Markdown("Create multi-line scripts with voice assignments for each line.")
                
                # Get initial channel and its scripts for Script Editor
                se_channels = get_channel_choices()
                se_channel = se_channels[0] if se_channels else None
                se_scripts = get_script_choices(se_channel) if se_channel else []
                
                with gr.Row():
                    script_channel_dropdown = gr.Dropdown(
                        choices=se_channels,
                        value=se_channel,
                        label="Channel",
                        interactive=True
                    )
                    script_file_dropdown = gr.Dropdown(
                        choices=se_scripts,
                        label="Load Existing Script",
                        interactive=True
                    )
                    script_name_input = gr.Textbox(
                        label="Script Name",
                        placeholder="my_video_script.csv"
                    )
                
                script_status = gr.Textbox(label="Status", interactive=False)
                
                script_dataframe = gr.Dataframe(
                    headers=["Text", "Filename", "Voice", "Exaggeration", "CFG Weight", "Temperature"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=[create_empty_script_row()],
                    interactive=True,
                    label="Script Lines",
                    col_count=(6, "fixed"),
                    wrap=True,
                    row_count=(5, "dynamic")
                )
                
                with gr.Row():
                    add_row_btn = gr.Button("➕ Add Row")
                    save_script_btn = gr.Button("💾 Save Script", variant="primary")
                
                gr.Markdown("""
                **Tips:**
                - Leave Filename blank to auto-generate from text
                - Leave Voice blank to use channel default
                - Use preset names from Voice Management for the Voice column
                """)
                
                # Event handlers for Script Editor
                script_channel_dropdown.change(
                    get_script_choices,
                    inputs=[script_channel_dropdown],
                    outputs=[script_file_dropdown]
                )
                
                script_file_dropdown.change(
                    refresh_script_data,
                    inputs=[script_channel_dropdown, script_file_dropdown],
                    outputs=[script_dataframe, script_status]
                ).then(
                    lambda name: name if name else "",
                    inputs=[script_file_dropdown],
                    outputs=[script_name_input]
                )
                
                add_row_btn.click(
                    add_script_row,
                    inputs=[script_dataframe],
                    outputs=[script_dataframe]
                )
                
                save_script_btn.click(
                    save_script,
                    inputs=[script_channel_dropdown, script_name_input, script_dataframe],
                    outputs=[script_status]
                ).then(
                    get_script_choices,
                    inputs=[script_channel_dropdown],
                    outputs=[script_file_dropdown]
                )
            
            # ================================================================
            # TAB 4: RENDER & OUTPUT
            # ================================================================
            with gr.TabItem("🔊 Render & Output", elem_classes="tab-content"):
                gr.Markdown("### Render Scripts to Audio")
                
                # Quick Generate Section
                with gr.Accordion("⚡ Quick Generate (Single Line)", open=False):
                    gr.Markdown("Generate a single audio line without creating a script.")
                    with gr.Row():
                        with gr.Column(scale=3):
                            quick_text = gr.Textbox(
                                label="Text to Generate",
                                placeholder="Enter text to speak...",
                                lines=2
                            )
                        with gr.Column(scale=1):
                            quick_voice = gr.Dropdown(
                                choices=get_preset_choices(),
                                label="Voice",
                                interactive=True
                            )
                            quick_channel = gr.Dropdown(
                                choices=get_channel_choices(),
                                label="Save to Channel",
                                interactive=True
                            )
                    with gr.Row():
                        quick_generate_btn = gr.Button("🎙️ Quick Generate", variant="primary")
                    quick_status = gr.Textbox(label="Status", interactive=False)
                    quick_audio = gr.Audio(label="Generated Audio", interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Render Settings")
                        # Get initial channel and its scripts
                        initial_channels = get_channel_choices()
                        initial_channel = initial_channels[0] if initial_channels else None
                        initial_scripts = get_script_choices(initial_channel) if initial_channel else []
                        
                        render_channel_dropdown = gr.Dropdown(
                            choices=initial_channels,
                            value=initial_channel,
                            label="Channel",
                            interactive=True
                        )
                        render_script_dropdown = gr.Dropdown(
                            choices=initial_scripts,
                            value=initial_scripts[0] if initial_scripts else None,
                            label="Script",
                            interactive=True
                        )
                        render_subfolder = gr.Textbox(
                            label="Subfolder (optional)",
                            placeholder="e.g., CCNA/Lesson01"
                        )
                        render_voice_override = gr.Dropdown(
                            choices=[""] + get_preset_choices(),
                            label="Voice Override (optional)",
                            value="",
                            interactive=True
                        )
                        
                        with gr.Row():
                            render_to_48k = gr.Checkbox(label="Resample to 48kHz", value=True)
                            render_overwrite = gr.Checkbox(label="Overwrite existing", value=False)
                        
                        render_lufs = gr.Slider(-30, 0, value=-16, step=1, label="LUFS Target (0 = disable)")
                        
                        render_btn = gr.Button("🎙️ Render Script", variant="primary", size="lg")
                        render_status = gr.Textbox(label="Render Status", interactive=False)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Output Files")
                        outputs_table = gr.Dataframe(
                            headers=["Filename", "Size (KB)", "Modified"],
                            datatype=["str", "number", "str"],
                            value=[],
                            interactive=False,
                            label="Rendered Files"
                        )
                        
                        with gr.Row():
                            refresh_outputs_btn = gr.Button("🔄 Refresh")
                            output_subfolder = gr.Textbox(
                                label="Subfolder Filter",
                                placeholder="Leave blank for main folder"
                            )
                        
                        gr.Markdown("#### Play Audio")
                        with gr.Row():
                            play_filename = gr.Textbox(label="Filename to Play")
                            play_btn = gr.Button("▶️ Play")
                        output_audio = gr.Audio(label="Audio Preview", interactive=False)
                        
                        gr.Markdown("#### Concatenate")
                        with gr.Row():
                            concat_gap = gr.Slider(0, 3, value=0.5, step=0.1, label="Gap (seconds)")
                            concat_lufs = gr.Slider(-30, 0, value=-16, step=1, label="LUFS Target")
                        concat_btn = gr.Button("🔗 Concatenate All", variant="secondary")
                        concat_status = gr.Textbox(label="Concatenation Status", interactive=False)
                        concat_audio = gr.Audio(label="Combined Audio", interactive=False)
                
                # Event handlers for Quick Generate
                quick_generate_btn.click(
                    quick_generate,
                    inputs=[model_state, quick_text, quick_voice, quick_channel],
                    outputs=[model_state, quick_status, quick_audio]
                )
                
                # Event handlers for Render & Output
                render_channel_dropdown.change(
                    get_script_choices,
                    inputs=[render_channel_dropdown],
                    outputs=[render_script_dropdown]
                ).then(
                    lambda ch, sf: refresh_outputs_table(ch, sf),
                    inputs=[render_channel_dropdown, output_subfolder],
                    outputs=[outputs_table]
                )
                
                render_btn.click(
                    render_script,
                    inputs=[
                        model_state,
                        render_channel_dropdown,
                        render_script_dropdown,
                        render_voice_override,
                        render_to_48k,
                        render_lufs,
                        render_overwrite,
                        render_subfolder
                    ],
                    outputs=[model_state, render_status, outputs_table]
                )
                
                refresh_outputs_btn.click(
                    refresh_outputs_table,
                    inputs=[render_channel_dropdown, output_subfolder],
                    outputs=[outputs_table]
                )
                
                play_btn.click(
                    get_audio_path,
                    inputs=[render_channel_dropdown, output_subfolder, play_filename],
                    outputs=[output_audio]
                )
                
                concat_btn.click(
                    concatenate_audio,
                    inputs=[render_channel_dropdown, output_subfolder, concat_gap, concat_lufs],
                    outputs=[concat_status, concat_audio]
                )
        
        # Load model on startup
        def init_model():
            model = load_model()
            return model, "✅ Model loaded and ready!"
        
        demo.load(
            init_model,
            outputs=[model_state, status_text]
        )
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()
    demo.queue(
        max_size=20,
        default_concurrency_limit=1
    ).launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
