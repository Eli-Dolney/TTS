#!/bin/bash
# Render all test scripts for all channels
# This creates test videos showcasing each voice

VENV=../.venv/bin
SCRIPT_DIR=$(dirname "$0")

echo "🎬 Rendering Test Videos for All Channels"
echo "=========================================="

# Channels with multiple voices
echo ""
echo "📺 TinyTalesTV (3 voices)"
$VENV/python ../tools/channel_manager.py render TinyTalesTV \
  --script "$SCRIPT_DIR/TinyTalesTV/test_all_voices.csv" \
  --subfolder test_all_voices \
  --to-48k --lufs-target -16

echo ""
echo "📺 Other Channel (3 voices)"
$VENV/python ../tools/channel_manager.py render Other \
  --script "$SCRIPT_DIR/Other/test_all_voices.csv" \
  --subfolder test_all_voices \
  --to-48k --lufs-target -16

# Single voice channels
echo ""
echo "📺 WiredWorkshop"
$VENV/python ../tools/channel_manager.py render WiredWorkshop \
  --script "$SCRIPT_DIR/WiredWorkshop/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 WiredToWork"
$VENV/python ../tools/channel_manager.py render WiredToWork \
  --script "$SCRIPT_DIR/WiredToWork/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 LearningTheWires"
$VENV/python ../tools/channel_manager.py render LearningTheWires \
  --script "$SCRIPT_DIR/LearningTheWires/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 ViceCityVault"
$VENV/python ../tools/channel_manager.py render ViceCityVault \
  --script "$SCRIPT_DIR/ViceCityVault/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 NeuralWires"
$VENV/python ../tools/channel_manager.py render NeuralWires \
  --script "$SCRIPT_DIR/NeuralWires/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 EliDolney"
$VENV/python ../tools/channel_manager.py render EliDolney \
  --script "$SCRIPT_DIR/EliDolney/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 LotsOfErrors"
$VENV/python ../tools/channel_manager.py render LotsOfErrors \
  --script "$SCRIPT_DIR/LotsOfErrors/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "📺 FomoFactory"
$VENV/python ../tools/channel_manager.py render FomoFactory \
  --script "$SCRIPT_DIR/FomoFactory/test_voice.csv" \
  --subfolder test_voice \
  --to-48k --lufs-target -16

echo ""
echo "✅ All test videos rendered!"
echo "Check outputs/[ChannelName]/test_voice/ or outputs/[ChannelName]/test_all_voices/"

