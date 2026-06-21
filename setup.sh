#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  VoiceBridge – One-time Setup
# ─────────────────────────────────────────────────────────
set -e

echo "=== VoiceBridge Setup ==="

# Check Python
python3 --version || { echo "Python 3 not found"; exit 1; }

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo ""
  echo "Installing ffmpeg…"
  if command -v apt-get &>/dev/null; then
    sudo apt-get install -y ffmpeg
  elif command -v brew &>/dev/null; then
    brew install ffmpeg
  else
    echo "Please install ffmpeg manually: https://ffmpeg.org/download.html"
    exit 1
  fi
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Creating virtual environment at $VENV_DIR…"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing Python dependencies…"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/backend/requirements.txt"

echo ""
echo "✅  Setup complete!"
echo "   Run:  ./run.sh"
echo "   Then open: http://localhost:8000"
