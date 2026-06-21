#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  VoiceBridge – Development Start (without Docker)
# ─────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/.venv"
BACKEND="$PROJECT_DIR/backend"
VENV_PY="$VENV/bin/python"

# 1. Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "Installing ffmpeg…"
  if command -v apt-get &>/dev/null; then sudo apt-get install -y ffmpeg;
  elif command -v brew &>/dev/null; then brew install ffmpeg;
  else echo "Please install ffmpeg manually"; exit 1; fi
fi

# 2. Create venv
[ -d "$VENV" ] || python3 -m venv "$VENV"

# 3. Install deps
"$VENV_PY" -m pip install -q --upgrade pip
"$VENV_PY" -m pip install -q -r "$BACKEND/requirements.txt"

# 4. Export env
export WHISPER_MODEL="${WHISPER_MODEL:-medium}"
export FINETUNED_MODEL_DIR="${FINETUNED_MODEL_DIR:-}"
export XTTS_MODEL_DIR="${XTTS_MODEL_DIR:-}"
export MMS_MODEL_DIR="${MMS_MODEL_DIR:-}"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  VoiceBridge  →  http://localhost:8000  ║"
echo "╚══════════════════════════════════════╝"
echo ""

cd "$BACKEND"
# Use modular app/main.py (falls back to legacy main.py if app/ not present)
if [ -f "$BACKEND/app/main.py" ]; then
  "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
  "$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi
