#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  VoiceBridge – Docker Compose Launcher
# ─────────────────────────────────────────────────────────────────────
set -e

ACTION="${1:-up}"

case "$ACTION" in
  up)
    echo "Building and starting VoiceBridge…"
    docker compose up --build -d
    echo ""
    echo "✅  VoiceBridge is running at: http://localhost"
    echo "   Logs: docker compose logs -f"
    echo "   Stop: ./docker-start.sh down"
    ;;
  down)
    docker compose down
    echo "✅  Stopped"
    ;;
  logs)
    docker compose logs -f
    ;;
  restart)
    docker compose restart backend
    ;;
  build)
    docker compose build --no-cache
    ;;
  *)
    echo "Usage: $0 [up|down|logs|restart|build]"
    exit 1
    ;;
esac
