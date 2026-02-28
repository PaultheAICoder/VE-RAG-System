#!/usr/bin/env bash
# stop-servers.sh — VE-RAG-System unified shutdown
#
# Usage:
#   ./scripts/stop-servers.sh           # stop app processes only
#   ./scripts/stop-servers.sh --infra   # also stop Docker postgres + redis

set -euo pipefail

STOP_INFRA=false
for arg in "$@"; do
    case $arg in
        --infra) STOP_INFRA=true ;;
    esac
done

echo "Stopping VE-RAG-System servers..."

# Kill backend (port 8502)
if lsof -ti:8502 > /dev/null 2>&1; then
    lsof -ti:8502 | xargs kill -9
    echo "  ✓ Backend (port 8502) stopped"
else
    echo "  - Backend (port 8502) not running"
fi

# Kill frontend dev server (port 5173)
if lsof -ti:5173 > /dev/null 2>&1; then
    lsof -ti:5173 | xargs kill -9
    echo "  ✓ Frontend dev server (port 5173) stopped"
else
    echo "  - Frontend dev server (port 5173) not running"
fi

# Kill any stale ARQ CLI workers
stale_arq_pids=$(pgrep -f 'arq ai_ready_rag' 2>/dev/null || true)
if [ -n "$stale_arq_pids" ]; then
    kill $stale_arq_pids 2>/dev/null || true
    echo "  ✓ Stale ARQ worker(s) stopped"
fi

# Stop Docker infra (opt-in)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$STOP_INFRA" = true ] && [ -f "$PROJECT_DIR/docker-compose.dev.yml" ]; then
    echo "▶ Stopping Docker infra (postgres + redis)..."
    docker-compose -f "$PROJECT_DIR/docker-compose.dev.yml" stop
    echo "  ✓ Docker infra stopped"
fi

echo "Done"
