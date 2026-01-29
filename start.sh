#!/bin/bash
# Start the AI Ready RAG FastAPI server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  AI Ready RAG - FastAPI Server"
echo "=========================================="

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}[OK]${NC} Virtual environment activated"
else
    echo -e "${RED}[ERROR]${NC} Virtual environment not found. Run: python -m venv .venv && pip install -r requirements-wsl.txt"
    exit 1
fi

# Load .env if exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}[OK]${NC} Loaded .env file"
else
    echo -e "${YELLOW}[WARN]${NC} No .env file found, using defaults"
fi

# Set defaults from environment or use fallbacks
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
export ADMIN_EMAIL="${ADMIN_EMAIL:-admin@test.com}"
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-npassword}"
export ADMIN_DISPLAY_NAME="${ADMIN_DISPLAY_NAME:-Administrator}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Check Ollama availability
echo -n "Checking Ollama at $OLLAMA_BASE_URL... "
if curl -s --max-time 2 "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${YELLOW}[WARN]${NC} Not reachable (RAG features will be limited)"
fi

# Check Qdrant availability
echo -n "Checking Qdrant at $QDRANT_URL... "
if curl -s --max-time 2 "$QDRANT_URL/collections" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC}"
else
    echo -e "${YELLOW}[WARN]${NC} Not reachable (vector search disabled)"
fi

# Ensure data directory exists
mkdir -p data/uploads

# Function to seed admin user
seed_admin() {
    echo -n "Seeding admin user... "

    # Wait for server to be ready
    for i in {1..10}; do
        if curl -s --max-time 1 "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done

    # Try to create admin user (will fail silently if already exists)
    RESPONSE=$(curl -s -X POST "http://localhost:$PORT/api/auth/setup" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$ADMIN_EMAIL\", \"password\": \"$ADMIN_PASSWORD\", \"display_name\": \"$ADMIN_DISPLAY_NAME\"}" \
        2>/dev/null)

    if echo "$RESPONSE" | grep -q '"email"'; then
        echo -e "${GREEN}[OK]${NC} Created: $ADMIN_EMAIL"
    elif echo "$RESPONSE" | grep -q 'already completed'; then
        echo -e "${GREEN}[OK]${NC} Already exists"
    else
        echo -e "${YELLOW}[SKIP]${NC} Could not seed admin"
    fi
}

# Function to seed default tags
seed_tags() {
    echo "Seeding default tags..."

    # Login to get token (API expects JSON, not form data)
    TOKEN=$(curl -s -X POST "http://localhost:$PORT/api/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$ADMIN_EMAIL\", \"password\": \"$ADMIN_PASSWORD\"}" \
        2>/dev/null | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$TOKEN" ]; then
        echo -e "  ${YELLOW}[SKIP]${NC} Could not get auth token"
        return
    fi

    # Seed tags
    TAGS='[
        {"name": "hr", "display_name": "HR", "color": "#10B981"},
        {"name": "it", "display_name": "IT", "color": "#3B82F6"},
        {"name": "legal", "display_name": "Legal", "color": "#8B5CF6"},
        {"name": "finance", "display_name": "Finance", "color": "#F59E0B"}
    ]'

    echo "$TAGS" | python3 -c "
import json, sys
tags = json.load(sys.stdin)
for tag in tags:
    print(json.dumps(tag))
" | while read -r tag; do
        NAME=$(echo "$tag" | python3 -c "import json,sys; print(json.load(sys.stdin)['display_name'])")
        RESPONSE=$(curl -s -X POST "http://localhost:$PORT/api/tags" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "$tag" 2>/dev/null)

        if echo "$RESPONSE" | grep -q '"id"'; then
            echo -e "  ${GREEN}[OK]${NC} Created tag: $NAME"
        elif echo "$RESPONSE" | grep -q 'already exists'; then
            echo -e "  ${GREEN}[OK]${NC} Tag exists: $NAME"
        else
            echo -e "  ${YELLOW}[SKIP]${NC} $NAME"
        fi
    done
}

echo ""
echo "Configuration:"
echo "  Ollama URL:  $OLLAMA_BASE_URL"
echo "  Qdrant URL:  $QDRANT_URL"
echo "  Admin Email: $ADMIN_EMAIL"
echo "  Server:      http://$HOST:$PORT"
echo "  Gradio UI:   http://$HOST:$PORT/app"
echo ""

# Seed admin and tags in background after server starts
(sleep 2 && seed_admin && seed_tags) &

# Start FastAPI server
echo "Starting FastAPI server..."
python -m uvicorn ai_ready_rag.main:app --host "$HOST" --port "$PORT" --reload
