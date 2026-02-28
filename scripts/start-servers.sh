#!/usr/bin/env bash
# start-servers.sh — VE-RAG-System unified startup
#
# Usage:
#   ./scripts/start-servers.sh                  # build frontend + start backend
#   ./scripts/start-servers.sh backend          # skip frontend build
#   ./scripts/start-servers.sh dev              # start frontend dev server (:5173)
#   ./scripts/start-servers.sh all              # explicit alias for default
#   ./scripts/start-servers.sh --reset          # drop & recreate postgres volume, then start
#   ./scripts/start-servers.sh --reload         # enable uvicorn --reload (hot reload)
#   ./scripts/start-servers.sh backend --reload # combine mode + flags
#
# Environment:
#   Reads .env from project root.
#   Set DOCKER_INFRA=false to skip docker-compose (e.g. Spark production).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
MODE="all"
RESET=false
RELOAD=false

for arg in "$@"; do
    case $arg in
        --reset)  RESET=true ;;
        --reload) RELOAD=true ;;
        backend|dev|all) MODE="$arg" ;;
        *)
            echo "Usage: $0 [backend|dev|all] [--reset] [--reload]"
            exit 1
            ;;
    esac
done

# ── Load env ──────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a && source .env && set +a
    echo "✓ Loaded .env"
else
    echo "Warning: No .env file found. Using defaults."
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Prevent Tesseract OpenMP threads from stacking with ARQ job concurrency.
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-1}"

# Ensure CUDA libraries are discoverable (DGX Spark).
if [ -d "/usr/local/cuda/targets/sbsa-linux/lib" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
fi

# ── Docker infrastructure ─────────────────────────────────────────────────────
DOCKER_INFRA="${DOCKER_INFRA:-true}"

if [ "$DOCKER_INFRA" = "true" ] && [ -f "docker-compose.dev.yml" ]; then
    if [ "$RESET" = true ]; then
        echo "⚠  Resetting postgres volume..."
        docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
    fi

    echo "▶ Starting postgres + redis..."
    docker-compose -f docker-compose.dev.yml up -d

    echo "⏳ Waiting for postgres to be healthy..."
    ATTEMPTS=0
    until docker exec vaultiq-dev-postgres pg_isready -U vaultiq -d vaultiq -q 2>/dev/null; do
        ATTEMPTS=$((ATTEMPTS + 1))
        if [ "$ATTEMPTS" -ge 30 ]; then
            echo "ERROR: Postgres did not become healthy in time."
            exit 1
        fi
        sleep 1
    done
    echo "✓ Postgres ready"
else
    echo "⚠ DOCKER_INFRA=false — skipping docker-compose (assuming external infra)"
fi

# ── Kill stale ARQ CLI workers ────────────────────────────────────────────────
# The embedded ARQ worker inside FastAPI replaces the standalone process.
# A leftover `arq` CLI process competes for Redis jobs using outdated code.
stale_arq_pids=$(pgrep -f 'arq ai_ready_rag' 2>/dev/null || true)
if [ -n "$stale_arq_pids" ]; then
    echo "Killing stale ARQ CLI worker(s): $stale_arq_pids"
    kill $stale_arq_pids 2>/dev/null || true
    sleep 1
fi

# ── Tenant config ─────────────────────────────────────────────────────────────
TENANT_DIR="tenant-instances/default"
TENANT_TEMPLATE="tenant-instances/tenant.dev-default.json"
if [ -f "$TENANT_TEMPLATE" ] && [ ! -f "$TENANT_DIR/tenant.json" ]; then
    mkdir -p "$TENANT_DIR"
    cp "$TENANT_TEMPLATE" "$TENANT_DIR/tenant.json"
    echo "✓ Created $TENANT_DIR/tenant.json from template"
fi

# ── Alembic migrations ────────────────────────────────────────────────────────
echo "▶ Running Alembic migrations..."
alembic upgrade head
echo "✓ Migrations complete"

# ── Init DB / admin user ──────────────────────────────────────────────────────
python << "PYTHON"
import os
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import User
from ai_ready_rag.core.security import hash_password

init_db()

db = SessionLocal()
try:
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@test.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "npassword2002!")
    admin_name = os.environ.get("ADMIN_DISPLAY_NAME", "Administrator")

    admin = db.query(User).filter(User.email == admin_email).first()
    if not admin:
        admin = User(
            email=admin_email,
            display_name=admin_name,
            password_hash=hash_password(admin_password),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print(f"Admin user created: {admin_email}")
    else:
        print(f"Admin user exists: {admin_email}")
finally:
    db.close()
PYTHON

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="
preflight_failed=0

# Redis
if redis-cli ping &>/dev/null 2>&1; then
    echo "  ✓ Redis ........ OK"
elif timeout 2 bash -c 'echo PING | nc -q1 localhost 6379 2>/dev/null' | grep -q PONG 2>/dev/null; then
    echo "  ✓ Redis ........ OK"
else
    echo "  ⚠ Redis ........ UNREACHABLE (ARQ worker will use BackgroundTasks fallback)"
fi

# Postgres
DB_URL="${DATABASE_URL:-}"
if [ -n "$DB_URL" ] && echo "$DB_URL" | grep -q "postgresql"; then
    if docker exec vaultiq-dev-postgres pg_isready -U vaultiq -d vaultiq -q 2>/dev/null; then
        echo "  ✓ Postgres ..... OK"
    else
        echo "  ✗ Postgres ..... NOT RESPONDING"
        preflight_failed=1
    fi
fi

# Ollama
if curl -sf "http://localhost:11434/api/version" &>/dev/null; then
    echo "  ✓ Ollama ....... OK"
    EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
    CHAT_MODEL="${CHAT_MODEL:-llama3.2:latest}"
    for model in "$EMBEDDING_MODEL" "$CHAT_MODEL"; do
        if ollama list 2>/dev/null | grep -q "^${model}"; then
            echo "    ✓ $model"
        else
            echo "    ⚠ $model not pulled (run: ollama pull $model)"
        fi
    done
else
    echo "  ✗ Ollama ....... NOT RESPONDING (is ollama running?)"
    preflight_failed=1
fi

if [ "$preflight_failed" -eq 1 ]; then
    echo ""
    echo "ERROR: Required services are not running. Fix above and retry."
    exit 1
fi
echo ""

# ── Print banner ──────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  VaultIQ — Starting"
echo "  Mode:     $MODE$([ "$RELOAD" = true ] && echo " (--reload)" || true)"
echo "  Backend:  http://localhost:8502"
echo "  Swagger:  http://localhost:8502/api/docs"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Frontend helpers ──────────────────────────────────────────────────────────
build_frontend() {
    echo "=== Building frontend ==="
    cd "$PROJECT_DIR/frontend"
    npm run build
    cd "$PROJECT_DIR"
    echo "Frontend build complete"
}

start_dev_frontend() {
    echo "=== Starting frontend dev server (port 5173) ==="
    cd "$PROJECT_DIR/frontend"
    npm run dev &
    echo "Frontend PID: $!"
    cd "$PROJECT_DIR"
}

# ── Launch ────────────────────────────────────────────────────────────────────
UVICORN_CMD="python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port 8502"
if [ "$RELOAD" = true ]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
fi

case "$MODE" in
    backend)
        exec $UVICORN_CMD
        ;;
    dev)
        start_dev_frontend
        exec $UVICORN_CMD
        ;;
    all)
        build_frontend
        exec $UVICORN_CMD
        ;;
esac
