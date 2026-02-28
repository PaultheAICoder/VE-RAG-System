#!/usr/bin/env bash
# start-dev-full.sh — Full-feature dev environment startup
# Starts postgres+redis via Docker, runs Alembic migrations, then launches the app.
#
# Usage:
#   bash scripts/start-dev-full.sh
#   bash scripts/start-dev-full.sh --reset   # drop & recreate postgres volume

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
RESET=false
for arg in "$@"; do
  case $arg in
    --reset) RESET=true ;;
  esac
done

# ── Load env ──────────────────────────────────────────────────────────────────
if [ ! -f ".env.dev-full" ]; then
  echo "ERROR: .env.dev-full not found. Run from project root."
  exit 1
fi
set -a && source .env.dev-full && set +a
echo "✓ Loaded .env.dev-full"

# ── Docker infrastructure ─────────────────────────────────────────────────────
if [ "$RESET" = true ]; then
  echo "⚠  Resetting postgres volume..."
  docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
fi

echo "▶ Starting postgres + redis..."
docker compose -f docker-compose.dev.yml up -d

echo "⏳ Waiting for postgres to be healthy..."
ATTEMPTS=0
until docker exec vaultiq-dev-postgres pg_isready -U vaultiq -d vaultiq -q 2>/dev/null; do
  ATTEMPTS=$((ATTEMPTS+1))
  if [ "$ATTEMPTS" -ge 30 ]; then
    echo "ERROR: postgres did not become healthy in time."
    exit 1
  fi
  sleep 1
done
echo "✓ Postgres ready"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "✓ Virtual environment activated"
fi

# ── Tenant config ─────────────────────────────────────────────────────────────
TENANT_DIR="tenant-instances/default"
TENANT_TEMPLATE="tenant-instances/tenant.dev-default.json"
if [ ! -f "$TENANT_DIR/tenant.json" ]; then
  mkdir -p "$TENANT_DIR"
  cp "$TENANT_TEMPLATE" "$TENANT_DIR/tenant.json"
  echo "✓ Created $TENANT_DIR/tenant.json from template"
fi

# ── Alembic migrations ────────────────────────────────────────────────────────
echo "▶ Running Alembic migrations..."
alembic upgrade head
echo "✓ Migrations complete"

# ── Launch app ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  VaultIQ Dev — Full Feature Mode"
echo "  Backend:  http://localhost:8502"
echo "  Swagger:  http://localhost:8502/api/docs"
echo "  Postgres: localhost:5432 (vaultiq/devpassword)"
echo "  CA module: ENABLED (tenant-instances/default/tenant.json)"
echo "  Claude:   DISABLED (set ANTHROPIC_API_KEY to enable)"
echo "═══════════════════════════════════════════════════"
echo ""

python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port 8502 --reload
