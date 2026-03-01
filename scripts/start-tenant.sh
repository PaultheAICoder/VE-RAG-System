#!/usr/bin/env bash
# start-tenant.sh — Start a named test tenant instance
#
# Usage:
#   ./scripts/start-tenant.sh insurance          # port 8502, vaultiq_insurance db
#   ./scripts/start-tenant.sh lawfirm            # port 8503, vaultiq_lawfirm db
#   ./scripts/start-tenant.sh insurance --reload # hot reload
#
# Prereqs: docker compose -f docker-compose.dev.yml up -d  (postgres + redis)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Args ──────────────────────────────────────────────────────────────────────
TENANT="${1:-}"
RELOAD=false

if [ -z "$TENANT" ]; then
    echo "Usage: $0 <insurance|lawfirm> [--reload]"
    exit 1
fi

for arg in "${@:2}"; do
    case $arg in
        --reload) RELOAD=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: $0 <insurance|lawfirm> [--reload]"
            exit 1
            ;;
    esac
done

# ── Load env file ─────────────────────────────────────────────────────────────
ENV_FILE=".env.${TENANT}"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Available tenants: insurance, lawfirm"
    exit 1
fi

set -a && source "$ENV_FILE" && set +a
echo "✓ Loaded $ENV_FILE"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-1}"

# ── Derive db name from DATABASE_URL ─────────────────────────────────────────
# e.g. postgresql://vaultiq:devpassword@localhost:5432/vaultiq_insurance → vaultiq_insurance
DB_NAME="${DATABASE_URL##*/}"
DB_USER="vaultiq"
DB_PASS="devpassword"
PG_CONTAINER="vaultiq-dev-postgres"

# Helper: run psql inside the postgres docker container
pg_exec() {
    docker exec -e PGPASSWORD="$DB_PASS" "$PG_CONTAINER" \
        psql -U "$DB_USER" "$@"
}

# ── Create database if it doesn't exist ──────────────────────────────────────
echo "▶ Ensuring database '$DB_NAME' exists..."
pg_exec -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" \
    | grep -q 1 || \
pg_exec -d postgres -c "CREATE DATABASE \"${DB_NAME}\";"

# Enable pgvector extension in the tenant database
pg_exec -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" -q
echo "✓ Database ready"

# ── Create data directories ───────────────────────────────────────────────────
mkdir -p "${UPLOAD_DIR}"
mkdir -p "${FORMS_TEMPLATE_STORAGE_PATH}"
mkdir -p "${RAGBENCH_DATA_DIR}"
echo "✓ Data directories ready"

# ── Tenant instance config ────────────────────────────────────────────────────
TENANT_DIR="tenant-instances/${DEFAULT_TENANT_ID}"
TENANT_JSON="${TENANT_DIR}/tenant.json"
if [ ! -f "$TENANT_JSON" ]; then
    mkdir -p "$TENANT_DIR"
    cat > "$TENANT_JSON" <<JSON
{
  "tenant_id": "${DEFAULT_TENANT_ID}",
  "tenant_name": "${ADMIN_DISPLAY_NAME}",
  "feature_flags": {
    "ca_enabled": true,
    "claude_enrichment_enabled": true,
    "structured_query_enabled": true,
    "claude_query_enabled": false
  },
  "ai_models": {
    "enrichment_model": "claude-haiku-4-5-20251001",
    "query_model": "claude-sonnet-4-6",
    "synopsis_model": "claude-sonnet-4-6"
  },
  "cost_controls": {
    "daily_enrichment_cap_usd": 5.0,
    "monthly_enrichment_cap_usd": 50.0
  }
}
JSON
    echo "✓ Created $TENANT_JSON"
fi

# ── Bootstrap schema + seed admin ────────────────────────────────────────────
# init_db() runs create_all (builds all tables from SQLAlchemy models).
# alembic stamp head marks all migrations as applied without re-running them
# (migrations only apply to incremental upgrades on an existing database).
echo "▶ Bootstrapping schema..."
python << "PYTHON"
import os
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import User
from ai_ready_rag.core.security import hash_password

init_db()
print("  Schema created via create_all")

db = SessionLocal()
try:
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@test.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "npassword2002")
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
        print(f"  Admin created: {admin_email}")
    else:
        print(f"  Admin exists:  {admin_email}")
finally:
    db.close()
PYTHON

# Stamp alembic at head so future incremental migrations apply correctly
alembic stamp head
echo "✓ Schema ready"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="

# Postgres
if pg_exec -d "$DB_NAME" -c "SELECT 1" -q &>/dev/null; then
    echo "  ✓ Postgres ..... OK ($DB_NAME)"
else
    echo "  ✗ Postgres ..... FAILED"
    exit 1
fi

# Redis
if redis-cli -u "${REDIS_URL:-redis://localhost:6379}" ping &>/dev/null; then
    echo "  ✓ Redis ........ OK"
else
    echo "  ⚠ Redis ........ UNREACHABLE (ARQ will use BackgroundTasks fallback)"
fi

# Ollama
if curl -sf "${OLLAMA_BASE_URL:-http://localhost:11434}/api/version" &>/dev/null; then
    echo "  ✓ Ollama ....... OK"
else
    echo "  ✗ Ollama ....... NOT RESPONDING"
    exit 1
fi

echo ""

# ── Banner ────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  Tenant:   ${DEFAULT_TENANT_ID}"
echo "  Strategy: ${AUTO_TAGGING_STRATEGY}"
echo "  Database: ${DB_NAME}"
echo "  Backend:  http://localhost:${PORT}"
echo "  Swagger:  http://localhost:${PORT}/api/docs"
echo "  Admin:    ${ADMIN_EMAIL} / ${ADMIN_PASSWORD}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
UVICORN_CMD="python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port ${PORT}"
if [ "$RELOAD" = true ]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
fi

exec $UVICORN_CMD
