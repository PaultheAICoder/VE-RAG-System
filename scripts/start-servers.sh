#!/bin/bash
# Start VE-RAG-System servers
# Usage: ./scripts/start-servers.sh [backend|dev|all]
#   backend  - Start only backend (port 8502), assumes frontend already built
#   dev      - Start frontend dev server (port 5173) for local development
#   all      - Build frontend + start backend (default, recommended for production)

MODE="${1:-all}"

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Export env vars from .env (skip comments and empty lines)
if [ -f ".env" ]; then
    export $(grep -v "^#" .env | grep -v "^$" | xargs)
else
    echo "Warning: No .env file found. Using defaults."
fi

# Prevent Tesseract OpenMP threads from stacking with ARQ job concurrency.
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-1}"

echo "=== VE-RAG-System Startup ==="
echo "Directory: $PROJECT_DIR"
echo "Mode:      $MODE"

# Pre-flight health checks
preflight_checks() {
    echo "=== Pre-flight checks ==="
    local failed=0

    # Redis
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping &>/dev/null; then
            echo "  ✓ Redis ........ OK"
        else
            echo "  ✗ Redis ........ NOT RESPONDING (is redis-server running?)"
            failed=1
        fi
    elif curl -s "http://localhost:6379" &>/dev/null || timeout 2 bash -c 'echo PING | nc -q1 localhost 6379' 2>/dev/null | grep -q PONG; then
        echo "  ✓ Redis ........ OK"
    else
        echo "  ⚠ Redis ........ UNREACHABLE (ARQ worker will be disabled, using BackgroundTasks fallback)"
    fi

    # Qdrant
    if curl -sf "http://localhost:6333/healthz" &>/dev/null; then
        echo "  ✓ Qdrant ....... OK"
    else
        echo "  ✗ Qdrant ....... NOT RESPONDING (is qdrant running?)"
        echo "    Try: docker start qdrant"
        failed=1
    fi

    # Ollama
    if curl -sf "http://localhost:11434/api/version" &>/dev/null; then
        echo "  ✓ Ollama ....... OK"
        # Check required models
        for model in nomic-embed-text qwen3:8b; do
            if ollama list 2>/dev/null | grep -q "^${model}"; then
                echo "    ✓ $model"
            else
                echo "    ⚠ $model not found (run: ollama pull $model)"
            fi
        done
    else
        echo "  ✗ Ollama ....... NOT RESPONDING (is ollama running?)"
        echo "    Try: ollama serve"
        failed=1
    fi

    if [ "$failed" -eq 1 ]; then
        echo ""
        echo "ERROR: Required services are not running. Fix the issues above and try again."
        exit 1
    fi
    echo ""
}

# Initialize database and create admin/tags
init_database() {
    echo "=== Initializing database ==="
    python << "PYTHON"
import os
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import User, Tag
from ai_ready_rag.core.security import hash_password

init_db()

db = SessionLocal()
try:
    # Create admin user
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@test.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "npassword")
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

    # Create default tags
    default_tags = [
        {"name": "hr", "display_name": "HR", "color": "#10B981"},
        {"name": "it", "display_name": "IT", "color": "#3B82F6"},
        {"name": "legal", "display_name": "Legal", "color": "#8B5CF6"},
        {"name": "finance", "display_name": "Finance", "color": "#F59E0B"},
    ]

    for tag_data in default_tags:
        tag = db.query(Tag).filter(Tag.name == tag_data["name"]).first()
        if not tag:
            tag = Tag(**tag_data)
            db.add(tag)
            print(f"Tag created: {tag_data['display_name']}")

    db.commit()
finally:
    db.close()
PYTHON
}

build_frontend() {
    echo "=== Building frontend ==="
    cd "$PROJECT_DIR/frontend"
    npm run build
    cd "$PROJECT_DIR"
    echo "Frontend build complete"
}

start_dev_server() {
    echo "=== Starting frontend dev server (port 5173) ==="
    cd "$PROJECT_DIR/frontend"
    npm run dev &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    cd "$PROJECT_DIR"
}

start_backend() {
    echo "=== Starting backend server (port 8502) ==="
    exec python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port 8502
}

# Run based on mode
case "$MODE" in
    backend)
        preflight_checks
        init_database
        start_backend
        ;;
    dev)
        start_dev_server
        wait
        ;;
    all)
        preflight_checks
        init_database
        build_frontend
        start_backend
        ;;
    *)
        echo "Usage: $0 [backend|dev|all]"
        exit 1
        ;;
esac
