#!/bin/bash
# Start VE-RAG-System servers
# Usage: ./scripts/start-servers.sh [backend|frontend|all]
#   backend  - Start only backend (port 8502)
#   frontend - Start only frontend dev server (port 5173)
#   all      - Start both (default)

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

echo "=== VE-RAG-System Startup ==="
echo "Directory: $PROJECT_DIR"
echo "Mode:      $MODE"

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

start_frontend() {
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
        init_database
        start_backend
        ;;
    frontend)
        start_frontend
        wait
        ;;
    all)
        init_database
        start_frontend
        sleep 2  # Give frontend time to start
        start_backend
        ;;
    *)
        echo "Usage: $0 [backend|frontend|all]"
        exit 1
        ;;
esac
