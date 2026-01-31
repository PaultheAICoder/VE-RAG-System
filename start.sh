#!/bin/bash
# Start AI Ready RAG - works on both laptop and Spark

# Change to script directory (works regardless of where script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Export env vars from .env (skip comments and empty lines)
if [ -f ".env" ]; then
    export $(grep -v "^#" .env | grep -v "^$" | xargs)
else
    echo "Warning: No .env file found. Using defaults."
fi

echo "=== AI Ready RAG Startup ==="
echo "Directory: $SCRIPT_DIR"
echo "Profile:   ${ENV_PROFILE:-not set}"

# Check/create admin user and tags via Python (more reliable than API)
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

echo "=== Starting server on port 8502 ==="
exec python -m uvicorn ai_ready_rag.main:app --host 0.0.0.0 --port 8502
