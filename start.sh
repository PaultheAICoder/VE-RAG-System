#!/bin/bash
cd /srv/VE-RAG-System
source .venv/bin/activate

# Export env vars from .env
export $(grep -v "^#" .env | xargs)

echo "=== AI Ready RAG Startup ==="

# Check/create admin user and tags via Python (more reliable than API)
python << "PYTHON"
from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import User, Tag
from ai_ready_rag.core.security import hash_password

init_db()

db = SessionLocal()
try:
    # Create admin user
    admin = db.query(User).filter(User.email == "admin@test.com").first()
    if not admin:
        admin = User(
            email="admin@test.com",
            display_name="Administrator",
            password_hash=hash_password("npassword"),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print("Admin user created: admin@test.com")
    else:
        print("Admin user exists: admin@test.com")

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
