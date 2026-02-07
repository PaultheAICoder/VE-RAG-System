# WSL2 Development Environment Setup

**Purpose:** Move AI Ready RAG development to WSL2 with Ollama and Qdrant
**Date:** January 28, 2026

---

## Prerequisites Checklist

- [ ] Windows 10 (2004+) or Windows 11
- [ ] Admin access on your machine
- [ ] ~20GB free disk space
- [ ] Docker Desktop (optional, or use Docker in WSL2)

---

## Phase 1: WSL2 Setup

### Step 1.1: Enable WSL2

Open PowerShell as Administrator:

```powershell
# Install WSL with Ubuntu (default)
wsl --install

# Or if WSL is already installed, set default version
wsl --set-default-version 2
```

**Restart your computer when prompted.**

### Step 1.2: Install Ubuntu

After restart, if Ubuntu didn't auto-install:

```powershell
wsl --install -d Ubuntu-24.04
```

### Step 1.3: Initial Ubuntu Setup

Ubuntu will launch and prompt for username/password. Then:

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl git wget software-properties-common

# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Verify
python3.12 --version
```

### Step 1.4: Configure Git in WSL2

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use Windows credential manager
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager.exe"
```

---

## Phase 2: Project Setup in WSL2

### Step 2.1: Choose Project Location

**Option A: Clone fresh in WSL2 filesystem (recommended for performance)**

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/YOUR_REPO/VE-RAG-System.git
cd VE-RAG-System
```

**Option B: Access Windows files (slower I/O)**

```bash
cd /mnt/c/Users/jjob/projects/VE-RAG-System
```

> **Recommendation:** Use Option A. WSL2 native filesystem is significantly faster.

### Step 2.2: Create Python Virtual Environment

```bash
cd ~/projects/VE-RAG-System

# Create venv with Python 3.12
python3.12 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2.3: Install Dependencies

```bash
# Install API + RAG dependencies (no chromadb)
pip install -r requirements-wsl.txt
```

---

## Phase 3: Ollama Setup

### Step 3.1: Install Ollama in WSL2

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 3.2: Start Ollama Service

```bash
# Start Ollama (runs in background)
ollama serve &

# Or in a separate terminal
ollama serve
```

### Step 3.3: Pull Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Chat model - choose based on your RAM
ollama pull llama3.2          # 3B - ~4GB RAM (lightweight)
# OR
ollama pull qwen3:8b          # 8B - ~8GB RAM (better quality)
```

### Step 3.4: Verify Ollama

```bash
# Test embedding
curl http://localhost:11434/api/embeddings -d '{"model": "nomic-embed-text", "prompt": "test"}'

# Test chat
ollama run llama3.2 "Hello, how are you?"
```

---

## Phase 4: Qdrant Setup

### Step 4.1: Install Docker in WSL2

```bash
# Install Docker
sudo apt install -y docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo service docker start

# Logout and login again for group changes
exit
```

Re-open WSL2 terminal, then verify:

```bash
docker --version
docker run hello-world
```

### Step 4.2: Run Qdrant Container

```bash
# Create data directory
mkdir -p ~/qdrant_data

# Run Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ~/qdrant_data:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant:v1.13.2
```

### Step 4.3: Verify Qdrant

```bash
# Check container
docker ps

# Test API
curl http://localhost:6333/collections
```

**Expected:** `{"result":{"collections":[]},"status":"ok","time":...}`

---

## Phase 5: Environment Configuration

### Step 5.1: Create .env File

```bash
cd ~/projects/VE-RAG-System

cat > .env << 'EOF'
# AI Ready RAG - WSL2 Development Environment

# Application
DEBUG=true
ENABLE_RAG=true

# JWT (generate a real secret for production)
JWT_SECRET_KEY=dev-secret-change-in-production-use-openssl-rand-hex-32

# Database
DATABASE_URL=sqlite:///./data/ai_ready_rag.db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
CHAT_MODEL=llama3.2

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

# Audit
AUDIT_LEVEL=full_debug
EOF
```

### Step 5.2: Create Start Script

```bash
cat > start-dev.sh << 'EOF'
#!/bin/bash
set -e

echo "=== AI Ready RAG Development Server ==="

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Check Qdrant
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "Starting Qdrant..."
    docker start qdrant 2>/dev/null || docker run -d \
        --name qdrant \
        -p 6333:6333 \
        -v ~/qdrant_data:/qdrant/storage \
        qdrant/qdrant:v1.13.2
    sleep 3
fi

# Activate venv
source .venv/bin/activate

# Start FastAPI
echo ""
echo "Starting server at http://localhost:8000"
echo "API docs at http://localhost:8000/api/docs"
echo ""
python -m uvicorn ai_ready_rag.main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x start-dev.sh
```

---

## Phase 6: Clean Up Windows Environment

### Step 6.1: Remove Windows venv (Optional)

In PowerShell on Windows:

```powershell
cd C:\Users\jjob\projects\VE-RAG-System

# Remove old venv
Remove-Item -Recurse -Force venv

# Remove cached data (optional)
Remove-Item -Recurse -Force chroma_db -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force data -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
```

### Step 6.2: Update .gitignore

Ensure these are in `.gitignore`:

```
# Virtual environments
.venv/
venv/

# Data
data/
chroma_db/
*.db

# Environment
.env

# Python
__pycache__/
*.pyc
*.pyo
```

---

## Phase 7: Requirements Optimization

### Step 7.1: Create WSL-Specific Requirements

```bash
# requirements-wsl.txt - Optimized for WSL2 development

# ============== Core API ==============
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
email-validator>=2.0.0
sqlalchemy>=2.0.0
bcrypt>=4.0.0
pyjwt>=2.8.0
httpx>=0.27.0
python-dotenv>=1.0.0

# ============== Vector Store ==============
qdrant-client>=1.13.0

# ============== LLM ==============
langchain>=0.3.0
langchain-community>=0.3.0
langchain-ollama>=0.2.0

# ============== Document Processing ==============
# Start with lightweight, add docling when needed
python-magic>=0.4.27
pypdf>=3.0.0
python-docx>=1.0.0
openpyxl>=3.1.0
markdown>=3.5.0

# ============== Testing ==============
pytest>=8.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.23.0

# ============== Optional: Full Docling (uncomment when needed) ==============
# docling>=2.0.0
# docling-core
```

### Step 7.2: What Was Removed

| Package | Why Removed |
|---------|-------------|
| `chromadb` | Replaced by Qdrant |
| `onnxruntime` | Was chromadb dependency |
| `aiohttp` | Not needed, httpx covers async |
| `numpy` | Pulled in by other deps |
| `docling` | Optional - add when needed for OCR/tables |

### Step 7.3: What Was Added

| Package | Why Added |
|---------|-----------|
| `qdrant-client` | Vector database |
| `python-magic` | File type detection |
| `pypdf` | PDF text extraction (lightweight) |
| `python-docx` | Word doc parsing |
| `openpyxl` | Excel parsing |
| `pytest-asyncio` | Async test support |

---

## Phase 8: Verification

### Step 8.1: Run Tests

```bash
cd ~/projects/VE-RAG-System
source .venv/bin/activate

# Run tests
pytest

# With coverage
pytest --cov=ai_ready_rag --cov-report=term-missing
```

### Step 8.2: Start Development Server

```bash
./start-dev.sh
```

### Step 8.3: Test Endpoints

From another WSL2 terminal (or Windows browser):

```bash
# Health check
curl http://localhost:8000/api/health

# Setup first admin
curl -X POST http://localhost:8000/api/auth/setup \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"AdminPassword123","display_name":"Admin"}'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"AdminPassword123"}'
```

### Step 8.4: Access from Windows Browser

WSL2 localhost is accessible from Windows:

- API Docs: http://localhost:8502/api/docs
- Frontend: http://localhost:5173 (dev) or http://localhost:8502 (production)

---

## Quick Reference

### Daily Development Workflow

```bash
# Open WSL2 terminal
wsl

# Navigate to project
cd ~/projects/VE-RAG-System

# Start services and server
./start-dev.sh
```

### Useful Commands

```bash
# Check Ollama models
ollama list

# Check Qdrant collections
curl http://localhost:6333/collections

# View Docker containers
docker ps

# Stop everything
docker stop qdrant
pkill ollama
```

### VS Code Integration

```powershell
# From Windows, open WSL2 project in VS Code
code --remote wsl+Ubuntu ~/projects/VE-RAG-System
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ollama: command not found` | Re-run install script, restart terminal |
| Docker permission denied | Run `sudo service docker start`, re-login |
| Port already in use | `lsof -i :8000` then `kill <PID>` |
| Slow file access | Move project to WSL2 filesystem (`~/`) |
| Can't access from Windows browser | Check Windows firewall, use `0.0.0.0` not `127.0.0.1` |

---

## Summary

| Component | Location | Port |
|-----------|----------|------|
| FastAPI | WSL2 | 8000 |
| Ollama | WSL2 | 11434 |
| Qdrant | Docker in WSL2 | 6333 |
| SQLite | WSL2 filesystem | - |
| Project | `~/projects/VE-RAG-System` | - |

**Total setup time:** ~30-45 minutes
