# AI Ready RAG - Spark Deployment Guide

This guide covers deploying AI Ready RAG on NVIDIA DGX Spark or compatible GPU servers for production use. For laptop/WSL development environments, see [WSL2_SETUP.md](./WSL2_SETUP.md).

---

## Overview

The Spark deployment profile enables:
- **Qdrant** vector database (replacing ChromaDB)
- **Docling** document processing with OCR support
- **qwen3:8b** chat model for improved response quality
- Enhanced token budgets for longer context and responses
- Hallucination checking enabled by default

---

## 1. Prerequisites

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA DGX Spark or compatible | DGX Spark with A100 |
| RAM | 32GB | 64GB+ |
| Storage | 100GB SSD | 500GB NVMe |
| Network | 1Gbps | 10Gbps (for model downloads) |

### 1.2 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Ubuntu | 22.04+ | Operating system |
| Python | 3.12+ | Application runtime |
| Docker | 24.0+ | Container runtime |
| NVIDIA Container Toolkit | Latest | GPU support for Docker |
| Git | 2.x | Source control |

### 1.3 Verify NVIDIA Container Toolkit

```bash
# Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

If this fails, install NVIDIA Container Toolkit:
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 2. System Dependencies

### 2.1 Tesseract OCR

Tesseract is required for OCR-enabled document processing (enabled by default in Spark profile).

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev
```

**RHEL/CentOS:**
```bash
sudo yum install -y tesseract tesseract-devel
```

**Verify installation:**
```bash
tesseract --version
# Expected: tesseract 4.x or 5.x
```

### 2.2 Additional System Libraries

```bash
# Required for python-magic and document processing
sudo apt-get install -y \
    libmagic1 \
    build-essential \
    poppler-utils
```

---

## 3. Infrastructure Setup

### 3.1 Qdrant Vector Database

Start Qdrant with persistent storage:

```bash
# Create data directory
sudo mkdir -p /opt/qdrant/data
sudo chown $(whoami):$(whoami) /opt/qdrant/data

# Run Qdrant container
docker run -d \
    --name qdrant \
    --restart unless-stopped \
    -p 6333:6333 \
    -p 6334:6334 \
    -v /opt/qdrant/data:/qdrant/storage:z \
    qdrant/qdrant:latest
```

**Verify Qdrant is running:**
```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vectorass database","version":"..."}
```

### 3.2 Ollama LLM Server

Start Ollama with GPU support:

```bash
# Run Ollama container with GPU access
docker run -d \
    --name ollama \
    --restart unless-stopped \
    --gpus all \
    -p 11434:11434 \
    -v /opt/ollama:/root/.ollama \
    ollama/ollama:latest
```

**Pull required models:**
```bash
# Chat model (Spark profile default)
docker exec ollama ollama pull qwen3:8b

# Embedding model
docker exec ollama ollama pull nomic-embed-text

# Verify models are available
docker exec ollama ollama list
```

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
# Should return JSON with model list
```

---

## 4. Application Installation

### 4.1 Clone Repository

```bash
cd /opt
sudo mkdir ai-ready-rag
sudo chown $(whoami):$(whoami) ai-ready-rag
git clone https://github.com/PaultheAICoder/VE-RAG-System.git ai-ready-rag
cd ai-ready-rag
```

### 4.2 Python Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 4.3 Install Dependencies

**Important:** Use `requirements-spark.txt` for Spark deployment (NOT `requirements.txt`).

```bash
pip install -r requirements-spark.txt
```

### 4.4 Create Data Directories

```bash
mkdir -p data/uploads
mkdir -p data/logs
chmod 700 data
```

---

## 5. Environment Configuration

### 5.1 Create Production `.env` File

Create `/opt/ai-ready-rag/.env`:

```bash
# =============================================================================
# AI Ready RAG - Spark Production Configuration
# =============================================================================

# Profile Selection (REQUIRED for Spark features)
ENV_PROFILE=spark

# =============================================================================
# SECURITY - MUST CHANGE THESE VALUES
# =============================================================================

# Generate with: openssl rand -hex 32
JWT_SECRET_KEY=your-secure-secret-key-change-this

# Initial admin credentials (change after first login)
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PASSWORD=your-secure-admin-password

# =============================================================================
# Application Settings
# =============================================================================
DEBUG=False
HOST=127.0.0.1
PORT=8000

# =============================================================================
# Database
# =============================================================================
DATABASE_URL=sqlite:///./data/ai_ready_rag.db

# =============================================================================
# Infrastructure Services
# =============================================================================
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
OLLAMA_BASE_URL=http://localhost:11434

# =============================================================================
# Document Processing
# =============================================================================
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=100
MAX_STORAGE_GB=50

# OCR Settings (Spark profile enables OCR by default)
OCR_LANGUAGE=eng
TABLE_EXTRACTION_MODE=accurate

# =============================================================================
# Audit Logging
# =============================================================================
# Options: essential, comprehensive, full_debug
AUDIT_LEVEL=comprehensive
```

### 5.2 Complete Environment Variable Reference

| Variable | Default | Spark Profile | Description |
|----------|---------|---------------|-------------|
| **Application** ||||
| `ENV_PROFILE` | laptop | **spark** | Deployment profile |
| `DEBUG` | True | **False** | Debug mode |
| `HOST` | 0.0.0.0 | 127.0.0.1 | Bind address |
| `PORT` | 8000 | 8000 | Server port |
| **Security** ||||
| `JWT_SECRET_KEY` | dev-secret-... | *MUST CHANGE* | JWT signing key |
| `JWT_EXPIRATION_HOURS` | 24 | 24 | Token lifetime |
| `PASSWORD_MIN_LENGTH` | 12 | 12 | Min password length |
| `LOCKOUT_ATTEMPTS` | 5 | 5 | Max failed logins |
| `LOCKOUT_MINUTES` | 15 | 15 | Lockout duration |
| **Admin** ||||
| `ADMIN_EMAIL` | admin@test.com | *CHANGE* | Initial admin email |
| `ADMIN_PASSWORD` | npassword | *CHANGE* | Initial admin password |
| **Database** ||||
| `DATABASE_URL` | sqlite:///./data/ai_ready_rag.db | same | SQLite path |
| **Vector Store** ||||
| `VECTOR_BACKEND` | chroma | **qdrant** | Vector backend |
| `QDRANT_URL` | http://localhost:6333 | same | Qdrant server |
| `QDRANT_COLLECTION` | documents | same | Collection name |
| **LLM** ||||
| `OLLAMA_BASE_URL` | http://localhost:11434 | same | Ollama server |
| `CHAT_MODEL` | llama3.2:latest | **qwen3:8b** | Chat model |
| `EMBEDDING_MODEL` | nomic-embed-text | same | Embedding model |
| **RAG Settings** ||||
| `RAG_MAX_CONTEXT_TOKENS` | 2000 | **6000** | Context window |
| `RAG_MAX_HISTORY_TOKENS` | 600 | **1500** | History budget |
| `RAG_MAX_RESPONSE_TOKENS` | 512 | **2048** | Response limit |
| `RAG_ENABLE_HALLUCINATION_CHECK` | False | **True** | Quality check |
| **Document Processing** ||||
| `CHUNKER_BACKEND` | simple | **docling** | Chunker backend |
| `ENABLE_OCR` | False | **True** | OCR enabled |
| `OCR_LANGUAGE` | eng | eng | OCR language |
| **Upload** ||||
| `UPLOAD_DIR` | ./data/uploads | same | Upload path |
| `MAX_UPLOAD_SIZE_MB` | 100 | 100 | Max file size |
| `MAX_STORAGE_GB` | 10 | 10-50 | Storage limit |

### 5.3 Profile Differences Summary

| Setting | Laptop | Spark |
|---------|--------|-------|
| Vector backend | ChromaDB | **Qdrant** |
| Chunker backend | Simple | **Docling** |
| OCR enabled | No | **Yes** |
| Chat model | llama3.2 | **qwen3:8b** |
| Context tokens | 2,000 | **6,000** |
| Response tokens | 512 | **2,048** |
| Hallucination check | Off | **On** |

---

## 6. Security Hardening

### 6.1 Generate JWT Secret

**Never use the default secret in production.**

```bash
# Generate a secure 256-bit secret
openssl rand -hex 32
```

Copy the output to `JWT_SECRET_KEY` in your `.env` file.

### 6.2 HTTPS with NGINX Reverse Proxy

Install NGINX:
```bash
sudo apt-get install -y nginx
```

Create `/etc/nginx/sites-available/ai-ready-rag`:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-spark-hostname;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-spark-hostname;

    # SSL certificates (use Let's Encrypt or corporate certs)
    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Proxy to application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (required for Gradio)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # File upload limit
    client_max_body_size 100M;
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/ai-ready-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### 6.3 Firewall Configuration

Using UFW (Uncomplicated Firewall):

```bash
# Allow SSH
sudo ufw allow ssh

# Allow HTTPS only (not HTTP - handled by redirect)
sudo ufw allow 443/tcp

# Block direct access to internal services
sudo ufw deny 6333/tcp   # Qdrant
sudo ufw deny 6334/tcp   # Qdrant gRPC
sudo ufw deny 11434/tcp  # Ollama
sudo ufw deny 8000/tcp   # Direct app access

# Enable firewall
sudo ufw enable
sudo ufw status
```

### 6.4 Database Security

```bash
# Secure SQLite database file
chmod 600 /opt/ai-ready-rag/data/ai_ready_rag.db

# Ensure data directory is not world-readable
chmod 700 /opt/ai-ready-rag/data
```

### 6.5 First-Time Admin Setup

On first startup, create the initial admin user via the setup endpoint:

```bash
curl -X POST http://localhost:8000/api/auth/setup \
    -H "Content-Type: application/json" \
    -d '{
        "email": "admin@yourcompany.com",
        "password": "your-secure-password",
        "display_name": "Administrator"
    }'
```

**Important:** This endpoint is disabled after the first admin is created.

---

## 7. Starting the Application

### 7.1 Direct Start (Development/Testing)

```bash
cd /opt/ai-ready-rag
source .venv/bin/activate
uvicorn ai_ready_rag.main:app --host 127.0.0.1 --port 8000
```

### 7.2 Systemd Service (Production)

Create `/etc/systemd/system/ai-ready-rag.service`:

```ini
[Unit]
Description=AI Ready RAG Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=airag
Group=airag
WorkingDirectory=/opt/ai-ready-rag
Environment=PATH=/opt/ai-ready-rag/.venv/bin:/usr/local/bin:/usr/bin
EnvironmentFile=/opt/ai-ready-rag/.env
ExecStart=/opt/ai-ready-rag/.venv/bin/uvicorn ai_ready_rag.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create service user and set permissions:
```bash
sudo useradd -r -s /bin/false airag
sudo chown -R airag:airag /opt/ai-ready-rag
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-ready-rag
sudo systemctl start ai-ready-rag
sudo systemctl status ai-ready-rag
```

### 7.3 Docker Compose (Alternative)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: unless-stopped
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "127.0.0.1:11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  qdrant_data:
  ollama_data:
```

Start infrastructure:
```bash
docker compose up -d
```

---

## 8. Health Checks

### 8.1 Application Health

```bash
# Basic health check
curl http://localhost:8000/api/health
# Expected: {"status":"healthy"}

# Version info
curl http://localhost:8000/api/version
# Expected: {"app_name":"AI Ready RAG","app_version":"0.5.0",...}

# Admin-only system health (requires auth)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/admin/health
```

### 8.2 Qdrant Health

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}

# Check collection
curl http://localhost:6333/collections/documents
```

### 8.3 Ollama Health

```bash
# Check API is responding
curl http://localhost:11434/api/tags
# Expected: {"models":[...]}

# Test embedding model
curl http://localhost:11434/api/embeddings \
    -d '{"model":"nomic-embed-text","prompt":"test"}'
```

### 8.4 Full System Check Script

Create `/opt/ai-ready-rag/scripts/health-check.sh`:

```bash
#!/bin/bash
set -e

echo "Checking services..."

# Check Qdrant
if curl -sf http://localhost:6333/healthz > /dev/null; then
    echo "[OK] Qdrant"
else
    echo "[FAIL] Qdrant"
    exit 1
fi

# Check Ollama
if curl -sf http://localhost:11434/api/tags > /dev/null; then
    echo "[OK] Ollama"
else
    echo "[FAIL] Ollama"
    exit 1
fi

# Check Application
if curl -sf http://localhost:8000/api/health > /dev/null; then
    echo "[OK] Application"
else
    echo "[FAIL] Application"
    exit 1
fi

echo "All services healthy!"
```

```bash
chmod +x /opt/ai-ready-rag/scripts/health-check.sh
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **OCR not working** | Documents without text content | Verify Tesseract: `tesseract --version`. Check `ENABLE_OCR=True` |
| **Slow embeddings** | Long wait on document upload | Check Ollama GPU access: `docker logs ollama`. Verify `nvidia-smi` shows usage |
| **Connection refused** | Cannot connect to Qdrant/Ollama | Check containers running: `docker ps`. Verify ports: `netstat -tlnp` |
| **GPU not detected** | Ollama using CPU | Run `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi` to verify toolkit |
| **Out of memory** | Container crashes | Reduce `RAG_MAX_CONTEXT_TOKENS`. Check `docker stats` |
| **JWT errors** | Authentication failures | Regenerate secret: `openssl rand -hex 32`. Restart app |
| **Upload fails** | Large file rejected | Check `MAX_UPLOAD_SIZE_MB` and NGINX `client_max_body_size` |
| **Slow response** | Long wait for chat | Check Ollama is using GPU. Consider smaller model |

### 9.2 Log Locations

| Component | Log Location |
|-----------|--------------|
| Application | `journalctl -u ai-ready-rag` |
| NGINX | `/var/log/nginx/error.log` |
| Qdrant | `docker logs qdrant` |
| Ollama | `docker logs ollama` |

### 9.3 Debug Mode

For troubleshooting, temporarily enable debug mode:

```bash
# In .env
DEBUG=True
AUDIT_LEVEL=full_debug
```

Restart the service and check logs:
```bash
sudo systemctl restart ai-ready-rag
journalctl -u ai-ready-rag -f
```

**Remember to disable debug mode in production.**

---

## 10. Maintenance

### 10.1 Backup Procedures

**SQLite Database:**
```bash
# Stop application first for consistent backup
sudo systemctl stop ai-ready-rag

# Backup database
cp /opt/ai-ready-rag/data/ai_ready_rag.db /backup/ai_ready_rag_$(date +%Y%m%d).db

# Restart application
sudo systemctl start ai-ready-rag
```

**Qdrant Data:**
```bash
# Create snapshot via API
curl -X POST http://localhost:6333/collections/documents/snapshots

# Copy snapshot from Docker volume
docker cp qdrant:/qdrant/storage/snapshots /backup/qdrant_snapshots/
```

**Uploaded Documents:**
```bash
# Backup upload directory
tar -czf /backup/uploads_$(date +%Y%m%d).tar.gz /opt/ai-ready-rag/data/uploads/
```

### 10.2 Log Rotation

Create `/etc/logrotate.d/ai-ready-rag`:

```
/opt/ai-ready-rag/data/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 airag airag
}
```

### 10.3 Model Updates

Update Ollama models:
```bash
# Pull latest versions
docker exec ollama ollama pull qwen3:8b
docker exec ollama ollama pull nomic-embed-text

# Remove old versions (optional)
docker exec ollama ollama rm qwen3:8b-old-version
```

### 10.4 Application Updates

```bash
# Stop service
sudo systemctl stop ai-ready-rag

# Backup current version
cp -r /opt/ai-ready-rag /opt/ai-ready-rag.backup

# Pull updates
cd /opt/ai-ready-rag
git pull

# Update dependencies
source .venv/bin/activate
pip install -r requirements-spark.txt

# Run database migrations (if any)
# alembic upgrade head

# Start service
sudo systemctl start ai-ready-rag

# Verify health
./scripts/health-check.sh
```

---

## Appendix A: Quick Reference

### Service Management

```bash
# Start all services
docker start qdrant ollama
sudo systemctl start ai-ready-rag

# Stop all services
sudo systemctl stop ai-ready-rag
docker stop ollama qdrant

# View logs
journalctl -u ai-ready-rag -f
docker logs -f qdrant
docker logs -f ollama
```

### Key URLs

| Service | URL |
|---------|-----|
| Web UI | https://your-hostname/app |
| API Docs | https://your-hostname/docs |
| Health | http://localhost:8000/api/health |
| Qdrant | http://localhost:6333 |
| Ollama | http://localhost:11434 |

### Environment Quick Check

```bash
# Verify profile
grep ENV_PROFILE /opt/ai-ready-rag/.env

# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check application service
sudo systemctl status ai-ready-rag
```
