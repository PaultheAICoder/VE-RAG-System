#!/usr/bin/env bash
# provision-customer.sh — Provision a new VaultIQ customer instance
# Usage: ./scripts/provision-customer.sh <tenant_id>
set -euo pipefail

TENANT_ID="${1:?Usage: $0 <tenant_id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Provisioning VaultIQ tenant: $TENANT_ID"

# 1. Create tenant directory
TENANT_DIR="$PROJECT_DIR/tenant-instances/$TENANT_ID"
mkdir -p "$TENANT_DIR"

# 2. Create tenant.json from template
if [ ! -f "$TENANT_DIR/tenant.json" ]; then
    cat > "$TENANT_DIR/tenant.json" <<EOF
{
  "tenant_id": "$TENANT_ID",
  "display_name": "$TENANT_ID",
  "brand": {
    "primary_color": "#1a56db",
    "logo_url": null
  },
  "feature_flags": {
    "ca_enabled": false,
    "claude_enrichment_enabled": true,
    "claude_query_enabled": true,
    "structured_query_enabled": true
  },
  "active_modules": ["core"]
}
EOF
    echo "    Created $TENANT_DIR/tenant.json"
fi

# 3. Generate VAULTIQ_ENCRYPTION_KEY if not set
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "==> Generating .env from template..."
    cp "$PROJECT_DIR/.env.customer.template" "$PROJECT_DIR/.env"

    # Generate Fernet key
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "")
    if [ -n "$FERNET_KEY" ]; then
        sed -i "s|VAULTIQ_ENCRYPTION_KEY=|VAULTIQ_ENCRYPTION_KEY=$FERNET_KEY|" "$PROJECT_DIR/.env"
        echo "    Generated VAULTIQ_ENCRYPTION_KEY"
    fi
    sed -i "s|TENANT_ID=|TENANT_ID=$TENANT_ID|" "$PROJECT_DIR/.env"
    echo "    .env created — fill in ANTHROPIC_API_KEY, POSTGRES_PASSWORD, JWT_SECRET_KEY"
fi

echo ""
echo "==> Next steps:"
echo "    1. Edit .env and fill in: ANTHROPIC_API_KEY, POSTGRES_PASSWORD, JWT_SECRET_KEY"
echo "    2. Start the stack: docker compose -f docker-compose.customer.yml up -d"
echo "    3. Run migrations: docker compose -f docker-compose.customer.yml exec api alembic upgrade head"
echo "    4. Verify health: curl http://localhost:8502/health"
