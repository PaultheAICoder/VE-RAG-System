-- Initialize pgvector extension for VaultIQ
-- This script runs automatically when the PostgreSQL container starts for the first time.

CREATE EXTENSION IF NOT EXISTS vector;

-- Grant permissions to vaultiq user
GRANT ALL PRIVILEGES ON DATABASE vaultiq TO vaultiq;
