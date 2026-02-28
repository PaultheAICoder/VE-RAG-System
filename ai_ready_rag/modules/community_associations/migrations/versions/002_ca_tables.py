"""CA Module migration 002: CA-specific operational tables.

Revision ID: ca_002
Revises: ca_001
Create Date: 2026-02-27

Creates 4 CA-specific tables:
- ca_reserve_studies: HOA reserve fund study tracking
- ca_unit_owners: Unit owner data with PII columns (Fernet-encrypted at app layer)
- ca_board_resolutions: Board meeting insurance resolutions
- ca_letter_batches + ca_letter_batch_items: Unit owner letter batch workflow
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "ca_002"
down_revision = "ca_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── ca_reserve_studies ─────────────────────────────────────────────────
    # Tracks reserve fund study documents and key financial metrics
    op.create_table(
        "ca_reserve_studies",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.String(100), nullable=False, index=True),
        sa.Column("study_date", sa.Date, nullable=True),
        sa.Column("preparer", sa.String(255), nullable=True),
        sa.Column("percent_funded", sa.Numeric(6, 2), nullable=True),  # 0.00–999.99
        sa.Column("actual_reserve_balance_usd", sa.Numeric(15, 2), nullable=True),
        sa.Column("fully_funded_balance_usd", sa.Numeric(15, 2), nullable=True),
        sa.Column("annual_contribution_usd", sa.Numeric(15, 2), nullable=True),
        sa.Column("document_id", sa.String, nullable=True),  # soft FK → documents
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=True),
        sa.Column("deleted_at", sa.DateTime, nullable=True),  # soft-delete
    )

    # ── ca_unit_owners ────────────────────────────────────────────────────
    # PII columns encrypted at the application layer using Fernet
    # Columns marked [ENCRYPTED] store ciphertext bytes as base64 strings
    op.create_table(
        "ca_unit_owners",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.String(100), nullable=False, index=True),
        sa.Column("unit_number", sa.String(50), nullable=False),
        sa.Column("owner_name_encrypted", sa.Text, nullable=True),  # [ENCRYPTED] full name
        sa.Column("email_encrypted", sa.Text, nullable=True),  # [ENCRYPTED] email
        sa.Column("phone_encrypted", sa.Text, nullable=True),  # [ENCRYPTED] phone
        sa.Column("ho6_required", sa.Boolean, nullable=True),
        sa.Column("ho6_verified", sa.Boolean, default=False),
        sa.Column("ho6_expiration_date", sa.Date, nullable=True),
        sa.Column(
            "compliance_status", sa.String(50), default="unknown"
        ),  # unknown|compliant|non_compliant|pending
        sa.Column("last_letter_sent_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=True),
        sa.Column("deleted_at", sa.DateTime, nullable=True),  # soft-delete + PII purge
        sa.UniqueConstraint("account_id", "unit_number", name="uq_ca_unit_owners_account_unit"),
    )

    # ── ca_board_resolutions ──────────────────────────────────────────────
    # Board meeting minutes — insurance-related resolutions and decisions
    op.create_table(
        "ca_board_resolutions",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.String(100), nullable=False, index=True),
        sa.Column("meeting_date", sa.Date, nullable=True),
        sa.Column(
            "resolution_type", sa.String(100), nullable=True
        ),  # coverage_approval|carrier_change|deductible_vote
        sa.Column("resolution_text", sa.Text, nullable=True),
        sa.Column("vote_result", sa.String(50), nullable=True),  # approved|rejected|tabled
        sa.Column("document_id", sa.String, nullable=True),  # soft FK → documents
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
    )

    # ── ca_letter_batches ─────────────────────────────────────────────────
    # Tracks a batch run of unit owner compliance letters
    op.create_table(
        "ca_letter_batches",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.String(100), nullable=False, index=True),
        sa.Column("batch_name", sa.String(255), nullable=True),
        sa.Column(
            "status", sa.String(50), default="pending"
        ),  # pending|generating|completed|failed
        sa.Column(
            "letter_type", sa.String(100), nullable=True
        ),  # ho6_reminder|compliance_notice|renewal_notice
        sa.Column("total_count", sa.Integer, default=0),
        sa.Column("generated_count", sa.Integer, default=0),
        sa.Column("failed_count", sa.Integer, default=0),
        sa.Column("initiated_by", sa.String, nullable=True),  # user_id
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )

    # ── ca_letter_batch_items ─────────────────────────────────────────────
    # One row per letter in a batch
    op.create_table(
        "ca_letter_batch_items",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "batch_id",
            sa.String,
            sa.ForeignKey("ca_letter_batches.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "unit_owner_id",
            sa.String,
            sa.ForeignKey("ca_unit_owners.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("unit_number", sa.String(50), nullable=True),  # denormalized for readability
        sa.Column("status", sa.String(50), default="pending"),  # pending|generated|sent|failed
        sa.Column("output_path", sa.String(500), nullable=True),  # path to generated PDF
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("sent_at", sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ca_letter_batch_items")
    op.drop_table("ca_letter_batches")
    op.drop_table("ca_board_resolutions")
    op.drop_table("ca_unit_owners")
    op.drop_table("ca_reserve_studies")
