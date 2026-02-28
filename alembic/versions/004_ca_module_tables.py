"""Create Community Associations module tables.

Revision ID: 004
Revises: 003
Create Date: 2026-02-28

Creates:
- insurance_accounts     — HOA / condo property accounts
- insurance_policies     — Insurance policies linked to an account
- insurance_coverages    — Individual coverage lines for a policy
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "insurance_accounts",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("account_name", sa.String, nullable=False),
        sa.Column("account_type", sa.String, nullable=True),
        sa.Column("property_address", sa.String, nullable=True),
        sa.Column("city", sa.String, nullable=True),
        sa.Column("state", sa.String, nullable=True),
        sa.Column("zip_code", sa.String, nullable=True),
        sa.Column("units_residential", sa.Integer, nullable=True),
        sa.Column("units_commercial", sa.Integer, nullable=True),
        sa.Column("year_built", sa.Integer, nullable=True),
        sa.Column("source_document_id", sa.String, nullable=True),
        sa.Column("extraction_confidence", sa.Float, nullable=True),
        sa.Column("custom_fields", sa.Text, nullable=True),
        sa.Column("tenant_id", sa.String, nullable=False),
        sa.Column("valid_from", sa.DateTime, nullable=True),
        sa.Column("valid_to", sa.DateTime, nullable=True),
        sa.Column("is_deleted", sa.Boolean, default=False),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_insurance_accounts_tenant_id", "insurance_accounts", ["tenant_id"])
    op.create_index("ix_insurance_accounts_account_name", "insurance_accounts", ["account_name"])

    op.create_table(
        "insurance_policies",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("policy_number", sa.String, nullable=True),
        sa.Column("carrier_name", sa.String, nullable=True),
        sa.Column("broker_name", sa.String, nullable=True),
        sa.Column("line_of_business", sa.String, nullable=False),
        sa.Column("policy_status", sa.String, server_default="active"),
        sa.Column("inception_date", sa.DateTime, nullable=True),
        sa.Column("effective_date", sa.DateTime, nullable=True),
        sa.Column("expiration_date", sa.DateTime, nullable=True),
        sa.Column("premium_amount", sa.Float, nullable=True),
        sa.Column("source_document_id", sa.String, nullable=True),
        sa.Column("idempotency_key", sa.String, nullable=True, unique=True),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("tenant_id", sa.String, nullable=False),
        sa.Column("valid_from", sa.DateTime, nullable=True),
        sa.Column("valid_to", sa.DateTime, nullable=True),
        sa.Column("is_deleted", sa.Boolean, default=False),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_insurance_policies_account_id", "insurance_policies", ["account_id"])
    op.create_index("ix_insurance_policies_tenant_id", "insurance_policies", ["tenant_id"])

    op.create_table(
        "insurance_coverages",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "policy_id",
            sa.String,
            sa.ForeignKey("insurance_policies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("coverage_type", sa.String, nullable=False),
        sa.Column("limit_amount", sa.Float, nullable=True),
        sa.Column("deductible_amount", sa.Float, nullable=True),
        sa.Column("deductible_type", sa.String, nullable=True),
        sa.Column("sublimit_type", sa.String, nullable=True),
        sa.Column("sublimit_amount", sa.Float, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("tenant_id", sa.String, nullable=False),
        sa.Column("is_deleted", sa.Boolean, default=False),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_insurance_coverages_policy_id", "insurance_coverages", ["policy_id"])
    op.create_index("ix_insurance_coverages_account_id", "insurance_coverages", ["account_id"])
    op.create_index("ix_insurance_coverages_tenant_id", "insurance_coverages", ["tenant_id"])
    op.create_index(
        "ix_insurance_coverages_coverage_type", "insurance_coverages", ["coverage_type"]
    )


def downgrade() -> None:
    op.drop_table("insurance_coverages")
    op.drop_table("insurance_policies")
    op.drop_table("insurance_accounts")
