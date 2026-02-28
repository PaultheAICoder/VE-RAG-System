"""CA Module: Insurance tables (accounts, policies, coverages, claims, certificates, requirements).

Revision ID: ca_001
Revises:
Create Date: 2026-02-27

Creates 6 insurance tables owned by the Community Associations module.
All tables include tenant_id and soft-delete columns.
Module rule: NO changes to core-owned tables.
"""

import sqlalchemy as sa
from alembic import op

revision: str = "ca_001"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── insurance_accounts ──────────────────────────────────────────────────
    op.create_table(
        "insurance_accounts",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("account_name", sa.String(500), nullable=False),
        sa.Column("account_type", sa.String(50)),  # condo_association|hoa|planned_community
        sa.Column("property_address", sa.String(500)),
        sa.Column("city", sa.String(100)),
        sa.Column("state", sa.String(2)),
        sa.Column("zip_code", sa.String(10)),
        sa.Column("units_residential", sa.Integer),
        sa.Column("units_commercial", sa.Integer),
        sa.Column("year_built", sa.Integer),
        sa.Column(
            "source_document_id", sa.String, sa.ForeignKey("documents.id", ondelete="SET NULL")
        ),
        sa.Column("extraction_confidence", sa.Float),
        sa.Column("custom_fields", sa.Text),  # JSON
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("valid_from", sa.DateTime),
        sa.Column("valid_to", sa.DateTime),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("deleted_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("account_name", "tenant_id", name="uq_insurance_accounts_name_tenant"),
    )
    op.create_index("ix_insurance_accounts_tenant_id", "insurance_accounts", ["tenant_id"])
    op.create_index("ix_insurance_accounts_account_name", "insurance_accounts", ["account_name"])

    # ── insurance_policies ──────────────────────────────────────────────────
    op.create_table(
        "insurance_policies",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("policy_number", sa.String(200)),
        sa.Column("carrier_name", sa.String(500)),
        sa.Column("broker_name", sa.String(500)),
        sa.Column("line_of_business", sa.String(100), nullable=False),
        # line_of_business values: commercial_property|gl|do|crime|umbrella|fidelity|residential|wc|epli|cyber|auto|equipment_breakdown
        sa.Column("policy_status", sa.String(50), server_default="active"),
        # policy_status values: active|expired|cancelled|pending
        sa.Column("inception_date", sa.Date),
        sa.Column("effective_date", sa.Date),
        sa.Column("expiration_date", sa.Date),
        sa.Column("premium_amount", sa.Float),
        sa.Column(
            "source_document_id", sa.String, sa.ForeignKey("documents.id", ondelete="SET NULL")
        ),
        sa.Column(
            "idempotency_key", sa.String(64)
        ),  # SHA256 of account_id+policy_number+inception_date
        sa.Column("is_active", sa.Boolean, server_default="1"),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("valid_from", sa.DateTime),
        sa.Column("valid_to", sa.DateTime),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("deleted_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "account_id",
            "policy_number",
            "inception_date",
            name="uq_insurance_policies_acct_pol_date",
        ),
    )
    op.create_index("ix_insurance_policies_account_id", "insurance_policies", ["account_id"])
    op.create_index(
        "ix_insurance_policies_acct_lob",
        "insurance_policies",
        ["account_id", "line_of_business"],
    )
    op.create_index(
        "ix_insurance_policies_expiration_date", "insurance_policies", ["expiration_date"]
    )

    # ── insurance_coverages ──────────────────────────────────────────────────
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
        sa.Column("coverage_type", sa.String(100), nullable=False),
        # coverage_type values: property|general_liability|directors_and_officers|crime|umbrella|fidelity|ho6_unit_owner|workers_comp|epli|cyber|auto_liability|equipment_breakdown
        sa.Column("limit_amount", sa.Float),
        sa.Column("deductible_amount", sa.Float),
        sa.Column("deductible_type", sa.String(50)),  # per_occurrence|aggregate|per_unit
        sa.Column("sublimit_type", sa.String(100)),
        sa.Column("sublimit_amount", sa.Float),
        sa.Column("notes", sa.Text),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("deleted_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "policy_id", "coverage_type", name="uq_insurance_coverages_policy_type"
        ),
    )
    op.create_index("ix_insurance_coverages_policy_id", "insurance_coverages", ["policy_id"])
    op.create_index("ix_insurance_coverages_account_id", "insurance_coverages", ["account_id"])

    # ── insurance_claims ────────────────────────────────────────────────────
    op.create_table(
        "insurance_claims",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "policy_id", sa.String, sa.ForeignKey("insurance_policies.id", ondelete="SET NULL")
        ),
        sa.Column("claim_number", sa.String(200)),
        sa.Column("date_of_loss", sa.Date),
        sa.Column("claim_type", sa.String(100)),
        sa.Column("line_of_business", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("claimant", sa.Text),  # PII — Fernet-encrypted at application layer
        sa.Column("amount_paid", sa.Float),
        sa.Column("amount_reserved", sa.Float),
        sa.Column("closed_amount", sa.Float),
        sa.Column("claim_status", sa.String(50), server_default="open"),
        # claim_status values: open|closed|reopened
        sa.Column("closed_date", sa.Date),
        sa.Column(
            "source_document_id", sa.String, sa.ForeignKey("documents.id", ondelete="SET NULL")
        ),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("deleted_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_insurance_claims_account_id", "insurance_claims", ["account_id"])
    op.create_index("ix_insurance_claims_date_of_loss", "insurance_claims", ["date_of_loss"])
    op.create_unique_constraint(
        "uq_insurance_claims_acct_number",
        "insurance_claims",
        ["account_id", "claim_number"],
    )

    # ── insurance_certificates ──────────────────────────────────────────────
    op.create_table(
        "insurance_certificates",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cert_holder_name", sa.String(500), nullable=False),
        sa.Column("cert_holder_address", sa.String(500)),
        sa.Column("acord_form_type", sa.String(20)),
        # acord_form_type values: acord_24|acord_25|acord_27|acord_28
        sa.Column("effective_date", sa.Date),
        sa.Column("expiration_date", sa.Date),
        sa.Column("additional_insured", sa.Boolean, server_default="0"),
        sa.Column(
            "source_document_id", sa.String, sa.ForeignKey("documents.id", ondelete="SET NULL")
        ),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("deleted_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "account_id",
            "cert_holder_name",
            "effective_date",
            "acord_form_type",
            name="uq_insurance_certificates_acct_holder_date_form",
        ),
    )
    op.create_index(
        "ix_insurance_certificates_account_id", "insurance_certificates", ["account_id"]
    )

    # ── insurance_requirements ──────────────────────────────────────────────
    op.create_table(
        "insurance_requirements",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "account_id",
            sa.String,
            sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("coverage_line", sa.String(50), nullable=False),
        sa.Column(
            "requirement_source", sa.String(100)
        ),  # ccr|bylaws|fannie_mae|fha|state_law|board_resolution
        sa.Column("requirement_text", sa.Text),
        sa.Column("min_limit", sa.Float),
        sa.Column(
            "min_limit_type", sa.String(50)
        ),  # per_occurrence|aggregate|per_unit|replacement_cost
        sa.Column("is_met", sa.Boolean, server_default="0"),
        sa.Column("current_limit", sa.Float),
        sa.Column("gap_amount", sa.Float),
        sa.Column("gap_description", sa.Text),
        sa.Column("external_standard", sa.String(100)),  # fannie_mae|fha|state_law
        sa.Column("section_reference", sa.String(200)),
        sa.Column("requirements_version", sa.String(50)),  # e.g., 2026-Q1
        sa.Column("effective_date", sa.Date),
        sa.Column("injected_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("injected_by_module_version", sa.String(50)),
        sa.Column("superseded_at", sa.DateTime),  # NULL = current active requirement
        sa.Column("advisory_only", sa.Boolean, server_default="0"),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("deleted_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_insurance_requirements_account_id", "insurance_requirements", ["account_id"]
    )
    op.create_index("ix_insurance_requirements_is_met", "insurance_requirements", ["is_met"])
    op.create_index(
        "ix_insurance_requirements_coverage_line",
        "insurance_requirements",
        ["coverage_line"],
    )

    # ── ca_carrier_aliases seed table ────────────────────────────────────────
    op.create_table(
        "ca_carrier_aliases",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("alias", sa.String(500), nullable=False),
        sa.Column("canonical_name", sa.String(500), nullable=False),
        sa.Column("carrier_code", sa.String(20)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("alias", name="uq_ca_carrier_aliases_alias"),
    )
    op.create_index("ix_ca_carrier_aliases_alias", "ca_carrier_aliases", ["alias"])


def downgrade() -> None:
    op.drop_table("ca_carrier_aliases")
    op.drop_table("insurance_requirements")
    op.drop_table("insurance_certificates")
    op.drop_table("insurance_claims")
    op.drop_table("insurance_coverages")
    op.drop_table("insurance_policies")
    op.drop_table("insurance_accounts")
