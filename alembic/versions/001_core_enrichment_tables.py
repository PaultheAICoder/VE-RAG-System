"""Core enrichment tables and document enrichment columns.

Revision ID: 001
Revises:
Create Date: 2026-02-27

Creates:
- enrichment_synopses table
- enrichment_entities table
- review_items table
- Adds enrichment columns to documents table
"""

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- enrichment_synopses ---
    op.create_table(
        "enrichment_synopses",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "document_id",
            sa.String,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("synopsis_text", sa.Text, nullable=False),
        sa.Column("model_id", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(50)),
        sa.Column("token_cost", sa.Integer),
        sa.Column("cost_usd", sa.Float),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_enrichment_synopses_document_id", "enrichment_synopses", ["document_id"])

    # --- enrichment_entities ---
    op.create_table(
        "enrichment_entities",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column(
            "synopsis_id",
            sa.String,
            sa.ForeignKey("enrichment_synopses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.String,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.String(100), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("canonical_value", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_enrichment_entities_synopsis_id", "enrichment_entities", ["synopsis_id"])
    op.create_index("ix_enrichment_entities_document_id", "enrichment_entities", ["document_id"])
    op.create_index("ix_enrichment_entities_entity_type", "enrichment_entities", ["entity_type"])

    # --- review_items ---
    op.create_table(
        "review_items",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("query_id", sa.String),
        sa.Column(
            "document_id",
            sa.String,
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
        ),
        sa.Column("review_reason", sa.String(100), nullable=False),
        sa.Column("answer_text", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("candidate_types", sa.Text),  # JSON array
        sa.Column("candidate_scores", sa.Text),  # JSON array
        sa.Column("resolved_at", sa.DateTime),
        sa.Column("resolved_by", sa.String),
        sa.Column("tenant_id", sa.String(100)),
        sa.Column("is_deleted", sa.Boolean, server_default="0"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_review_items_query_id", "review_items", ["query_id"])
    op.create_index("ix_review_items_review_reason", "review_items", ["review_reason"])
    op.create_index("ix_review_items_resolved_at", "review_items", ["resolved_at"])

    # --- Add enrichment columns to documents ---
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(sa.Column("enrichment_status", sa.String(50)))
        batch_op.add_column(sa.Column("enrichment_model", sa.String(100)))
        batch_op.add_column(sa.Column("enrichment_version", sa.String(50)))
        batch_op.add_column(sa.Column("enrichment_tokens_used", sa.Integer))
        batch_op.add_column(sa.Column("enrichment_cost_usd", sa.Float))
        batch_op.add_column(sa.Column("enrichment_completed_at", sa.DateTime))
        batch_op.add_column(sa.Column("document_role", sa.String(50)))


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("document_role")
        batch_op.drop_column("enrichment_completed_at")
        batch_op.drop_column("enrichment_cost_usd")
        batch_op.drop_column("enrichment_tokens_used")
        batch_op.drop_column("enrichment_version")
        batch_op.drop_column("enrichment_model")
        batch_op.drop_column("enrichment_status")

    op.drop_index("ix_review_items_resolved_at", table_name="review_items")
    op.drop_index("ix_review_items_review_reason", table_name="review_items")
    op.drop_index("ix_review_items_query_id", table_name="review_items")
    op.drop_table("review_items")

    op.drop_index("ix_enrichment_entities_entity_type", table_name="enrichment_entities")
    op.drop_index("ix_enrichment_entities_document_id", table_name="enrichment_entities")
    op.drop_index("ix_enrichment_entities_synopsis_id", table_name="enrichment_entities")
    op.drop_table("enrichment_entities")

    op.drop_index("ix_enrichment_synopses_document_id", table_name="enrichment_synopses")
    op.drop_table("enrichment_synopses")
