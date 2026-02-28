"""Insurance ORM models for the Community Associations module.

These models are registered against a separate SQLAlchemy Base (CABase)
so that CA tables are isolated from the core platform metadata.

Tables:
    insurance_accounts  — HOA / condo property accounts
    insurance_policies  — Insurance policies linked to an account
    insurance_coverages — Individual coverage lines for a policy
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class CABase(DeclarativeBase):
    """Separate declarative base for CA module tables."""


class InsuranceAccount(CABase):
    """HOA / condo property account extracted from insurance documents."""

    __tablename__ = "insurance_accounts"

    id = Column(String, primary_key=True, default=_generate_uuid)
    account_name = Column(String, nullable=False)
    account_type = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    zip_code = Column(String, nullable=True)
    units_residential = Column(Integer, nullable=True)
    units_commercial = Column(Integer, nullable=True)
    year_built = Column(Integer, nullable=True)
    source_document_id = Column(String, nullable=True)  # FK to core documents
    extraction_confidence = Column(Float, nullable=True)
    custom_fields = Column(Text, nullable=True)  # JSON blob
    tenant_id = Column(String, nullable=False, index=True)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    policies = relationship(
        "InsurancePolicy", back_populates="account", cascade="all, delete-orphan"
    )
    coverages = relationship(
        "InsuranceCoverage", back_populates="account", cascade="all, delete-orphan"
    )


class InsurancePolicy(CABase):
    """Insurance policy linked to an account."""

    __tablename__ = "insurance_policies"

    id = Column(String, primary_key=True, default=_generate_uuid)
    account_id = Column(
        String,
        ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    policy_number = Column(String, nullable=True)
    carrier_name = Column(String, nullable=True)
    broker_name = Column(String, nullable=True)
    line_of_business = Column(String, nullable=False)
    policy_status = Column(String, default="active")
    inception_date = Column(DateTime, nullable=True)
    effective_date = Column(DateTime, nullable=True)
    expiration_date = Column(DateTime, nullable=True)
    premium_amount = Column(Float, nullable=True)
    source_document_id = Column(String, nullable=True)
    idempotency_key = Column(String, nullable=True, unique=True)
    is_active = Column(Boolean, default=True)
    tenant_id = Column(String, nullable=False, index=True)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    account = relationship("InsuranceAccount", back_populates="policies")
    coverages = relationship(
        "InsuranceCoverage", back_populates="policy", cascade="all, delete-orphan"
    )


class InsuranceCoverage(CABase):
    """Individual coverage line within an insurance policy."""

    __tablename__ = "insurance_coverages"

    id = Column(String, primary_key=True, default=_generate_uuid)
    policy_id = Column(
        String,
        ForeignKey("insurance_policies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    account_id = Column(
        String,
        ForeignKey("insurance_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    coverage_type = Column(String, nullable=False)
    limit_amount = Column(Float, nullable=True)
    deductible_amount = Column(Float, nullable=True)
    deductible_type = Column(String, nullable=True)
    sublimit_type = Column(String, nullable=True)
    sublimit_amount = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    tenant_id = Column(String, nullable=False, index=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    policy = relationship("InsurancePolicy", back_populates="coverages")
    account = relationship("InsuranceAccount", back_populates="coverages")
