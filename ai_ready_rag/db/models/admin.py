"""Admin, audit, settings, and reindex models."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text

from ai_ready_rag.db.database import Base
from ai_ready_rag.db.models.base import TimestampMixin, generate_uuid


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, nullable=False)
    event_type = Column(String, nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    user_email = Column(String, nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    request_id = Column(String, nullable=True)


class AdminSetting(Base):
    __tablename__ = "admin_settings"

    id = Column(String, primary_key=True, default=generate_uuid)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)  # JSON encoded
    updated_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemSetup(TimestampMixin, Base):
    """Tracks system setup state for first-run wizard."""

    __tablename__ = "system_setup"

    id = Column(String, primary_key=True, default=generate_uuid)
    setup_complete = Column(Boolean, default=False)
    admin_password_changed = Column(Boolean, default=False)
    setup_completed_at = Column(DateTime, nullable=True)
    setup_completed_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class SettingsAudit(Base):
    """Tracks all admin settings changes for audit trail."""

    __tablename__ = "settings_audit"

    id = Column(String, primary_key=True, default=generate_uuid)
    setting_key = Column(String, nullable=False, index=True)
    old_value = Column(Text, nullable=True)  # JSON encoded
    new_value = Column(Text, nullable=False)  # JSON encoded
    changed_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow)
    change_reason = Column(String, nullable=True)


class ReindexJob(TimestampMixin, Base):
    """Tracks background knowledge base reindex operations."""

    __tablename__ = "reindex_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    status = Column(
        String, default="pending"
    )  # pending, running, paused, completed, failed, aborted
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    failed_documents = Column(Integer, default=0)
    current_document_id = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    triggered_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    settings_changed = Column(Text, nullable=True)  # JSON encoded dict of changed settings
    temp_collection_name = Column(String, nullable=True)  # Temp collection being built

    # Phase 3: Failure handling fields
    failed_document_ids = Column(Text, nullable=True)  # JSON list of failed doc IDs
    last_error = Column(Text, nullable=True)  # Last error message
    retry_count = Column(Integer, default=0)  # Retries for current document
    max_retries = Column(Integer, default=3)  # Max retries before skip
    paused_at = Column(DateTime, nullable=True)  # When job was paused
    paused_reason = Column(String, nullable=True)  # 'failure' or 'user_request'
    auto_skip_failures = Column(Boolean, default=False)  # Skip-all mode
