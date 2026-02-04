"""Admin settings persistence service."""

import json
import os
from typing import Any

from sqlalchemy.orm import Session

from ai_ready_rag.db.models import AdminSetting, SettingsAudit

# Default values for RAG settings
RAG_SETTINGS_DEFAULTS = {
    # Retrieval settings
    "retrieval_top_k": 5,
    "retrieval_min_score": 0.3,
    "retrieval_enable_expansion": True,
    # LLM settings
    "llm_temperature": 0.1,
    "llm_max_response_tokens": 2048,
    "llm_confidence_threshold": 40,
}

# Default values for cache settings
CACHE_SETTINGS_DEFAULTS = {
    "cache_enabled": True,
    "cache_ttl_hours": 24,
    "cache_max_entries": 1000,
    "cache_semantic_threshold": 0.95,
    "cache_min_confidence": 40,
    "cache_auto_warm_enabled": True,
    "cache_auto_warm_count": 20,
}


def get_model_setting(key: str, default: Any = None) -> Any:
    """Get a model setting from database with fallback to default.

    This is a standalone function that creates its own db session,
    suitable for use from services which may not have a session.

    Args:
        key: Setting key (e.g., 'embedding_model', 'chat_model')
        default: Default value if not found in database

    Returns:
        Setting value from database, or default if not found
    """
    from ai_ready_rag.db.database import SessionLocal

    db = SessionLocal()
    try:
        setting = db.query(AdminSetting).filter(AdminSetting.key == key).first()
        if setting is not None:
            return json.loads(setting.value)
        return default
    finally:
        db.close()


def get_cache_setting(key: str, default: Any = None) -> Any:
    """Get a cache setting from database with fallback to default.

    This is a standalone function that creates its own db session,
    suitable for use from services which may not have a session.

    Args:
        key: Setting key (e.g., 'cache_enabled', 'cache_ttl_hours')
        default: Default value if not found in database

    Returns:
        Setting value from database, or default if not found
    """
    from ai_ready_rag.db.database import SessionLocal

    # Use provided default or look up in our defaults dict
    if default is None:
        default = CACHE_SETTINGS_DEFAULTS.get(key)

    db = SessionLocal()
    try:
        setting = db.query(AdminSetting).filter(AdminSetting.key == key).first()
        if setting is not None:
            return json.loads(setting.value)
        return default
    finally:
        db.close()


def get_rag_setting(key: str, default: Any = None) -> Any:
    """Get a RAG setting from database with fallback to default.

    This is a standalone function that creates its own db session,
    suitable for use from the RAG service which may not have a session.

    Args:
        key: Setting key (e.g., 'retrieval_top_k', 'llm_temperature')
        default: Default value if not found in database

    Returns:
        Setting value from database, or default if not found
    """
    from ai_ready_rag.db.database import SessionLocal

    # Use provided default or look up in our defaults dict
    if default is None:
        default = RAG_SETTINGS_DEFAULTS.get(key)

    db = SessionLocal()
    try:
        setting = db.query(AdminSetting).filter(AdminSetting.key == key).first()
        if setting is not None:
            return json.loads(setting.value)
        return default
    finally:
        db.close()


class SettingsService:
    """Service for admin settings CRUD operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, key: str) -> Any | None:
        """Get setting value by key. Returns None if not found."""
        setting = self.db.query(AdminSetting).filter(AdminSetting.key == key).first()
        if setting is None:
            return None
        return json.loads(setting.value)

    def get_with_env_fallback(
        self, key: str, env_key: str | None = None, default: Any = None
    ) -> Any:
        """Get setting value with fallback to environment variable, then default.

        Priority: DB setting > Environment variable > Default value
        """
        # First check database
        db_value = self.get(key)
        if db_value is not None:
            return db_value

        # Then check environment variable
        if env_key:
            env_value = os.environ.get(env_key)
            if env_value is not None:
                # Try to parse as JSON (for booleans, numbers)
                try:
                    return json.loads(env_value.lower())
                except (json.JSONDecodeError, AttributeError):
                    return env_value

        # Fall back to default
        return default

    def set(self, key: str, value: Any, updated_by: str | None = None) -> AdminSetting:
        """Set setting value (create or update)."""
        setting = self.db.query(AdminSetting).filter(AdminSetting.key == key).first()

        json_value = json.dumps(value)

        if setting is None:
            # Create new
            setting = AdminSetting(
                key=key,
                value=json_value,
                updated_by=updated_by,
            )
            self.db.add(setting)
        else:
            # Update existing
            setting.value = json_value
            setting.updated_by = updated_by

        self.db.commit()
        self.db.refresh(setting)
        return setting

    def set_with_audit(
        self,
        key: str,
        value: Any,
        changed_by: str | None = None,
        reason: str | None = None,
    ) -> AdminSetting:
        """Set setting value with audit trail.

        Records the old value, new value, who changed it, and optional reason
        in the settings_audit table.
        """
        # Get old value before update
        old_value = self.get(key)
        old_json = json.dumps(old_value) if old_value is not None else None
        new_json = json.dumps(value)

        # Create audit record
        audit = SettingsAudit(
            setting_key=key,
            old_value=old_json,
            new_value=new_json,
            changed_by=changed_by,
            change_reason=reason,
        )
        self.db.add(audit)

        # Update the setting
        setting = self.set(key, value, updated_by=changed_by)
        return setting

    def get_audit_history(
        self, key: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[SettingsAudit]:
        """Get audit history, optionally filtered by key.

        Args:
            key: Optional setting key to filter by
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of SettingsAudit records, newest first
        """
        query = self.db.query(SettingsAudit)
        if key:
            query = query.filter(SettingsAudit.setting_key == key)
        query = query.order_by(SettingsAudit.changed_at.desc())
        query = query.offset(offset).limit(limit)
        return query.all()

    def get_all(self) -> dict[str, Any]:
        """Get all settings as dictionary."""
        settings = self.db.query(AdminSetting).all()
        return {s.key: json.loads(s.value) for s in settings}

    def delete(self, key: str) -> bool:
        """Delete setting by key. Returns True if deleted, False if not found."""
        setting = self.db.query(AdminSetting).filter(AdminSetting.key == key).first()
        if setting is None:
            return False
        self.db.delete(setting)
        self.db.commit()
        return True

    def get_with_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Get all settings, filling in defaults for missing keys."""
        stored = self.get_all()
        result = defaults.copy()
        result.update(stored)
        return result
