"""Admin settings persistence service."""

import json
from typing import Any

from sqlalchemy.orm import Session

from ai_ready_rag.db.models import AdminSetting


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
