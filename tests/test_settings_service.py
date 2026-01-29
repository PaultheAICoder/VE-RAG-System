"""Tests for SettingsService."""

from ai_ready_rag.services.settings_service import SettingsService


class TestSettingsService:
    """Test SettingsService CRUD operations."""

    def test_get_nonexistent_returns_none(self, db):
        """Getting nonexistent key returns None."""
        service = SettingsService(db)
        assert service.get("nonexistent") is None

    def test_set_creates_new_setting(self, db, admin_user):
        """Set creates new setting when key doesn't exist."""
        service = SettingsService(db)
        setting = service.set("chat_model", "llama3.2", updated_by=admin_user.id)

        assert setting.key == "chat_model"
        assert setting.updated_by == admin_user.id

        # Verify persisted
        value = service.get("chat_model")
        assert value == "llama3.2"

    def test_set_updates_existing_setting(self, db, admin_user):
        """Set updates existing setting when key exists."""
        service = SettingsService(db)

        # Create initial
        service.set("enable_ocr", True, updated_by=admin_user.id)

        # Update
        service.set("enable_ocr", False, updated_by=admin_user.id)

        assert service.get("enable_ocr") is False

    def test_get_all_returns_dict(self, db, admin_user):
        """get_all returns all settings as dict."""
        service = SettingsService(db)

        service.set("key1", "value1")
        service.set("key2", 123)
        service.set("key3", {"nested": True})

        all_settings = service.get_all()

        assert all_settings == {
            "key1": "value1",
            "key2": 123,
            "key3": {"nested": True},
        }

    def test_delete_removes_setting(self, db):
        """delete removes existing setting."""
        service = SettingsService(db)

        service.set("to_delete", "value")
        assert service.get("to_delete") == "value"

        result = service.delete("to_delete")

        assert result is True
        assert service.get("to_delete") is None

    def test_delete_nonexistent_returns_false(self, db):
        """delete returns False for nonexistent key."""
        service = SettingsService(db)
        assert service.delete("nonexistent") is False

    def test_get_with_defaults(self, db):
        """get_with_defaults merges stored with defaults."""
        service = SettingsService(db)

        # Store one setting
        service.set("chat_model", "deepseek-r1:32b")

        defaults = {
            "chat_model": "qwen3:8b",
            "enable_ocr": True,
            "ocr_language": "eng",
        }

        result = service.get_with_defaults(defaults)

        # Stored value overrides default
        assert result["chat_model"] == "deepseek-r1:32b"
        # Defaults fill in missing
        assert result["enable_ocr"] is True
        assert result["ocr_language"] == "eng"

    def test_json_encoding_types(self, db):
        """Various Python types are correctly JSON encoded/decoded."""
        service = SettingsService(db)

        # String
        service.set("str_val", "hello")
        assert service.get("str_val") == "hello"

        # Integer
        service.set("int_val", 42)
        assert service.get("int_val") == 42

        # Boolean
        service.set("bool_val", True)
        assert service.get("bool_val") is True

        # List
        service.set("list_val", [1, 2, 3])
        assert service.get("list_val") == [1, 2, 3]

        # Dict
        service.set("dict_val", {"a": 1, "b": 2})
        assert service.get("dict_val") == {"a": 1, "b": 2}
