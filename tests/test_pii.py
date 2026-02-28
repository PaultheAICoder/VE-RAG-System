"""Tests for PII redaction and Fernet encryption."""

import pytest

from ai_ready_rag.core.pii import FernetEncryption, PIIRedactor, RedactionResult


class TestPIIRedactor:
    def setup_method(self):
        self.redactor = PIIRedactor()

    def test_redacts_ssn(self):
        result = self.redactor.redact("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "[SSN-REDACTED]" in result

    def test_redacts_phone(self):
        result = self.redactor.redact("Call 555-123-4567 for details")
        assert "555-123-4567" not in result
        assert "[PHONE-REDACTED]" in result

    def test_redacts_email(self):
        result = self.redactor.redact("Contact john.doe@example.com")
        assert "john.doe@example.com" not in result
        assert "[EMAIL-REDACTED]" in result

    def test_redacts_dob(self):
        result = self.redactor.redact("Born 01/15/1980")
        assert "01/15/1980" not in result
        assert "[DOB-REDACTED]" in result

    def test_no_pii_unchanged(self):
        text = "The insurance policy covers $1,000,000 in liability."
        result = self.redactor.redact(text)
        assert result == text

    def test_redact_with_report(self):
        text = "SSN: 123-45-6789, Email: test@test.com"
        report = self.redactor.redact_with_report(text)
        assert isinstance(report, RedactionResult)
        assert report.redacted_count >= 2
        assert "ssn" in report.patterns_matched or "email" in report.patterns_matched

    def test_contains_pii_true(self):
        assert self.redactor.contains_pii("My email is a@b.com") is True

    def test_contains_pii_false(self):
        assert self.redactor.contains_pii("No PII here at all.") is False

    def test_multiple_instances_in_text(self):
        text = "Call 555-111-2222 or 555-333-4444"
        result = self.redactor.redact(text)
        assert "555-111-2222" not in result
        assert "555-333-4444" not in result


class TestFernetEncryption:
    def test_generate_key(self):
        key = FernetEncryption.generate_key()
        assert isinstance(key, str)
        assert len(key) > 30

    def test_encrypt_decrypt_roundtrip(self):
        key = FernetEncryption.generate_key()
        crypto = FernetEncryption(key)
        plaintext = "John Doe"
        encrypted = crypto.encrypt(plaintext)
        assert encrypted != plaintext
        assert crypto.decrypt(encrypted) == plaintext

    def test_encrypt_empty_string(self):
        key = FernetEncryption.generate_key()
        crypto = FernetEncryption(key)
        assert crypto.encrypt("") == ""

    def test_decrypt_empty_string(self):
        key = FernetEncryption.generate_key()
        crypto = FernetEncryption(key)
        assert crypto.decrypt("") == ""

    def test_is_encrypted_true(self):
        key = FernetEncryption.generate_key()
        crypto = FernetEncryption(key)
        encrypted = crypto.encrypt("test value")
        assert crypto.is_encrypted(encrypted) is True

    def test_is_encrypted_false_for_plaintext(self):
        key = FernetEncryption.generate_key()
        crypto = FernetEncryption(key)
        assert crypto.is_encrypted("plain text value") is False

    def test_from_env(self, monkeypatch):
        key = FernetEncryption.generate_key()
        monkeypatch.setenv("VAULTIQ_ENCRYPTION_KEY", key)
        crypto = FernetEncryption.from_env()
        assert crypto.encrypt("test") != "test"

    def test_from_env_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("VAULTIQ_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="VAULTIQ_ENCRYPTION_KEY"):
            FernetEncryption.from_env()
