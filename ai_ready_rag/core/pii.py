"""PII redaction and Fernet encryption utilities.

Two capabilities:
1. PIIRedactor — regex-based redaction of PII from text before LLM processing
2. FernetEncryption — symmetric column encryption for database storage

Usage:
    # Redact PII from text
    redactor = PIIRedactor()
    clean_text = redactor.redact("Call John at 555-123-4567 or john@example.com")

    # Encrypt/decrypt sensitive columns
    crypto = FernetEncryption.from_env()  # reads VAULTIQ_ENCRYPTION_KEY
    encrypted = crypto.encrypt("123-45-6789")
    original = crypto.decrypt(encrypted)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Regex Patterns
# ---------------------------------------------------------------------------

PII_PATTERNS: list[tuple[str, str, str]] = [
    # (name, pattern, replacement)
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]"),
    ("ssn_no_dashes", r"\b\d{9}\b(?!\d)", "[SSN-REDACTED]"),
    ("phone_us", r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE-REDACTED]"),
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL-REDACTED]"),
    (
        "dob_slash",
        r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b",
        "[DOB-REDACTED]",
    ),
    ("dob_dash", r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b", "[DOB-REDACTED]"),
    ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CC-REDACTED]"),
    ("routing_number", r"\b\d{9}\b(?=.{0,20}routing)", "[ROUTING-REDACTED]"),
]


@dataclass
class RedactionResult:
    original_length: int
    redacted_text: str
    redacted_count: int
    patterns_matched: list[str] = field(default_factory=list)


class PIIRedactor:
    """Regex-based PII redaction for text content.

    Redacts SSNs, phone numbers, email addresses, dates of birth,
    and credit card numbers before sending text to LLMs.
    """

    def __init__(self, extra_patterns: list[tuple[str, str, str]] | None = None) -> None:
        patterns = list(PII_PATTERNS) + (extra_patterns or [])
        self._compiled = [
            (name, re.compile(pattern, re.IGNORECASE), replacement)
            for name, pattern, replacement in patterns
        ]

    def redact(self, text: str) -> str:
        """Return text with PII replaced by redaction tokens."""
        result = text
        for _, pattern, replacement in self._compiled:
            result = pattern.sub(replacement, result)
        return result

    def redact_with_report(self, text: str) -> RedactionResult:
        """Redact and return a report of what was found."""
        result = text
        patterns_matched: list[str] = []
        total_count = 0

        for name, pattern, replacement in self._compiled:
            matches = pattern.findall(result)
            if matches:
                patterns_matched.append(name)
                total_count += len(matches)
                result = pattern.sub(replacement, result)

        return RedactionResult(
            original_length=len(text),
            redacted_text=result,
            redacted_count=total_count,
            patterns_matched=patterns_matched,
        )

    def contains_pii(self, text: str) -> bool:
        """Return True if the text contains any detectable PII."""
        return any(pattern.search(text) for _, pattern, _ in self._compiled)


# ---------------------------------------------------------------------------
# Fernet Column Encryption
# ---------------------------------------------------------------------------


class FernetEncryption:
    """Symmetric Fernet encryption for sensitive database columns.

    Fernet guarantees that a message encrypted using it cannot be
    manipulated or read without the key.

    Key management:
    - Load from VAULTIQ_ENCRYPTION_KEY env var (base64-encoded 32-byte key)
    - Generate a new key with FernetEncryption.generate_key()
    """

    def __init__(self, key: str | bytes) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as err:
            raise RuntimeError(
                "cryptography package not installed. "
                "Add cryptography>=41.0.0 to requirements-wsl.txt"
            ) from err
        if isinstance(key, str):
            key = key.encode()
        self._fernet = Fernet(key)

    @classmethod
    def from_env(cls, env_var: str = "VAULTIQ_ENCRYPTION_KEY") -> FernetEncryption:
        """Create instance from environment variable."""
        key = os.environ.get(env_var)
        if not key:
            raise ValueError(
                f"Environment variable {env_var!r} is not set. "
                "Generate a key with: FernetEncryption.generate_key()"
            )
        return cls(key)

    @classmethod
    def generate_key(cls) -> str:
        """Generate a new Fernet key (base64-encoded). Store securely."""
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode()

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        if not plaintext:
            return plaintext
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext and return plaintext."""
        if not ciphertext:
            return ciphertext
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def is_encrypted(self, value: str) -> bool:
        """Heuristic check if a value looks like a Fernet token."""
        # Fernet tokens start with 'gAAA' (version byte 0x80, base64 encoded)
        return value.startswith("gAAA") and len(value) > 50
