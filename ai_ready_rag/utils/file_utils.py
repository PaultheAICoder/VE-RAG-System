"""File handling utilities for document management."""

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_storage_usage(upload_dir: Path) -> int:
    """Calculate total storage used in upload directory.

    Args:
        upload_dir: Path to uploads directory

    Returns:
        Total size in bytes
    """
    if not upload_dir.exists():
        return 0

    total = 0
    for file_path in upload_dir.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def validate_file_extension(filename: str, allowed_extensions: list[str]) -> bool:
    """Check if file extension is allowed.

    Args:
        filename: Original filename
        allowed_extensions: List of allowed extensions (without dots)

    Returns:
        True if extension is allowed
    """
    ext = get_file_extension(filename)
    return ext in allowed_extensions


def get_file_extension(filename: str) -> str:
    """Extract file extension from filename.

    Args:
        filename: Original filename

    Returns:
        Extension without leading dot, lowercase
    """
    return Path(filename).suffix.lstrip(".").lower()


def get_mime_type(extension: str) -> str:
    """Get MIME type for a file extension.

    Args:
        extension: File extension (without dot)

    Returns:
        MIME type string
    """
    mime_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "txt": "text/plain",
        "md": "text/markdown",
        "html": "text/html",
        "csv": "text/csv",
    }
    return mime_types.get(extension.lower(), "application/octet-stream")
