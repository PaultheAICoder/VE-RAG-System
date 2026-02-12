#!/usr/bin/env python3
"""Recursively upload a directory of documents via /api/documents/upload.

Usage:
  python scripts/upload_directory.py \
    --dir /path/to/docs \
    --tag-id <tag-id> [--tag-id <tag-id> ...] \
    --api-url http://localhost:8502 \
    --token <jwt-token>

Notes:
- Uses the existing single-file upload endpoint for compatibility.
- Walks the directory recursively and filters by supported extensions.
- Uploads sequentially by default to avoid overloading OCR/ARQ workers.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".eml",
    ".msg",
}


@dataclass
class UploadResult:
    path: Path
    status: str
    detail: str


def discover_files(root: Path) -> list[Path]:
    """Return recursively discovered files with supported extensions."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def build_multipart_data(
    file_path: Path,
    tag_ids: Iterable[str],
) -> tuple[list[tuple[str, str]], dict[str, tuple[str, object, str]]]:
    """Create form fields and file payload for multipart upload."""
    form_fields = [("tag_ids", tid) for tid in tag_ids]
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_field = {
        "file": (file_path.name, file_path.open("rb"), content_type),
    }
    return form_fields, file_field


def upload_one(
    client: httpx.Client,
    api_url: str,
    file_path: Path,
    tag_ids: list[str],
    replace: bool,
) -> UploadResult:
    """Upload one file and return result."""
    endpoint = f"{api_url.rstrip('/')}/api/documents/upload"
    params = {"replace": "true"} if replace else None

    data, files = build_multipart_data(file_path, tag_ids)
    try:
        response = client.post(endpoint, data=data, files=files, params=params)
    finally:
        files["file"][1].close()

    if response.status_code in (200, 201, 202):
        return UploadResult(file_path, "uploaded", "ok")

    detail = response.text
    try:
        payload = response.json()
        if isinstance(payload, dict):
            if isinstance(payload.get("detail"), str):
                detail = payload["detail"]
            elif isinstance(payload.get("detail"), dict):
                detail = json.dumps(payload["detail"], ensure_ascii=True)
    except Exception:
        pass

    return UploadResult(file_path, "failed", f"HTTP {response.status_code}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively upload directory documents")
    parser.add_argument("--dir", required=True, help="Directory to upload recursively")
    parser.add_argument(
        "--tag-id",
        dest="tag_ids",
        action="append",
        required=True,
        help="Tag ID to assign (repeat for multiple)",
    )
    parser.add_argument("--api-url", default="http://localhost:8502", help="Base API URL")
    parser.add_argument("--token", required=True, help="JWT bearer token")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Use replace=true when duplicate content is found",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without sending requests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dir).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: directory not found: {root}")
        return 2

    files = discover_files(root)
    if not files:
        print("No supported files found.")
        return 0

    print(f"Found {len(files)} supported files under {root}")
    for p in files:
        print(f"  - {p}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    headers = {"Authorization": f"Bearer {args.token}"}
    timeout = httpx.Timeout(120.0)

    uploaded = 0
    failed = 0
    results: list[UploadResult] = []

    with httpx.Client(headers=headers, timeout=timeout) as client:
        for file_path in files:
            result = upload_one(
                client=client,
                api_url=args.api_url,
                file_path=file_path,
                tag_ids=args.tag_ids,
                replace=args.replace,
            )
            results.append(result)
            if result.status == "uploaded":
                uploaded += 1
                print(f"[OK] {file_path}")
            else:
                failed += 1
                print(f"[FAIL] {file_path} -> {result.detail}")

    print("\nSummary")
    print(f"  Uploaded: {uploaded}")
    print(f"  Failed:   {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
