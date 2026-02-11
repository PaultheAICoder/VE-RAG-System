#!/usr/bin/env python3
"""List all documents stored in the vector database.

Usage:
    python scripts/list_documents.py
    python scripts/list_documents.py --status ready
    python scripts/list_documents.py --status failed
    python scripts/list_documents.py --tag hr
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_ready_rag.db.database import SessionLocal, init_db
from ai_ready_rag.db.models import Document, Tag, document_tags


def main():
    parser = argparse.ArgumentParser(description="List documents in the RAG system")
    parser.add_argument("--status", help="Filter by status (pending, processing, ready, failed)")
    parser.add_argument("--tag", help="Filter by tag name")
    parser.add_argument("--format", choices=["table", "ids"], default="table", help="Output format")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    try:
        query = db.query(Document).order_by(Document.uploaded_at.desc())

        if args.status:
            query = query.filter(Document.status == args.status)

        if args.tag:
            query = query.join(document_tags).join(Tag).filter(Tag.name == args.tag)

        docs = query.all()

        if not docs:
            print("No documents found.")
            return

        if args.format == "ids":
            for doc in docs:
                print(f"{doc.id}  {doc.original_filename}")
            return

        # Table format
        print(
            f"{'ID':<38} {'Filename':<40} {'Type':<5} {'Status':<10} {'Chunks':>6} {'Words':>7} {'Tags'}"
        )
        print("-" * 140)

        for doc in docs:
            tags = ", ".join(t.name for t in doc.tags) if doc.tags else ""
            chunks = str(doc.chunk_count) if doc.chunk_count else "-"
            words = str(doc.word_count) if doc.word_count else "-"
            filename = doc.original_filename or doc.filename
            if len(filename) > 38:
                filename = filename[:35] + "..."

            print(
                f"{doc.id:<38} {filename:<40} {doc.file_type:<5} {doc.status:<10} "
                f"{chunks:>6} {words:>7} {tags}"
            )

        print(f"\nTotal: {len(docs)} documents")

        # Summary by status
        statuses = {}
        for doc in docs:
            statuses[doc.status] = statuses.get(doc.status, 0) + 1
        status_str = ", ".join(f"{s}: {c}" for s, c in sorted(statuses.items()))
        print(f"Status: {status_str}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
