"""SQLite ↔ pgvector reconciliation service.

Detects and optionally repairs drift between SQLite document metadata
and the pgvector chunk_vectors table.
"""

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from ai_ready_rag.db.models import Document

logger = logging.getLogger(__name__)


async def reconcile(
    db: Session,
    vector_service,
    dry_run: bool = True,
) -> dict:
    """Detect and optionally repair SQLite ↔ pgvector drift.

    Scenarios detected:
    - ghost_docs: SQLite status=ready but 0 vectors in chunk_vectors
    - orphan_vectors: chunk_vectors has rows for document_id not in SQLite
    - chunk_count_mismatch: SQLite chunk_count != actual vector count

    Args:
        db: SQLAlchemy session
        vector_service: Initialized vector service instance (PgVectorService or VectorService)
        dry_run: If True, only detect issues. If False, also repair.

    Returns:
        Dict with total_documents, synced count, issues list, and repairs list.
    """
    issues: list[dict] = []
    repairs: list[dict] = []

    # 1. Get all documents from SQLite
    all_docs = db.query(Document).all()
    sqlite_doc_ids = {doc.id for doc in all_docs}
    ready_docs = [d for d in all_docs if d.status == "ready"]

    # 2. Get all unique document_ids from chunk_vectors (pgvector SQL)
    tenant_id = getattr(
        vector_service, "_tenant_id", getattr(vector_service, "tenant_id", "default")
    )
    vector_doc_ids: set[str] = set()
    vector_doc_counts: dict[str, int] = {}

    rows = db.execute(
        text(
            "SELECT document_id, COUNT(*) AS chunk_count "
            "FROM chunk_vectors "
            "WHERE tenant_id = :tenant "
            "GROUP BY document_id"
        ),
        {"tenant": tenant_id},
    ).fetchall()
    for row in rows:
        vector_doc_ids.add(row[0])
        vector_doc_counts[row[0]] = row[1]

    # 3. Detect ghost docs: ready in SQLite but 0 vectors in chunk_vectors
    for doc in ready_docs:
        vector_count = vector_doc_counts.get(doc.id, 0)
        if vector_count == 0:
            issues.append(
                {
                    "document_id": doc.id,
                    "filename": doc.original_filename,
                    "issue": "ghost_doc",
                    "detail": "status=ready in SQLite but 0 vectors in chunk_vectors",
                    "sqlite_chunks": doc.chunk_count,
                    "vector_chunks": 0,
                }
            )
            if not dry_run:
                doc.status = "pending"
                doc.chunk_count = None
                doc.error_message = None
                repairs.append(
                    {
                        "document_id": doc.id,
                        "action": "reset_to_pending",
                    }
                )

    # 4. Detect orphan vectors: in chunk_vectors but not in SQLite
    orphan_ids = vector_doc_ids - sqlite_doc_ids
    for orphan_id in orphan_ids:
        issues.append(
            {
                "document_id": orphan_id,
                "filename": None,
                "issue": "orphan_vectors",
                "detail": "Vectors exist in chunk_vectors but no SQLite record",
                "sqlite_chunks": None,
                "vector_chunks": vector_doc_counts.get(orphan_id, 0),
            }
        )
        if not dry_run:
            try:
                await vector_service.delete_document(orphan_id)
                repairs.append(
                    {
                        "document_id": orphan_id,
                        "action": "deleted_orphan_vectors",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to delete orphan vectors for {orphan_id}: {e}")

    # 5. Detect chunk count mismatches
    for doc in ready_docs:
        if doc.id in orphan_ids:
            continue  # Already handled
        vector_count = vector_doc_counts.get(doc.id, 0)
        if vector_count == 0:
            continue  # Already handled as ghost_doc
        if doc.chunk_count is not None and doc.chunk_count != vector_count:
            old_count = doc.chunk_count
            issues.append(
                {
                    "document_id": doc.id,
                    "filename": doc.original_filename,
                    "issue": "chunk_count_mismatch",
                    "detail": f"SQLite chunk_count={doc.chunk_count} != vector chunks={vector_count}",
                    "sqlite_chunks": doc.chunk_count,
                    "vector_chunks": vector_count,
                }
            )
            if not dry_run:
                doc.chunk_count = vector_count
                repairs.append(
                    {
                        "document_id": doc.id,
                        "action": "updated_chunk_count",
                        "old_value": old_count,
                        "new_value": vector_count,
                    }
                )

    if not dry_run:
        db.commit()

    synced = len(ready_docs) - sum(
        1 for i in issues if i["issue"] in ("ghost_doc", "chunk_count_mismatch")
    )

    return {
        "total_documents": len(all_docs),
        "total_vector_documents": len(vector_doc_ids),
        "synced": max(synced, 0),
        "issues": issues,
        "repairs": repairs,
        "dry_run": dry_run,
    }
