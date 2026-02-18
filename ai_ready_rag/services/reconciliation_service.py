"""SQLite ↔ Qdrant reconciliation service.

Detects and optionally repairs drift between SQLite document metadata
and Qdrant vector store.
"""

import logging

from sqlalchemy.orm import Session

from ai_ready_rag.db.models import Document

logger = logging.getLogger(__name__)


async def reconcile(
    db: Session,
    vector_service,
    dry_run: bool = True,
) -> dict:
    """Detect and optionally repair SQLite ↔ Qdrant drift.

    Scenarios detected:
    - ghost_docs: SQLite status=ready but 0 vectors in Qdrant
    - orphan_vectors: Qdrant has vectors for document_id not in SQLite
    - chunk_count_mismatch: SQLite chunk_count != actual Qdrant vector count

    Args:
        db: SQLAlchemy session
        vector_service: Initialized VectorService instance
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

    # 2. Get all unique document_ids from Qdrant (scroll in batches)
    qdrant_doc_ids: set[str] = set()
    qdrant_doc_counts: dict[str, int] = {}

    from qdrant_client import models

    offset = None
    while True:
        results, offset = await vector_service._qdrant.scroll(
            collection_name=vector_service.collection_name,
            limit=1000,
            offset=offset,
            with_payload=["document_id"],
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=vector_service.tenant_id),
                    )
                ]
            ),
        )
        if not results:
            break
        for point in results:
            if point.payload and "document_id" in point.payload:
                doc_id = point.payload["document_id"]
                qdrant_doc_ids.add(doc_id)
                qdrant_doc_counts[doc_id] = qdrant_doc_counts.get(doc_id, 0) + 1
        if offset is None:
            break

    # 3. Detect ghost docs: ready in SQLite but 0 vectors in Qdrant
    for doc in ready_docs:
        qdrant_count = qdrant_doc_counts.get(doc.id, 0)
        if qdrant_count == 0:
            issues.append(
                {
                    "document_id": doc.id,
                    "filename": doc.original_filename,
                    "issue": "ghost_doc",
                    "detail": "status=ready in SQLite but 0 vectors in Qdrant",
                    "sqlite_chunks": doc.chunk_count,
                    "qdrant_chunks": 0,
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

    # 4. Detect orphan vectors: in Qdrant but not in SQLite
    orphan_ids = qdrant_doc_ids - sqlite_doc_ids
    for orphan_id in orphan_ids:
        issues.append(
            {
                "document_id": orphan_id,
                "filename": None,
                "issue": "orphan_vectors",
                "detail": "Vectors exist in Qdrant but no SQLite record",
                "sqlite_chunks": None,
                "qdrant_chunks": qdrant_doc_counts.get(orphan_id, 0),
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
        qdrant_count = qdrant_doc_counts.get(doc.id, 0)
        if qdrant_count == 0:
            continue  # Already handled as ghost_doc
        if doc.chunk_count is not None and doc.chunk_count != qdrant_count:
            issues.append(
                {
                    "document_id": doc.id,
                    "filename": doc.original_filename,
                    "issue": "chunk_count_mismatch",
                    "detail": f"SQLite chunk_count={doc.chunk_count} != Qdrant vectors={qdrant_count}",
                    "sqlite_chunks": doc.chunk_count,
                    "qdrant_chunks": qdrant_count,
                }
            )
            if not dry_run:
                doc.chunk_count = qdrant_count
                repairs.append(
                    {
                        "document_id": doc.id,
                        "action": "updated_chunk_count",
                        "old_value": doc.chunk_count,
                        "new_value": qdrant_count,
                    }
                )

    if not dry_run:
        db.commit()

    synced = len(ready_docs) - sum(
        1 for i in issues if i["issue"] in ("ghost_doc", "chunk_count_mismatch")
    )

    return {
        "total_documents": len(all_docs),
        "total_qdrant_documents": len(qdrant_doc_ids),
        "synced": max(synced, 0),
        "issues": issues,
        "repairs": repairs,
        "dry_run": dry_run,
    }
