"""Document management service."""

import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models import Document, Tag
from ai_ready_rag.utils.file_utils import (
    compute_file_hash,
    get_file_extension,
    get_storage_usage,
    validate_file_extension,
)


class DocumentService:
    """Document management business logic."""

    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.storage_path = Path(settings.upload_dir)

    def check_duplicates_by_filename(self, filenames: list[str]) -> tuple[list[dict], list[str]]:
        """Check which filenames already exist in the database.

        Args:
            filenames: List of filenames to check

        Returns:
            Tuple of (duplicates list, unique filenames list)
            Each duplicate is a dict with: filename, existing_id, existing_filename, uploaded_at
        """
        duplicates = []
        unique = []

        for filename in filenames:
            existing = (
                self.db.query(Document).filter(Document.original_filename == filename).first()
            )

            if existing:
                duplicates.append(
                    {
                        "filename": filename,
                        "existing_id": existing.id,
                        "existing_filename": existing.original_filename,
                        "uploaded_at": existing.uploaded_at,
                    }
                )
            else:
                unique.append(filename)

        return duplicates, unique

    async def upload(
        self,
        file: UploadFile,
        tag_ids: list[str],
        uploaded_by: str,
        title: str | None = None,
        description: str | None = None,
        replace: bool = False,
        vector_service=None,
    ) -> Document:
        """Upload and store a document.

        Args:
            file: Uploaded file
            tag_ids: List of tag IDs to assign
            uploaded_by: User ID of uploader
            title: Optional title
            description: Optional description

        Returns:
            Created Document record

        Raises:
            HTTPException: On validation failure
        """
        # Validate file exists
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided",
            )

        # Validate file extension
        extension = get_file_extension(file.filename)
        if not validate_file_extension(file.filename, self.settings.allowed_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{extension}' not allowed. Allowed: {', '.join(self.settings.allowed_extensions)}",
            )

        # Validate tags
        if not tag_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one tag is required",
            )

        tags = self.db.query(Tag).filter(Tag.id.in_(tag_ids)).all()
        if len(tags) != len(tag_ids):
            found_ids = {t.id for t in tags}
            missing = [tid for tid in tag_ids if tid not in found_ids]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tag IDs: {missing}",
            )

        # Read file content to check size
        content = await file.read()
        file_size = len(content)

        # Validate file size
        max_size = self.settings.max_upload_size_mb * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Max size: {self.settings.max_upload_size_mb}MB",
            )

        # Check storage quota
        current_usage = get_storage_usage(self.storage_path)
        max_storage = self.settings.max_storage_gb * 1024 * 1024 * 1024
        if current_usage + file_size > max_storage:
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail={
                    "message": "Storage quota exceeded",
                    "current_usage_gb": round(current_usage / (1024**3), 2),
                    "max_storage_gb": self.settings.max_storage_gb,
                    "file_size_mb": round(file_size / (1024**2), 2),
                },
            )

        # Create document record first to get ID
        document = Document(
            filename="",  # Will update after saving
            original_filename=file.filename,
            file_path="",  # Will update after saving
            file_type=extension,
            file_size=file_size,
            status="pending",
            uploaded_by=uploaded_by,
            title=title,
            description=description,
        )
        self.db.add(document)
        self.db.flush()  # Get the ID

        # Create storage directory
        doc_dir = self.storage_path / document.id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        stored_filename = f"original.{extension}"
        file_path = doc_dir / stored_filename

        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except OSError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {e}",
            ) from e

        # Compute content hash
        content_hash = compute_file_hash(file_path)

        # Check for duplicate
        existing = (
            self.db.query(Document)
            .filter(Document.content_hash == content_hash, Document.id != document.id)
            .first()
        )
        if existing:
            if replace and vector_service is not None:
                # Replace mode: delete existing document atomically
                try:
                    # Delete vectors from vector store
                    await vector_service.delete_document(existing.id)
                except Exception:
                    # Continue even if vector deletion fails
                    pass

                # Delete existing file from storage
                existing_dir = self.storage_path / existing.id
                if existing_dir.exists():
                    shutil.rmtree(existing_dir)

                # Delete existing database record
                self.db.delete(existing)
                self.db.flush()
            else:
                # Capture existing doc info before cleanup
                existing_id = existing.id
                existing_filename = existing.original_filename
                existing_uploaded_at = existing.uploaded_at.isoformat()

                # Clean up saved file
                shutil.rmtree(doc_dir)

                # Expunge the pending document from session (don't rollback - let caller handle)
                self.db.expunge(document)

                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "detail": "Duplicate file detected",
                        "error_code": "DUPLICATE_FILE",
                        "existing_id": existing_id,
                        "existing_filename": existing_filename,
                        "uploaded_at": existing_uploaded_at,
                    },
                )

        # Update document with file info
        document.filename = stored_filename
        document.file_path = str(file_path)
        document.content_hash = content_hash

        # Link tags
        document.tags = tags

        self.db.commit()
        self.db.refresh(document)

        return document

    def get_document(
        self,
        document_id: str,
        user_id: str,
        user_tags: list[str],
        is_admin: bool,
    ) -> Document | None:
        """Get document if user has access.

        Args:
            document_id: Document ID
            user_id: Current user ID
            user_tags: User's assigned tag names
            is_admin: Whether user is admin

        Returns:
            Document if found and accessible, None otherwise
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()

        if not document:
            return None

        # Admins can see all documents
        if is_admin:
            return document

        # Users can only see documents with matching tags
        doc_tag_names = {tag.name for tag in document.tags}
        if not doc_tag_names.intersection(set(user_tags)):
            return None

        return document

    def list_documents(
        self,
        user_id: str,
        user_tags: list[str],
        is_admin: bool,
        limit: int = 20,
        offset: int = 0,
        status_filter: str | None = None,
        tag_id: str | None = None,
        search: str | None = None,
        sort_by: str = "uploaded_at",
        sort_order: str = "desc",
    ) -> tuple[list[Document], int]:
        """List documents with filtering.

        Args:
            user_id: Current user ID
            user_tags: User's assigned tag names
            is_admin: Whether user is admin
            limit: Max results
            offset: Pagination offset
            status_filter: Filter by status
            tag_id: Filter by tag ID
            search: Search term for filename/title
            sort_by: Sort field
            sort_order: asc or desc

        Returns:
            Tuple of (documents list, total count)
        """
        from sqlalchemy import func, or_

        query = self.db.query(Document)

        # Access control for non-admins
        if not is_admin:
            query = query.join(Document.tags).filter(Tag.name.in_(user_tags)).distinct()

        # Filter by status
        if status_filter:
            query = query.filter(Document.status == status_filter)

        # Filter by tag
        if tag_id:
            query = query.join(Document.tags).filter(Tag.id == tag_id)

        # Search filter
        if search:
            search_term = f"%{search.lower()}%"
            query = query.filter(
                or_(
                    func.lower(Document.original_filename).like(search_term),
                    func.lower(Document.title).like(search_term),
                )
            )

        # Get total count before pagination
        total = query.count()

        # Sorting
        sort_column = getattr(Document, sort_by, Document.uploaded_at)
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Pagination
        documents = query.offset(offset).limit(limit).all()

        return documents, total

    async def delete_document(self, document_id: str) -> bool:
        """Delete document, file, and vectors.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()

        if not document:
            return False

        # Delete file from storage
        doc_dir = self.storage_path / document_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

        # Delete database record (cascades to document_tags)
        self.db.delete(document)
        self.db.commit()

        return True

    def recover_stuck_documents(self, max_age_hours: int = 2) -> int:
        """Reset stuck processing documents to pending.

        Args:
            max_age_hours: Maximum hours a document can be in processing state

        Returns:
            Count of recovered documents
        """
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        stuck = (
            self.db.query(Document)
            .filter(
                Document.status == "processing",
                Document.uploaded_at < cutoff,
            )
            .all()
        )

        for doc in stuck:
            doc.status = "pending"
            doc.error_message = None

        self.db.commit()
        return len(stuck)

    def reset_all_processing(self) -> int:
        """Reset all processing documents to pending (startup recovery).

        Returns:
            Count of reset documents
        """
        count = (
            self.db.query(Document)
            .filter(Document.status == "processing")
            .update({"status": "pending", "error_message": None})
        )
        self.db.commit()
        return count

    def delete_all_documents(self) -> int:
        """Delete all documents and their files from storage.

        Returns:
            Count of deleted documents

        Warning:
            This is a destructive operation. Intended for knowledge base reset.
        """
        # Get all documents
        documents = self.db.query(Document).all()
        count = len(documents)

        # Delete files from storage
        for doc in documents:
            doc_dir = self.storage_path / doc.id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)

        # Delete all documents from database
        self.db.query(Document).delete()
        self.db.commit()

        return count
