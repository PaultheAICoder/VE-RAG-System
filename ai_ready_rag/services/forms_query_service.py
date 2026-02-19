"""Forms-aware query service for RAG context injection.

Queries structured form data from forms_data.db (populated by ingestkit-forms)
and returns synthetic SearchResult objects for the RAG pipeline.
"""

import json
import logging
import re
import sqlite3
from fnmatch import fnmatch
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models.base import document_tags
from ai_ready_rag.db.models.document import Document
from ai_ready_rag.db.models.user import Tag
from ai_ready_rag.services.forms_processing_service import ACORD_25_GROUPS
from ai_ready_rag.services.vector_service import SearchResult

logger = logging.getLogger(__name__)

# Intent labels that should trigger forms DB lookup
FORMS_ELIGIBLE_INTENTS: set[str] = {
    "certificate",
    "active_policy",
    "policy_detail",
    "gl",
    "workers_comp",
    "umbrella",
    "crime",
    "do",
    "epli",
}

# Maps intent labels to which ACORD_25_GROUPS to return.
# None means all groups.
INTENT_TO_GROUPS: dict[str, list[str] | None] = {
    "certificate": None,
    "active_policy": None,
    "policy_detail": None,
    "gl": ["General Liability"],
    "workers_comp": ["Workers Comp"],
    "umbrella": ["Umbrella/Excess"],
    "crime": ["Other"],
    "do": ["Other"],
    "epli": ["Other"],
}


def field_name_to_label(field_name: str) -> str:
    """Convert a form field name to a human-readable label.

    Examples:
        GeneralLiability_EachOccurrenceLimit -> Each Occurrence Limit
        Producer_ContactName -> Contact Name
    """
    parts = field_name.split("_", 1)
    name = parts[-1] if len(parts) > 1 else parts[0]
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = name.replace("_", " ")
    return name.strip()


def strip_column_prefix(col_name: str) -> str:
    """Strip F[0].P1[0]. prefix and trailing [0] from form field column names.

    Example: "F[0].P1[0].GeneralLiability_EachOccurrence_LimitAmount_A[0]"
             -> "GeneralLiability_EachOccurrence_LimitAmount_A"
    """
    stripped = re.sub(r"^F\[\d+\]\.P\d+\[\d+\]\.", "", col_name)
    stripped = re.sub(r"\[\d+\]$", "", stripped)
    return stripped


def _matches_group(field_name: str, patterns: list[str]) -> bool:
    """Check if a field name matches any fnmatch pattern in a group."""
    return any(fnmatch(field_name, pat) for pat in patterns)


class FormsQueryService:
    """Queries structured form data from forms_data.db for RAG context."""

    def __init__(self, settings: Settings):
        self._forms_db_path = settings.forms_db_path

    def query_forms(
        self,
        intent,  # QueryIntent (import avoided for circular dep)
        user_tags: list[str] | None,
        db: Session,
        max_results: int = 4,
    ) -> list[SearchResult]:
        """Query forms DB for structured data matching the intent.

        1. Find accessible documents with forms data (tag-filtered)
        2. Determine relevant field groups from intent
        3. Read fields from forms_data.db
        4. Format as synthetic SearchResult objects
        """
        if not Path(self._forms_db_path).exists():
            return []

        # Determine which field groups to return
        field_groups = self._get_field_groups(intent)
        if field_groups is not None and not field_groups:
            return []  # Intent not mapped

        # Find accessible documents with forms data, most recent first
        documents = self._get_accessible_form_documents(user_tags, db)
        if not documents:
            return []

        documents = self._sort_by_recency(documents)

        results: list[SearchResult] = []
        chunk_index = 0

        for doc_rank, doc in enumerate(documents):
            if chunk_index >= max_results:
                break

            # Parse table names from JSON
            try:
                table_names = json.loads(doc.forms_db_table_names)
            except (json.JSONDecodeError, TypeError):
                continue

            ingest_key = doc.forms_ingest_key
            if not ingest_key or not table_names:
                continue

            # Read fields from forms_data.db
            grouped_fields = self._read_form_fields(table_names[0], ingest_key, field_groups)

            # Most recent doc (rank 0) gets 0.97, older docs get 0.93
            score = 0.97 if doc_rank == 0 else 0.93

            for group_name, fields in grouped_fields.items():
                if not fields or chunk_index >= max_results:
                    continue

                result = self._format_as_search_result(doc, group_name, fields, chunk_index, score)
                results.append(result)
                chunk_index += 1

        return results

    def _get_accessible_form_documents(
        self, user_tags: list[str] | None, db: Session
    ) -> list[Document]:
        """Get documents that have forms data and are accessible to the user."""
        stmt = (
            select(Document)
            .where(Document.forms_db_table_names.isnot(None))
            .where(Document.status == "ready")
        )

        # Apply access control: None = admin bypass, [] = public only
        if user_tags is not None:
            accessible_tags = user_tags + ["public"]
            stmt = (
                stmt.join(document_tags)
                .join(Tag, Tag.id == document_tags.c.tag_id)
                .where(Tag.name.in_(accessible_tags))
                .distinct()
            )

        return list(db.scalars(stmt).all())

    @staticmethod
    def _sort_by_recency(documents: list[Document]) -> list[Document]:
        """Sort documents by year tag descending (most recent first)."""

        def _year_key(doc: Document) -> str:
            for tag in doc.tags:
                if tag.name.startswith("year:"):
                    return tag.name
            return "year:0000"

        return sorted(documents, key=_year_key, reverse=True)

    def _get_field_groups(self, intent) -> list[str] | None:
        """Map intent labels to relevant ACORD 25 field groups.

        When multiple intents match (e.g. active_policy + gl), prefer the
        most specific one (non-None groups) over broad ones (None = all).

        Returns:
            list of group names, or None for all groups.
            Empty list if intent is not forms-eligible.
        """
        labels = getattr(intent, "intent_labels", None) or (
            [intent.intent_label] if intent.intent_label else []
        )

        # Collect groups from all matched labels, preferring specific over broad
        specific_groups: list[str] = []
        has_broad = False

        for label in labels:
            groups = INTENT_TO_GROUPS.get(label)
            if groups is None and label in INTENT_TO_GROUPS:
                has_broad = True  # Label maps to all groups
            elif groups:
                specific_groups.extend(g for g in groups if g not in specific_groups)

        if specific_groups:
            return specific_groups
        if has_broad:
            return None
        return []

    def _read_form_fields(
        self,
        table_name: str,
        ingest_key: str,
        field_groups: list[str] | None,
    ) -> dict[str, list[tuple[str, str]]]:
        """Read form fields from forms_data.db, grouped by ACORD section.

        Args:
            table_name: SQLite table in forms_data.db
            ingest_key: Row identifier (_ingest_key column)
            field_groups: Groups to include, or None for all

        Returns:
            Dict of group_name -> [(label, value), ...]
        """
        result: dict[str, list[tuple[str, str]]] = {}

        try:
            conn = sqlite3.connect(f"file:{self._forms_db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(
                    f'SELECT * FROM "{table_name}" WHERE _ingest_key = ?',
                    (ingest_key,),
                )
                row = cursor.fetchone()
                if row is None:
                    return result

                columns = row.keys()

                for col in columns:
                    if col.startswith("_"):
                        continue  # Skip metadata columns

                    value = row[col]
                    if not value or str(value).strip() == "":
                        continue

                    stripped = strip_column_prefix(col)

                    # Determine which group this field belongs to
                    matched_group = None
                    for group_name, patterns in ACORD_25_GROUPS.items():
                        if _matches_group(stripped, patterns):
                            matched_group = group_name
                            break

                    if matched_group is None:
                        matched_group = "Other"

                    # Filter by requested groups
                    if field_groups is not None and matched_group not in field_groups:
                        continue

                    label = field_name_to_label(stripped)
                    if matched_group not in result:
                        result[matched_group] = []
                    result[matched_group].append((label, str(value)))
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"Failed to read forms DB: {e}")

        return result

    def _format_as_search_result(
        self,
        document: Document,
        group_name: str,
        fields: list[tuple[str, str]],
        chunk_index: int,
        score: float = 0.95,
    ) -> SearchResult:
        """Format grouped form fields as a synthetic SearchResult."""
        header = "ACORD 25 - Certificate of Liability Insurance\n"
        header += f"Document: {document.original_filename}\n"

        # Extract named insured if present in fields
        for label, value in fields:
            if "Full Name" in label or "Named" in label:
                header += f"Insured: {value}\n"
                break

        text = header + f"\n{group_name}\n"
        for label, value in fields:
            text += f"{label}: {value}\n"

        doc_tags = [tag.name for tag in document.tags] if document.tags else []

        return SearchResult(
            chunk_id=f"{document.id}_form_{chunk_index}",
            document_id=document.id,
            document_name=document.original_filename,
            chunk_text=text,
            chunk_index=9000 + chunk_index,
            score=score,
            page_number=1,
            section=group_name,
            tags=doc_tags,
        )
