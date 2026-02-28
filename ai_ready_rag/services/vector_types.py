"""Zero-dependency type definitions for vector search results.

Extracted from vector_service.py so that rag_service.py and other modules
can import SearchResult without pulling in qdrant-client at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Single search result with metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    chunk_text: str
    chunk_index: int
    score: float
    page_number: int | None
    section: str | None
    tags: list[str] | None = field(default=None)
