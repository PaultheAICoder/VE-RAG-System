"""Multi-layer RAG response caching service."""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0.
    Returns 0.0 if either vector is zero-length (no division error).
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class CacheEntry:
    """Cached RAG response."""

    # Key fields
    query_hash: str
    query_text: str
    query_embedding: list[float]

    # Response fields
    answer: str
    sources: list[dict]
    confidence_overall: int
    confidence_retrieval: float
    confidence_coverage: float
    confidence_llm: int
    generation_time_ms: float
    model_used: str

    # Metadata
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    document_ids: list[str]


@dataclass
class CacheStats:
    """Cache statistics."""

    enabled: bool
    total_entries: int
    memory_entries: int
    sqlite_entries: int
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class MemoryCache:
    """Fast in-memory LRU cache."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self._lock = threading.Lock()

    def get(self, query_hash: str) -> CacheEntry | None:
        """Get entry by hash, moving to end (MRU) if found."""
        with self._lock:
            if query_hash not in self.cache:
                return None
            self.cache.move_to_end(query_hash)
            entry = self.cache[query_hash]
            entry.last_accessed_at = datetime.utcnow()
            entry.access_count += 1
            return entry

    def get_semantic(self, embedding: list[float], threshold: float) -> CacheEntry | None:
        """Get entry by semantic similarity (Layer 2).

        Scans all cached embeddings and returns the best match
        if similarity > threshold.
        """
        with self._lock:
            best_entry: CacheEntry | None = None
            best_similarity = threshold  # Must exceed threshold

            for entry in self.cache.values():
                similarity = cosine_similarity(embedding, entry.query_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if best_entry is not None:
                # Move to MRU and update stats
                self.cache.move_to_end(best_entry.query_hash)
                best_entry.last_accessed_at = datetime.utcnow()
                best_entry.access_count += 1
                logger.debug(f"Semantic cache hit: similarity={best_similarity:.3f}")

            return best_entry

    def put(self, entry: CacheEntry) -> None:
        """Add entry to cache, evicting LRU if needed."""
        with self._lock:
            if entry.query_hash in self.cache:
                self.cache.move_to_end(entry.query_hash)
                self.cache[entry.query_hash] = entry
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Remove oldest
                self.cache[entry.query_hash] = entry

    def clear(self) -> int:
        """Clear all entries. Returns count removed."""
        with self._lock:
            count = len(self.cache)
            self.cache.clear()
            return count

    def __len__(self) -> int:
        return len(self.cache)


class CacheService:
    """Multi-layer RAG response cache with memory + SQLite backend."""

    def __init__(self, db: Session):
        self.db = db
        self.memory = MemoryCache()
        self._invalidation_paused = False
        self._pending_invalidation = False
        self._hit_count = 0
        self._miss_count = 0
        self._load_from_sqlite()

    def _load_from_sqlite(self) -> None:
        """Warm memory cache from SQLite on startup."""
        from ai_ready_rag.db.models import ResponseCache

        try:
            # Calculate cutoff for TTL
            cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)

            # Load up to max_entries most recently accessed (non-expired only)
            entries = (
                self.db.query(ResponseCache)
                .filter(ResponseCache.created_at >= cutoff)
                .order_by(ResponseCache.last_accessed_at.desc())
                .limit(self.max_entries)
                .all()
            )
            for row in entries:
                entry = self._row_to_entry(row)
                self.memory.put(entry)
            if entries:
                logger.info(f"Loaded {len(entries)} cache entries from SQLite")
        except Exception as e:
            logger.warning(f"Failed to load cache from SQLite: {e}")

    def _row_to_entry(self, row: Any) -> CacheEntry:
        """Convert SQLAlchemy row to CacheEntry."""
        return CacheEntry(
            query_hash=row.query_hash,
            query_text=row.query_text,
            query_embedding=json.loads(row.query_embedding),
            answer=row.answer,
            sources=json.loads(row.sources),
            confidence_overall=row.confidence_overall,
            confidence_retrieval=row.confidence_retrieval,
            confidence_coverage=row.confidence_coverage,
            confidence_llm=row.confidence_llm,
            generation_time_ms=row.generation_time_ms,
            model_used=row.model_used,
            created_at=row.created_at,
            last_accessed_at=row.last_accessed_at,
            access_count=row.access_count,
            document_ids=json.loads(row.document_ids),
        )

    @staticmethod
    def _hash_query(query: str) -> str:
        """Generate SHA256 hash of normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    # ----- Settings Properties -----

    @property
    def enabled(self) -> bool:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_enabled", True)

    @property
    def ttl_hours(self) -> int:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_ttl_hours", 24)

    @property
    def max_entries(self) -> int:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_max_entries", 1000)

    @property
    def min_confidence(self) -> int:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_min_confidence", 40)

    @property
    def semantic_threshold(self) -> float:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_semantic_threshold", 0.95)

    @property
    def auto_warm_enabled(self) -> bool:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_auto_warm_enabled", True)

    @property
    def auto_warm_count(self) -> int:
        from ai_ready_rag.services.settings_service import get_cache_setting

        return get_cache_setting("cache_auto_warm_count", 20)

    # ----- Core Operations -----

    async def get(
        self,
        query: str,
        user_tags: list[str],
        query_embedding: list[float] | None = None,
    ) -> CacheEntry | None:
        """
        Try cache lookup with Layer 1 (exact) and Layer 2 (semantic).

        Layer 1: Exact hash match (O(1))
        Layer 2: Semantic similarity match (O(n)) - only if query_embedding provided

        Verifies user has access to all cited documents.
        Returns None if cache miss or access denied.
        """
        if not self.enabled:
            return None

        query_hash = self._hash_query(query)

        # Layer 1: Try memory cache (exact hash match)
        entry = self.memory.get(query_hash)
        if entry:
            if await self.verify_access(entry, user_tags):
                self._log_access(query_hash, query, was_hit=True)
                self._hit_count += 1
                return entry
            # Access denied - treat as miss
            self._log_access(query_hash, query, was_hit=False)
            self._miss_count += 1
            return None

        # Layer 2: Semantic similarity lookup (if embedding provided)
        if query_embedding is not None:
            entry = self.memory.get_semantic(query_embedding, self.semantic_threshold)
            if entry and await self.verify_access(entry, user_tags):
                self._log_access(query_hash, query, was_hit=True)
                self._hit_count += 1
                return entry
            # Access denied or no match - continue to SQLite fallback

        # Try SQLite fallback (Layer 1 only - no semantic in SQLite for perf)
        entry = self._get_from_sqlite(query_hash)
        if entry:
            if await self.verify_access(entry, user_tags):
                self.memory.put(entry)  # Warm memory
                self._log_access(query_hash, query, was_hit=True)
                self._hit_count += 1
                return entry
            self._log_access(query_hash, query, was_hit=False)
            self._miss_count += 1
            return None

        self._log_access(query_hash, query, was_hit=False)
        self._miss_count += 1
        return None

    def _get_from_sqlite(self, query_hash: str) -> CacheEntry | None:
        """Get entry from SQLite by hash."""
        from ai_ready_rag.db.models import ResponseCache

        row = self.db.query(ResponseCache).filter(ResponseCache.query_hash == query_hash).first()
        if row:
            # Check TTL
            if datetime.utcnow() - row.created_at > timedelta(hours=self.ttl_hours):
                return None  # Expired
            return self._row_to_entry(row)
        return None

    async def verify_access(self, entry: CacheEntry, user_tags: list[str]) -> bool:
        """Verify user can access all documents cited in cached response."""
        if not entry.document_ids:
            return True  # No documents cited

        # Lazy import to avoid circular dependency
        from ai_ready_rag.db.models import Document, Tag, document_tags

        # Get user's tag IDs
        user_tag_objs = self.db.query(Tag).filter(Tag.name.in_(user_tags)).all()
        user_tag_ids = [t.id for t in user_tag_objs]

        if not user_tag_ids:
            return False  # User has no tags

        # Count accessible documents
        accessible = (
            self.db.query(Document)
            .join(document_tags)
            .filter(
                Document.id.in_(entry.document_ids),
                document_tags.c.tag_id.in_(user_tag_ids),
            )
            .distinct()
            .count()
        )

        return accessible == len(entry.document_ids)

    async def get_embedding(self, query: str) -> list[float] | None:
        """Get cached embedding for query (Layer 3)."""
        if not self.enabled:
            return None

        from ai_ready_rag.db.models import EmbeddingCache

        query_hash = self._hash_query(query)
        row = self.db.query(EmbeddingCache).filter(EmbeddingCache.query_hash == query_hash).first()
        if row:
            return json.loads(row.embedding)
        return None

    async def put(
        self,
        query: str,
        embedding: list[float],
        response: Any,  # RAGResponse - avoid import
    ) -> None:
        """Store response in cache (memory + SQLite)."""
        if not self.enabled:
            return

        # Skip low-confidence responses
        if response.confidence.overall < self.min_confidence:
            logger.debug(
                f"Skipping cache for low-confidence response: {response.confidence.overall}%"
            )
            return

        from ai_ready_rag.db.models import ResponseCache

        query_hash = self._hash_query(query)

        # Extract document IDs from citations
        doc_ids = list({c.document_id for c in response.citations if c.document_id})

        # Serialize sources
        sources = [
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "section": c.section,
                "excerpt": c.snippet,  # Citation uses 'snippet' not 'excerpt'
            }
            for c in response.citations
        ]

        entry = CacheEntry(
            query_hash=query_hash,
            query_text=query,
            query_embedding=embedding,
            answer=response.answer,
            sources=sources,
            confidence_overall=response.confidence.overall,
            confidence_retrieval=response.confidence.retrieval_score,
            confidence_coverage=response.confidence.coverage_score,
            confidence_llm=response.confidence.llm_score,
            generation_time_ms=response.generation_time_ms,
            model_used=response.model_used,
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            access_count=1,
            document_ids=doc_ids,
        )

        # Store in memory
        self.memory.put(entry)

        # Store in SQLite (upsert)
        existing = (
            self.db.query(ResponseCache).filter(ResponseCache.query_hash == query_hash).first()
        )
        if existing:
            existing.answer = entry.answer
            existing.sources = json.dumps(sources)
            existing.confidence_overall = entry.confidence_overall
            existing.confidence_retrieval = entry.confidence_retrieval
            existing.confidence_coverage = entry.confidence_coverage
            existing.confidence_llm = entry.confidence_llm
            existing.generation_time_ms = entry.generation_time_ms
            existing.model_used = entry.model_used
            existing.document_ids = json.dumps(doc_ids)
            existing.last_accessed_at = datetime.utcnow()
            existing.access_count += 1
        else:
            row = ResponseCache(
                query_hash=query_hash,
                query_text=query,
                query_embedding=json.dumps(embedding),
                answer=entry.answer,
                sources=json.dumps(sources),
                confidence_overall=entry.confidence_overall,
                confidence_retrieval=entry.confidence_retrieval,
                confidence_coverage=entry.confidence_coverage,
                confidence_llm=entry.confidence_llm,
                generation_time_ms=entry.generation_time_ms,
                model_used=entry.model_used,
                document_ids=json.dumps(doc_ids),
            )
            self.db.add(row)

        self.db.commit()
        logger.debug(f"Cached response for query: {query[:50]}...")

    async def put_embedding(self, query: str, embedding: list[float]) -> None:
        """Store query embedding (Layer 3)."""
        if not self.enabled:
            return

        from ai_ready_rag.db.models import EmbeddingCache

        query_hash = self._hash_query(query)

        existing = (
            self.db.query(EmbeddingCache).filter(EmbeddingCache.query_hash == query_hash).first()
        )
        if not existing:
            row = EmbeddingCache(
                query_hash=query_hash,
                query_text=query,
                embedding=json.dumps(embedding),
            )
            self.db.add(row)
            self.db.commit()

    def invalidate_all(self, reason: str) -> int:
        """Clear entire cache. Returns entries removed."""
        if self._invalidation_paused:
            self._pending_invalidation = True
            logger.debug(f"Cache invalidation deferred (reason: {reason})")
            return 0

        from ai_ready_rag.db.models import EmbeddingCache, ResponseCache

        memory_count = self.memory.clear()

        # Clear SQLite tables
        sqlite_count = self.db.query(ResponseCache).delete()
        self.db.query(EmbeddingCache).delete()
        self.db.commit()

        total = memory_count + sqlite_count
        logger.info(f"Cache invalidated: {total} entries (reason: {reason})")
        return total

    def evict_expired(self) -> int:
        """Remove entries older than TTL. Returns entries removed."""
        from ai_ready_rag.db.models import ResponseCache

        cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
        removed = self.db.query(ResponseCache).filter(ResponseCache.created_at < cutoff).delete()
        self.db.commit()
        logger.info(f"Evicted {removed} expired cache entries")
        return removed

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        from ai_ready_rag.db.models import ResponseCache

        sqlite_count = self.db.query(ResponseCache).count()

        return CacheStats(
            enabled=self.enabled,
            total_entries=len(self.memory),
            memory_entries=len(self.memory),
            sqlite_entries=sqlite_count,
            hit_count=self._hit_count,
            miss_count=self._miss_count,
        )

    def _log_access(self, query_hash: str, query_text: str, was_hit: bool) -> None:
        """Log cache access for frequency tracking."""
        from ai_ready_rag.db.models import CacheAccessLog

        log = CacheAccessLog(
            query_hash=query_hash,
            query_text=query_text,
            was_hit=was_hit,
        )
        self.db.add(log)
        self.db.commit()

    # ----- Cache Warming -----

    def get_top_queries(self, limit: int = 20) -> list[dict]:
        """Get most frequently accessed queries for cache warming.

        Returns queries sorted by access count (descending), which can be
        used for manual cache warming or auto-warming after invalidation.

        Args:
            limit: Maximum number of queries to return (default: 20)

        Returns:
            List of dicts with query_text, access_count, and last_accessed
        """
        from sqlalchemy import func

        from ai_ready_rag.db.models import CacheAccessLog

        results = (
            self.db.query(
                CacheAccessLog.query_text,
                func.count().label("access_count"),
                func.max(CacheAccessLog.accessed_at).label("last_accessed"),
            )
            .group_by(CacheAccessLog.query_hash)
            .order_by(func.count().desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "query_text": r.query_text,
                "access_count": r.access_count,
                "last_accessed": r.last_accessed,
            }
            for r in results
        ]

    # ----- Batch Context -----

    def pause_invalidation(self) -> None:
        """Pause cache invalidation (for batch uploads)."""
        self._invalidation_paused = True
        self._pending_invalidation = False

    def resume_invalidation(self) -> None:
        """Resume cache invalidation."""
        self._invalidation_paused = False


class UploadBatchContext:
    """Context manager for batch uploads with delayed cache invalidation."""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.pending_invalidation = False

    def __enter__(self) -> "UploadBatchContext":
        self.cache_service.pause_invalidation()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cache_service.resume_invalidation()
        if self.pending_invalidation:
            self.cache_service.invalidate_all(reason="batch_upload_complete")
