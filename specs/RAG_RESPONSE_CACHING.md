# RAG Response Caching Specification

| Field | Value |
|-------|-------|
| **Status** | IMPLEMENTED |
| **Version** | 1.0 |
| **Created** | 2026-01-31 |
| **Type** | Fullstack Enhancement |
| **Complexity** | COMPLEX |

---

## Summary

Add a multi-layer caching system to the RAG pipeline to dramatically reduce response times for repeated and similar queries. The cache uses a hybrid Memory + SQLite backend for fast reads with persistence across restarts, and includes admin-configurable settings for TTL, size limits, and semantic similarity thresholds.

## Goals

1. Reduce response time from ~36s to <100ms for cached queries
2. Support exact match, semantic similarity, and embedding caching
3. Provide admin controls for cache behavior via Settings UI
4. Maintain access control security with verification on cache hits
5. Persist cache across server restarts

## Scope

### In Scope

- Three-layer cache: exact query, semantic similarity, embedding
- Hybrid backend: in-memory LRU + SQLite persistence
- Global cache with access verification on hit
- Admin settings UI for all cache parameters
- Cache invalidation on document/settings changes
- Cache statistics display (hit rate, size, entries)
- Manual cache clear functionality

### Out of Scope

- Distributed caching (Redis/Memcached)
- Per-user or per-tag cache partitioning
- Partial cache invalidation (document-specific)
- Cache export/import

---

## Technical Specification

### 1. Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cache Layers                              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Exact Query Cache                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Key: hash(normalized_query)                                 ││
│  │ Hit: O(1) lookup, instant response                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                           ↓ Miss                                 │
│  Layer 2: Semantic Cache                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Key: query_embedding                                        ││
│  │ Hit: cosine_similarity >= threshold (default 0.95)          ││
│  └─────────────────────────────────────────────────────────────┘│
│                           ↓ Miss                                 │
│  Layer 3: Embedding Cache                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Key: hash(query_text)                                       ││
│  │ Value: embedding vector (skip Ollama embed call)            ││
│  └─────────────────────────────────────────────────────────────┘│
│                           ↓ Miss                                 │
│  Full RAG Pipeline (embed → search → generate → evaluate)        │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Cache Entry Structure

```python
@dataclass
class CacheEntry:
    """Cached RAG response."""

    # Key fields
    query_hash: str              # SHA256 of normalized query
    query_text: str              # Original query for semantic matching
    query_embedding: list[float] # For semantic similarity lookup

    # Response fields (full RAG response)
    answer: str
    sources: list[dict]          # Serialized citations
    confidence_overall: int
    confidence_retrieval: float
    confidence_coverage: float
    confidence_llm: int
    generation_time_ms: float    # Original generation time
    model_used: str

    # Metadata
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    document_ids: list[str]      # For access verification
```

### 3. Hybrid Storage Backend

#### In-Memory Layer (LRU)
```python
class MemoryCache:
    """Fast in-memory LRU cache."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.embeddings: dict[str, list[float]] = {}  # For semantic lookup

    def get(self, query_hash: str) -> CacheEntry | None: ...
    def get_semantic(self, embedding: list[float], threshold: float) -> CacheEntry | None: ...
    def put(self, entry: CacheEntry) -> None: ...
    def evict_lru(self) -> None: ...
    def clear(self) -> None: ...
```

#### SQLite Persistence Layer
```sql
-- New table: response_cache
CREATE TABLE response_cache (
    id TEXT PRIMARY KEY,
    query_hash TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding BLOB NOT NULL,      -- JSON-encoded float array
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,              -- JSON-encoded
    confidence_overall INTEGER NOT NULL,
    confidence_retrieval REAL NOT NULL,
    confidence_coverage REAL NOT NULL,
    confidence_llm INTEGER NOT NULL,
    generation_time_ms REAL NOT NULL,
    model_used TEXT NOT NULL,
    document_ids TEXT NOT NULL,         -- JSON-encoded list
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1
);

CREATE INDEX idx_cache_query_hash ON response_cache(query_hash);
CREATE INDEX idx_cache_created_at ON response_cache(created_at);
CREATE INDEX idx_cache_last_accessed ON response_cache(last_accessed_at);

-- New table: embedding_cache (Layer 3)
CREATE TABLE embedding_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    embedding BLOB NOT NULL,            -- JSON-encoded float array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. Cache Service

```python
class CacheService:
    """Multi-layer RAG response cache."""

    def __init__(self, db: Session):
        self.db = db
        self.memory = MemoryCache()
        self._load_from_sqlite()  # Warm memory on startup

    # Settings (from database)
    @property
    def enabled(self) -> bool: ...
    @property
    def ttl_hours(self) -> int: ...
    @property
    def max_entries(self) -> int: ...
    @property
    def semantic_threshold(self) -> float: ...

    # Core operations
    async def get(
        self,
        query: str,
        user_tags: list[str]
    ) -> CacheEntry | None:
        """
        Try cache layers in order:
        1. Exact match (query_hash)
        2. Semantic match (embedding similarity)

        Verifies user has access to all cited documents.
        Returns None if cache miss or access denied.
        """
        ...

    async def get_embedding(self, query: str) -> list[float] | None:
        """Get cached embedding for query (Layer 3)."""
        ...

    async def put(
        self,
        query: str,
        embedding: list[float],
        response: RAGResponse
    ) -> None:
        """Store response in cache (memory + SQLite)."""
        ...

    async def put_embedding(
        self,
        query: str,
        embedding: list[float]
    ) -> None:
        """Store query embedding (Layer 3)."""
        ...

    def invalidate_all(self, reason: str) -> int:
        """Clear entire cache. Returns entries removed."""
        ...

    def evict_expired(self) -> int:
        """Remove entries older than TTL. Returns entries removed."""
        ...

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...
```

### 5. Access Verification on Cache Hit

```python
async def verify_access(
    self,
    entry: CacheEntry,
    user_tags: list[str]
) -> bool:
    """
    Verify user can access all documents cited in cached response.

    Returns True if:
    - No documents cited (general knowledge answer)
    - All cited documents have tags matching user's tags

    Returns False if any cited document is inaccessible.
    """
    if not entry.document_ids:
        return True

    # Query documents table for tag intersection
    accessible = self.db.query(Document).filter(
        Document.id.in_(entry.document_ids),
        Document.tags.any(Tag.id.in_(user_tag_ids))
    ).count()

    return accessible == len(entry.document_ids)
```

### 6. Integration with RAG Service

```python
# In rag_service.py

async def query(self, request: RAGRequest) -> RAGResponse:
    """Modified query flow with caching."""

    # Check cache first (if enabled)
    if self.cache.enabled:
        cached = await self.cache.get(request.query, request.user_tags)
        if cached:
            return self._cache_entry_to_response(cached, from_cache=True)

    # Check embedding cache (Layer 3)
    query_embedding = await self.cache.get_embedding(request.query)
    if query_embedding is None:
        query_embedding = await self._embed_query(request.query)
        await self.cache.put_embedding(request.query, query_embedding)

    # ... rest of RAG pipeline ...

    # Store in cache before returning
    if self.cache.enabled:
        await self.cache.put(request.query, query_embedding, response)

    return response
```

### 7. Cache Invalidation Triggers

```python
# In document upload/delete handlers
async def upload_document(...):
    # ... upload logic ...
    cache_service.invalidate_all(reason="document_upload")

async def delete_document(...):
    # ... delete logic ...
    cache_service.invalidate_all(reason="document_delete")

# In settings update handlers
async def update_retrieval_settings(...):
    # ... update logic ...
    cache_service.invalidate_all(reason="settings_change")

async def update_llm_settings(...):
    # ... update logic ...
    cache_service.invalidate_all(reason="settings_change")

async def change_chat_model(...):
    # ... change logic ...
    cache_service.invalidate_all(reason="model_change")

async def change_embedding_model(...):
    # ... change logic ...
    cache_service.invalidate_all(reason="embedding_model_change")
```

### 8. Background TTL Cleanup

```python
# Periodic task (run every hour)
async def cleanup_expired_cache():
    """Remove cache entries older than TTL."""
    cache_service = CacheService(db)
    removed = cache_service.evict_expired()
    logger.info(f"Cache cleanup: removed {removed} expired entries")
```

### 9. Batch Upload Handling

Cache invalidation is delayed during bulk uploads to avoid clearing multiple times:

```python
class UploadBatchContext:
    """Context manager for batch uploads."""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.pending_invalidation = False

    def __enter__(self):
        self.cache_service.pause_invalidation()
        return self

    def __exit__(self, *args):
        self.cache_service.resume_invalidation()
        if self.pending_invalidation:
            self.cache_service.invalidate_all(reason="batch_upload_complete")

# Usage in upload endpoint
async def upload_documents(files: list[UploadFile]):
    with UploadBatchContext(cache_service) as batch:
        for file in files:
            await process_upload(file)
            batch.pending_invalidation = True
    # Cache cleared once after all uploads complete
```

### 10. Cache Warming

#### Manual Warming
Admin can upload a list of common queries to pre-cache:

```python
# API endpoint
POST /api/admin/cache/warm
Authorization: Bearer <admin_token>
Content-Type: application/json

{
    "queries": [
        "What is the refund policy?",
        "How do I reset my password?",
        "What are the shipping options?"
    ]
}

Response 200:
{
    "queued": 3,
    "message": "Cache warming started"
}
```

#### Auto Warming
System tracks query frequency and automatically re-caches popular queries after invalidation:

```python
class AutoWarmingService:
    """Automatically re-cache frequent queries after invalidation."""

    def __init__(self, cache_service: CacheService, db: Session):
        self.cache = cache_service
        self.db = db

    async def warm_top_queries(self, limit: int = 20) -> int:
        """Re-run and cache the most frequent queries."""
        top_queries = self.db.query(CacheAccessLog)\
            .group_by(CacheAccessLog.query_text)\
            .order_by(func.count().desc())\
            .limit(limit)\
            .all()

        warmed = 0
        for query_record in top_queries:
            response = await rag_service.query(query_record.query_text)
            await self.cache.put(query_record.query_text, response)
            warmed += 1

        return warmed
```

New table for tracking query frequency:

```sql
CREATE TABLE cache_access_log (
    id TEXT PRIMARY KEY,
    query_hash TEXT NOT NULL,
    query_text TEXT NOT NULL,
    was_hit BOOLEAN NOT NULL,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_access_log_query_hash ON cache_access_log(query_hash);
CREATE INDEX idx_access_log_accessed_at ON cache_access_log(accessed_at);
```

### 11. Low-Confidence Exclusion

Responses below the confidence threshold are not cached:

```python
async def put(
    self,
    query: str,
    embedding: list[float],
    response: RAGResponse
) -> None:
    """Store response in cache if confidence meets threshold."""

    # Skip caching low-confidence responses
    if response.confidence.overall < self.min_cache_confidence:
        logger.debug(f"Skipping cache for low-confidence response: {response.confidence.overall}%")
        return

    # ... store in cache ...
```

---

## Admin Settings

### New Settings Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cache_enabled` | bool | `true` | Master switch for caching |
| `cache_ttl_hours` | int | `24` | Time-to-live for cache entries |
| `cache_max_entries` | int | `1000` | Maximum entries before LRU eviction |
| `cache_semantic_threshold` | float | `0.95` | Cosine similarity threshold for semantic cache |
| `cache_min_confidence` | int | `40` | Minimum confidence to cache (responses below are not cached) |
| `cache_auto_warm_enabled` | bool | `true` | Automatically re-cache popular queries after invalidation |
| `cache_auto_warm_count` | int | `20` | Number of top queries to auto-warm |

### Settings Service Defaults

```python
# In settings_service.py

CACHE_SETTINGS_DEFAULTS = {
    "cache_enabled": True,
    "cache_ttl_hours": 24,
    "cache_max_entries": 1000,
    "cache_semantic_threshold": 0.95,
    "cache_min_confidence": 40,
    "cache_auto_warm_enabled": True,
    "cache_auto_warm_count": 20,
}
```

---

## API Endpoints

### Get Cache Settings

```
GET /api/admin/cache/settings
Authorization: Bearer <admin_token>

Response 200:
{
    "cache_enabled": true,
    "cache_ttl_hours": 24,
    "cache_max_entries": 1000,
    "cache_semantic_threshold": 0.95
}
```

### Update Cache Settings

```
PUT /api/admin/cache/settings
Authorization: Bearer <admin_token>
Content-Type: application/json

{
    "cache_enabled": true,
    "cache_ttl_hours": 12,
    "cache_max_entries": 2000,
    "cache_semantic_threshold": 0.92
}

Response 200:
{
    "cache_enabled": true,
    "cache_ttl_hours": 12,
    "cache_max_entries": 2000,
    "cache_semantic_threshold": 0.92
}
```

### Get Cache Statistics

```
GET /api/admin/cache/stats
Authorization: Bearer <admin_token>

Response 200:
{
    "enabled": true,
    "total_entries": 847,
    "memory_entries": 847,
    "sqlite_entries": 1203,
    "hit_count": 4521,
    "miss_count": 1203,
    "hit_rate": 0.79,
    "avg_response_time_cached_ms": 45,
    "avg_response_time_uncached_ms": 34200,
    "storage_size_bytes": 2457600,
    "oldest_entry": "2026-01-30T10:15:00Z",
    "newest_entry": "2026-01-31T18:45:00Z"
}
```

### Clear Cache

```
POST /api/admin/cache/clear
Authorization: Bearer <admin_token>

Response 200:
{
    "cleared_entries": 847,
    "message": "Cache cleared successfully"
}
```

### Get Hit Rate Trend

```
GET /api/admin/cache/trend?days=7
Authorization: Bearer <admin_token>

Response 200:
{
    "trend": [
        {"date": "2026-01-25", "hit_rate": 72.5, "hits": 145, "misses": 55},
        {"date": "2026-01-26", "hit_rate": 75.2, "hits": 188, "misses": 62},
        {"date": "2026-01-27", "hit_rate": 78.1, "hits": 203, "misses": 57},
        ...
    ]
}
```

### Get Top Cached Queries

```
GET /api/admin/cache/top-queries?limit=10
Authorization: Bearer <admin_token>

Response 200:
{
    "queries": [
        {"query_text": "What is the refund policy?", "hit_count": 342, "last_hit": "2026-01-31T18:30:00Z"},
        {"query_text": "How do I reset my password?", "hit_count": 287, "last_hit": "2026-01-31T18:25:00Z"},
        ...
    ]
}
```

### Manual Cache Warming

```
POST /api/admin/cache/warm
Authorization: Bearer <admin_token>
Content-Type: application/json

{
    "queries": [
        "What is the refund policy?",
        "How do I contact support?",
        "What are the shipping options?"
    ]
}

Response 202:
{
    "queued": 3,
    "message": "Cache warming started in background"
}
```

---

## Frontend UI

### Cache Settings Card (in SettingsView.tsx)

```tsx
<Card>
  <h2>Cache Settings</h2>
  <p>Configure response caching to improve query performance.</p>

  <Checkbox
    label="Enable Response Cache"
    checked={cacheSettings.cache_enabled}
    onChange={...}
  />

  <Slider
    label="Cache TTL (Hours)"
    description="How long cached responses remain valid"
    value={cacheSettings.cache_ttl_hours}
    min={1}
    max={168}  // 7 days
    step={1}
  />

  <Slider
    label="Max Cache Entries"
    description="Maximum number of cached responses"
    value={cacheSettings.cache_max_entries}
    min={100}
    max={10000}
    step={100}
  />

  <Slider
    label="Semantic Similarity Threshold"
    description="How similar queries must be for semantic cache hit (higher = stricter)"
    value={cacheSettings.cache_semantic_threshold}
    min={0.85}
    max={0.99}
    step={0.01}
    valueFormatter={(v) => v.toFixed(2)}
  />
</Card>
```

### Cache Statistics Card (in SettingsView.tsx)

```tsx
<Card>
  <div className="flex justify-between items-center">
    <h2>Cache Statistics</h2>
    <Button variant="danger" icon={Trash2} onClick={handleClearCache}>
      Clear Cache
    </Button>
  </div>

  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
    <StatBox label="Hit Rate" value={`${(stats.hit_rate * 100).toFixed(1)}%`} />
    <StatBox label="Entries" value={stats.total_entries.toLocaleString()} />
    <StatBox label="Avg Cached" value={`${stats.avg_response_time_cached_ms}ms`} />
    <StatBox label="Avg Uncached" value={`${(stats.avg_response_time_uncached_ms / 1000).toFixed(1)}s`} />
  </div>

  <div className="mt-4 text-sm text-gray-500">
    Storage: {formatBytes(stats.storage_size_bytes)} |
    Oldest: {formatDate(stats.oldest_entry)} |
    Newest: {formatDate(stats.newest_entry)}
  </div>
</Card>
```

### Cache Impact Visualizations

#### Response Time Comparison (Bar Chart)

```tsx
<Card>
  <h2>Response Time Comparison</h2>
  <div className="mt-4">
    {/* Horizontal bar chart showing cached vs uncached times */}
    <div className="space-y-3">
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span>Cached</span>
          <span>{stats.avg_response_time_cached_ms}ms</span>
        </div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded">
          <div
            className="h-4 bg-green-500 rounded"
            style={{ width: `${(stats.avg_response_time_cached_ms / stats.avg_response_time_uncached_ms) * 100}%` }}
          />
        </div>
      </div>
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span>Uncached</span>
          <span>{(stats.avg_response_time_uncached_ms / 1000).toFixed(1)}s</span>
        </div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded">
          <div className="h-4 bg-amber-500 rounded w-full" />
        </div>
      </div>
    </div>
    <p className="text-sm text-gray-500 mt-3">
      Cache provides ~{Math.round(stats.avg_response_time_uncached_ms / stats.avg_response_time_cached_ms)}x speedup
    </p>
  </div>
</Card>
```

#### Hit Rate Trend (Line Chart)

```tsx
<Card>
  <h2>Hit Rate Trend (Last 7 Days)</h2>
  <div className="mt-4 h-48">
    {/* Line chart using recharts or similar */}
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={hitRateTrend}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
        <Tooltip formatter={(v) => `${v}%`} />
        <Line type="monotone" dataKey="hit_rate" stroke="#10b981" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  </div>
</Card>
```

#### Top Cached Queries (Table)

```tsx
<Card>
  <h2>Top Cached Queries</h2>
  <div className="mt-4">
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-gray-500 border-b">
          <th className="pb-2">Query</th>
          <th className="pb-2 text-right">Hits</th>
          <th className="pb-2 text-right">Last Hit</th>
        </tr>
      </thead>
      <tbody>
        {topQueries.map((q, i) => (
          <tr key={i} className="border-b border-gray-100 dark:border-gray-800">
            <td className="py-2 truncate max-w-xs">{q.query_text}</td>
            <td className="py-2 text-right">{q.hit_count}</td>
            <td className="py-2 text-right text-gray-500">{formatRelativeTime(q.last_hit)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
</Card>
```

### Cache Warming Card

```tsx
<Card>
  <h2>Cache Warming</h2>
  <p className="text-sm text-gray-500 mb-4">
    Pre-populate cache with common queries for faster responses.
  </p>

  <div className="space-y-4">
    <Checkbox
      label="Auto-warm after cache clear"
      description="Automatically re-cache popular queries when cache is invalidated"
      checked={cacheSettings.cache_auto_warm_enabled}
      onChange={...}
    />

    <Slider
      label="Auto-warm Query Count"
      description="Number of top queries to automatically re-cache"
      value={cacheSettings.cache_auto_warm_count}
      min={5}
      max={50}
      step={5}
      disabled={!cacheSettings.cache_auto_warm_enabled}
    />

    <div className="pt-4 border-t">
      <h3 className="font-medium mb-2">Manual Warming</h3>
      <textarea
        className="w-full h-24 p-2 border rounded text-sm"
        placeholder="Enter queries to warm (one per line)..."
        value={warmQueries}
        onChange={(e) => setWarmQueries(e.target.value)}
      />
      <Button
        className="mt-2"
        icon={Flame}
        onClick={handleWarmCache}
        disabled={!warmQueries.trim()}
      >
        Warm Cache
      </Button>
    </div>
  </div>
</Card>
```

---

## TypeScript Types

```typescript
// In frontend/src/types/index.ts

export interface CacheSettings {
  cache_enabled: boolean;
  cache_ttl_hours: number;
  cache_max_entries: number;
  cache_semantic_threshold: number;
  cache_min_confidence: number;
  cache_auto_warm_enabled: boolean;
  cache_auto_warm_count: number;
}

export interface CacheStats {
  enabled: boolean;
  total_entries: number;
  memory_entries: number;
  sqlite_entries: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
  avg_response_time_cached_ms: number;
  avg_response_time_uncached_ms: number;
  storage_size_bytes: number;
  oldest_entry: string | null;
  newest_entry: string | null;
}

export interface HitRateTrendPoint {
  date: string;       // ISO date
  hit_rate: number;   // 0-100
  hits: number;
  misses: number;
}

export interface TopCachedQuery {
  query_text: string;
  hit_count: number;
  last_hit: string;   // ISO timestamp
}

export interface CacheImpactData {
  hit_rate_trend: HitRateTrendPoint[];
  top_queries: TopCachedQuery[];
}
```

---

## Risk Flags

### MULTI_MODEL Risk
- Cache entry references multiple documents for access verification
- Ensure atomic operations when storing/retrieving

### INVALIDATION Risk
- Full cache clear on document changes may frustrate users during bulk uploads
- **Mitigated**: Batch uploads delay invalidation until complete

### SEMANTIC_SEARCH Risk
- Semantic cache lookup requires comparing against all cached embeddings
- For large caches (>10k), consider using a vector index (FAISS/Annoy) for memory cache

### STALE_DATA Risk
- Cached responses may cite documents that were updated (content changed, not deleted)
- Mitigation: TTL ensures eventual refresh; admin can clear manually

---

## Implementation Phases

### Phase 1: Core Cache Service
- [ ] Create `response_cache`, `embedding_cache`, and `cache_access_log` SQLite tables
- [ ] Implement `CacheService` with memory + SQLite hybrid
- [ ] Add exact query cache (Layer 1)
- [ ] Add embedding cache (Layer 3)
- [ ] Integrate with RAG service
- [ ] Add low-confidence exclusion logic
- [ ] Add batch upload handling (delayed invalidation)

### Phase 2: Semantic Cache
- [ ] Add semantic similarity lookup (Layer 2)
- [ ] Store and compare embeddings for cached queries
- [ ] Add similarity threshold setting

### Phase 3: Cache Warming
- [ ] Implement manual warming endpoint (POST /api/admin/cache/warm)
- [ ] Implement `AutoWarmingService` for automatic re-caching
- [ ] Track query frequency in `cache_access_log` table
- [ ] Add auto-warm settings (enabled, count)

### Phase 4: Admin API
- [ ] Add cache settings endpoints (GET/PUT)
- [ ] Add cache statistics endpoint
- [ ] Add clear cache endpoint
- [ ] Add hit rate trend endpoint
- [ ] Add top queries endpoint
- [ ] Add invalidation triggers to document/settings handlers

### Phase 5: Frontend UI
- [ ] Add TypeScript types for cache settings and stats
- [ ] Add cache API functions
- [ ] Create Cache Settings card in SettingsView
- [ ] Create Cache Statistics card with clear button
- [ ] Create Response Time Comparison bar chart
- [ ] Create Hit Rate Trend line chart (use recharts)
- [ ] Create Top Cached Queries table
- [ ] Create Cache Warming card with manual input

### Phase 6: Testing & Polish
- [ ] Unit tests for CacheService
- [ ] Integration tests for cache hit/miss scenarios
- [ ] Access verification tests
- [ ] Cache warming tests
- [ ] Performance benchmarks

---

## Acceptance Criteria

### Core Caching
- [ ] Exact query cache returns response in <100ms
- [ ] Semantic cache matches similar queries with configurable threshold
- [ ] Embedding cache eliminates redundant Ollama embed calls
- [ ] Cache persists across server restarts (SQLite)
- [ ] Low-confidence responses (below threshold) are not cached

### Admin Controls
- [ ] Admin can enable/disable cache via Settings UI
- [ ] Admin can configure TTL, max entries, semantic threshold, and min confidence
- [ ] Admin can view cache statistics (hit rate, size, timings)
- [ ] Admin can manually clear cache

### Cache Warming
- [ ] Admin can upload a list of queries for manual cache warming
- [ ] System automatically re-caches top N queries after invalidation (configurable)
- [ ] Query frequency is tracked for auto-warming

### Visualizations
- [ ] Response time comparison bar chart shows cached vs uncached times
- [ ] Hit rate trend line chart shows daily hit rate over last 7 days
- [ ] Top cached queries table shows most frequently hit queries

### Invalidation
- [ ] Cache automatically invalidates on document upload/delete
- [ ] Cache automatically invalidates on settings/model changes
- [ ] Bulk uploads delay invalidation until batch completes
- [ ] LRU eviction prevents unbounded memory growth
- [ ] TTL expiry removes stale entries

### Security
- [ ] Cache respects access control (verifies user can access cited docs on hit)

---

## Resolved Questions

| Question | Decision |
|----------|----------|
| **Bulk upload handling** | Delay invalidation until batch completes (avoid multiple clears) |
| **Cache warming** | Support both manual (admin uploads query list) and auto (re-cache popular queries after invalidation) |
| **Impact visualization** | Response time comparison bar chart, hit rate trend line chart, top queries table |
| **Low-confidence exclusion** | Exclude responses below configurable threshold (`cache_min_confidence` setting) |

---

## Effort Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Core Cache | 2-3 days |
| Phase 2: Semantic Cache | 1-2 days |
| Phase 3: Cache Warming | 1-2 days |
| Phase 4: Admin API | 1 day |
| Phase 5: Frontend UI | 2-3 days |
| Phase 6: Testing | 1-2 days |
| **Total** | **8-13 days** |

---

## References

- Settings Service: `ai_ready_rag/services/settings_service.py`
- RAG Service: `ai_ready_rag/services/rag_service.py`
- Settings View: `frontend/src/views/SettingsView.tsx`
- Admin API: `ai_ready_rag/api/admin.py`
