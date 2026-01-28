# RAG Service Specification

**Version:** 2.0
**Date:** January 28, 2026
**Status:** Draft (Revised per engineering review)
**Depends On:** VectorService, Ollama (llama3.2, qwen3:8b)

---

## Overview

The RAG Service orchestrates retrieval-augmented generation by combining context from VectorService with LLM responses from Ollama. It handles prompt construction, response generation, confidence scoring, citation formatting, and routing decisions.

**Key Principles:**
- **Context-grounded only**: All substantive answers MUST be grounded in retrieved context
- Uses VectorService for context retrieval (access control already enforced)
- Deterministic citation mapping via source IDs (no LLM-dependent parsing)
- Hybrid confidence scoring (retrieval signals + LLM evaluation)
- Routes to human when confidence is low (<60%)
- Explicit token budgeting and truncation strategies
- All operations are async
- Stateless service (session state managed by Chat API)

---

## File Location

```
ai_ready_rag/
├── services/
│   ├── __init__.py
│   ├── vector_service.py    # Dependency
│   └── rag_service.py       # This specification
```

---

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| langchain-ollama | LLM client | >=0.3.0 |
| tiktoken | Token counting | >=0.5.0 |
| VectorService | Context retrieval | Internal |

---

## Configuration

Environment variables (loaded via config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| CHAT_MODEL | llama3.2 | Default LLM model |
| RAG_TEMPERATURE | 0.1 | LLM temperature (lower = more deterministic) |
| RAG_TIMEOUT_SECONDS | 30 | LLM response timeout |

### Token Budget Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| RAG_MAX_CONTEXT_TOKENS | 3000 | Max tokens for retrieved context |
| RAG_MAX_HISTORY_TOKENS | 1000 | Max tokens for chat history |
| RAG_MAX_RESPONSE_TOKENS | 1024 | Max tokens for LLM response |
| RAG_SYSTEM_PROMPT_TOKENS | 500 | Reserved for system prompt |

**Total budget per model:**
| Model | Context Window | Available for RAG |
|-------|----------------|-------------------|
| llama3.2 | 8192 | 8192 - 500 (system) = 7692 |
| qwen3:8b | 32768 | 32768 - 500 = 32268 |
| deepseek-r1:32b | 65536 | 65536 - 500 = 65036 |

### Retrieval Quality Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| RAG_MIN_SIMILARITY_SCORE | 0.3 | Minimum cosine similarity to include chunk |
| RAG_MAX_CHUNKS_PER_DOC | 3 | Max chunks from single document |
| RAG_TOTAL_CONTEXT_CHUNKS | 5 | Total chunks to include |
| RAG_CHUNK_OVERLAP_THRESHOLD | 0.9 | Similarity threshold for dedup |
| RAG_CONFIDENCE_THRESHOLD | 60 | Min confidence for CITE action (%) |

---

## Data Models

### ChatMessage

```python
@dataclass
class ChatMessage:
    """Single message in chat history."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime | None = None
```

### RAGRequest

```python
@dataclass
class RAGRequest:
    """Input for RAG generation."""
    query: str                          # User's question
    user_tags: list[str]                # User's access tags (for filtering)
    tenant_id: str = "default"          # Multi-tenant isolation (passed to VectorService)
    chat_history: list[ChatMessage] | None = None  # Previous messages
    model: str | None = None            # Override default model (must be in allowlist)
    max_context_chunks: int = 5         # Limit context size
```

### RAGResponse

```python
@dataclass
class RAGResponse:
    """Output from RAG generation."""
    answer: str                         # Generated response text
    confidence: ConfidenceScore         # Detailed confidence breakdown
    citations: list[Citation]           # Source references (deterministic)
    action: Literal["CITE", "ROUTE"]    # Response action
    route_to: RouteTarget | None        # Routing details if action=ROUTE
    model_used: str                     # Actual model used
    context_chunks_used: int            # Number of chunks in context
    context_tokens_used: int            # Actual tokens used for context
    generation_time_ms: float           # Response time
    grounded: bool                      # True if answer based on retrieved context
```

### ConfidenceScore

```python
@dataclass
class ConfidenceScore:
    """Hybrid confidence scoring with breakdown."""
    overall: int                        # 0-100 final score
    retrieval_score: float              # Average similarity of top chunks (0-1)
    coverage_score: float               # % of answer supported by context (0-1)
    llm_score: int                      # LLM self-assessment (0-100)

    # Weights for final calculation
    # overall = (retrieval_score * 30) + (coverage_score * 40) + (llm_score * 0.3)
```

### Citation

```python
@dataclass
class Citation:
    """Deterministic source citation."""
    source_id: str                      # Format: "{document_id}:{chunk_index}"
    document_id: str
    document_name: str
    chunk_index: int
    page_number: int | None             # None displayed as "N/A"
    section: str | None                 # None displayed as "N/A"
    relevance_score: float              # 0.0-1.0 from vector search
    snippet: str                        # Truncated to 200 chars, ellipsis if cut
    snippet_full: str                   # Full chunk text for hover/expand
```

### RouteTarget

```python
@dataclass
class RouteTarget:
    """Routing destination when confidence is low."""
    tag: str                            # Primary tag from context
    owner_user_id: str | None           # From tags.owner_user_id, may be None
    owner_email: str | None             # From users.email if owner exists
    reason: str                         # Why routing (e.g., "Low confidence: 45%")
    fallback: bool                      # True if no owner found, route to admin
```

---

## Prompt Templates

### RAG System Prompt (with context)

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

STRICT RULES:
1. ONLY use information from the provided CONTEXT sections below
2. If the context doesn't contain enough information, respond with: "I don't have enough information in the available documents to answer this question."
3. NEVER use external knowledge or make assumptions beyond the context
4. Cite sources using the exact SourceId provided (e.g., [SourceId: abc123:0])
5. Be concise but thorough
6. If partially answerable, answer what you can and note what's missing

CONTEXT (each section has a SourceId for citation):
{context}

PREVIOUS CONVERSATION:
{chat_history}
"""
```

### Context Chunk Format

Each chunk is formatted with a deterministic SourceId:

```
[SourceId: {document_id}:{chunk_index}]
[Document: {document_name}]
[Page: {page_number or 'N/A'}] [Section: {section or 'N/A'}]
---
{chunk_text}
---
```

### Confidence Evaluation Prompt

```python
CONFIDENCE_PROMPT = """Evaluate how well this answer is supported by the provided context.

CONTEXT SUMMARY:
{context_summary}

QUESTION: {query}

ANSWER: {answer}

Rate the support level from 0-100:
- 90-100: Answer is fully and directly supported by context
- 70-89: Answer is mostly supported, minor inferences made
- 50-69: Answer is partially supported, some gaps filled
- 30-49: Answer has weak support, significant assumptions
- 0-29: Answer is not supported by context

Respond with ONLY a number between 0 and 100."""
```

### Query Classification Prompt

**Note:** The DIRECT path is removed. All queries go through retrieval. For truly non-document queries (greetings, clarifications), the retrieval will return low-relevance results and the response will indicate insufficient context.

```python
QUERY_CLASSIFIER_PROMPT = """Classify this query for retrieval optimization.

QUERY: {query}

Categories:
- FACTUAL: Seeking specific facts from documents
- PROCEDURAL: Asking how to do something
- CLARIFICATION: Follow-up on previous answer (use chat history)
- GREETING: Social interaction (still attempt retrieval, expect low relevance)

Respond with exactly one category."""
```

---

## Citation Mapping (Deterministic)

### Problem (from review)
The original spec relied on LLM-generated inline citations like `[Source: filename, page X]` which requires parsing and is brittle/hallucination-prone.

### Solution: Source ID Mapping

1. **Context includes deterministic SourceIds:**
   ```
   [SourceId: doc-abc123:2]
   [Document: hr-handbook.pdf]
   ...
   ```

2. **LLM cites using SourceId:**
   ```
   According to the vacation policy [SourceId: doc-abc123:2], employees receive...
   ```

3. **Service extracts citations via regex:**
   ```python
   CITATION_PATTERN = r'\[SourceId:\s*([a-zA-Z0-9-]+:\d+)\]'
   ```

4. **Mapping is deterministic:**
   - SourceId `doc-abc123:2` maps to `document_id="doc-abc123"`, `chunk_index=2`
   - Metadata looked up from the search results (already fetched)
   - Unknown SourceIds logged as warnings, not included in citations

### Citation Snippet Rules

| Field | Rule |
|-------|------|
| snippet | First 200 characters of chunk_text, add "..." if truncated |
| snippet_full | Full chunk_text (for UI expand/hover) |
| page_number | Display as "N/A" if None |
| section | Display as "N/A" if None |

---

## Confidence Scoring (Hybrid)

### Problem (from review)
LLM self-assessment alone is unreliable and tends to be over-confident.

### Solution: Hybrid Scoring

```python
def calculate_confidence(
    retrieval_results: list[SearchResult],
    answer: str,
    context: str,
    llm_score: int,
) -> ConfidenceScore:
    """
    Combine multiple signals for robust confidence.

    Weights:
    - Retrieval quality (30%): Are the retrieved chunks relevant?
    - Coverage (40%): Does the answer use the context?
    - LLM assessment (30%): LLM's self-evaluation
    """

    # 1. Retrieval score: average similarity of used chunks
    retrieval_score = sum(r.score for r in retrieval_results) / len(retrieval_results)

    # 2. Coverage score: keyword/phrase overlap between answer and context
    coverage_score = calculate_coverage(answer, context)

    # 3. LLM score: from CONFIDENCE_PROMPT (0-100, normalized to 0-1)
    llm_normalized = llm_score / 100

    # Weighted combination
    overall = int(
        (retrieval_score * 30) +
        (coverage_score * 40) +
        (llm_normalized * 30)
    )

    return ConfidenceScore(
        overall=min(100, max(0, overall)),
        retrieval_score=retrieval_score,
        coverage_score=coverage_score,
        llm_score=llm_score,
    )
```

### Coverage Calculation

```python
def calculate_coverage(answer: str, context: str) -> float:
    """
    Estimate how much of the answer is grounded in context.

    Method: Extract significant phrases from answer, check presence in context.
    """
    # Extract noun phrases and key terms from answer
    answer_phrases = extract_key_phrases(answer)

    # Check how many appear in context
    matches = sum(1 for phrase in answer_phrases if phrase.lower() in context.lower())

    if not answer_phrases:
        return 0.0

    return matches / len(answer_phrases)
```

### Confidence Thresholds

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 80-100 | High confidence, well-grounded | CITE |
| 60-79 | Moderate confidence | CITE |
| 40-59 | Low confidence | ROUTE |
| 0-39 | Very low confidence | ROUTE |

---

## Retrieval Quality Controls

### Problem (from review)
Original spec only had top-k, no quality filters.

### Solution: Multi-stage Filtering

```python
async def get_quality_context(
    self,
    query: str,
    user_tags: list[str],
    tenant_id: str,
    max_chunks: int = 5,
) -> list[SearchResult]:
    """
    Retrieve context with quality controls.

    Pipeline:
    1. Retrieve 3x candidates from VectorService
    2. Filter by minimum similarity score
    3. Deduplicate near-identical chunks
    4. Limit chunks per document
    5. Return top-k by score
    """
    # 1. Over-fetch candidates
    candidates = await self.vector_service.search(
        query=query,
        user_tags=user_tags,
        tenant_id=tenant_id,
        limit=max_chunks * 3,
        score_threshold=0.0,  # Filter later for more control
    )

    # 2. Filter by minimum similarity
    filtered = [c for c in candidates if c.score >= self.min_similarity_score]

    # 3. Deduplicate (remove chunks with >90% text similarity)
    deduped = self._deduplicate_chunks(filtered)

    # 4. Limit per document
    limited = self._limit_per_document(deduped, max_per_doc=3)

    # 5. Return top-k
    return limited[:max_chunks]
```

### Deduplication

```python
def _deduplicate_chunks(self, chunks: list[SearchResult]) -> list[SearchResult]:
    """Remove near-duplicate chunks (>90% similar)."""
    result = []
    for chunk in chunks:
        is_duplicate = False
        for existing in result:
            similarity = self._text_similarity(chunk.chunk_text, existing.chunk_text)
            if similarity > self.chunk_overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            result.append(chunk)
    return result
```

---

## Token Budget Management

### Problem (from review)
No token counting or truncation strategy.

### Solution: Explicit Token Budgeting

```python
class TokenBudget:
    """Manages token allocation for RAG prompt."""

    def __init__(self, model: str):
        self.model = model
        self.limits = MODEL_TOKEN_LIMITS.get(model, DEFAULT_LIMITS)

    def allocate(
        self,
        system_prompt: str,
        chat_history: list[ChatMessage],
        context_chunks: list[SearchResult],
    ) -> tuple[str, str, list[SearchResult]]:
        """
        Allocate tokens and truncate as needed.

        Priority (highest to lowest):
        1. System prompt (fixed, never truncate)
        2. Most recent chat history (truncate oldest first)
        3. Highest-relevance context chunks (drop lowest-scored first)

        Returns:
            (system_prompt, truncated_history, filtered_chunks)
        """
        available = self.limits["context_window"]

        # 1. Reserve for system prompt
        system_tokens = count_tokens(system_prompt)
        available -= system_tokens

        # 2. Reserve for response
        available -= self.limits["max_response"]

        # 3. Allocate history (truncate oldest)
        history_str, history_tokens = self._truncate_history(
            chat_history,
            max_tokens=min(available // 3, self.limits["max_history"])
        )
        available -= history_tokens

        # 4. Allocate context (drop lowest-scored chunks)
        final_chunks = self._fit_context(
            context_chunks,
            max_tokens=available
        )

        return system_prompt, history_str, final_chunks
```

### History Truncation

```python
def _truncate_history(
    self,
    history: list[ChatMessage],
    max_tokens: int,
) -> tuple[str, int]:
    """
    Truncate chat history, keeping most recent messages.

    Strategy: Remove oldest messages until under budget.
    Always keep at least the last user message.
    """
    if not history:
        return "", 0

    # Start from most recent, work backwards
    included = []
    total_tokens = 0

    for msg in reversed(history):
        msg_tokens = count_tokens(f"{msg.role}: {msg.content}")
        if total_tokens + msg_tokens <= max_tokens:
            included.insert(0, msg)
            total_tokens += msg_tokens
        else:
            break

    # Format for prompt
    formatted = "\n".join(f"{m.role}: {m.content}" for m in included)
    return formatted, total_tokens
```

---

## Tag Owner Lookup

### Problem (from review)
Tag owner lookup source not specified.

### Solution: Database Lookup with Fallback

```python
async def get_route_target(
    self,
    context_results: list[SearchResult],
    db: Session,
) -> RouteTarget:
    """
    Determine routing target based on context tags.

    Lookup chain:
    1. Find most common tag in retrieved documents
    2. Look up tag in `tags` table
    3. If tag.owner_user_id exists, get user email
    4. If no owner, fallback to system admin
    """
    # 1. Find primary tag from context
    tag_counts: dict[str, int] = {}
    for result in context_results:
        for tag in result.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        return self._admin_fallback("No tags in context")

    primary_tag = max(tag_counts, key=tag_counts.get)

    # 2. Look up tag owner
    tag_record = db.query(Tag).filter(Tag.name == primary_tag).first()

    if not tag_record or not tag_record.owner_user_id:
        return self._admin_fallback(f"No owner for tag '{primary_tag}'")

    # 3. Get owner details
    owner = db.query(User).filter(User.id == tag_record.owner_user_id).first()

    if not owner:
        return self._admin_fallback(f"Owner user not found for tag '{primary_tag}'")

    return RouteTarget(
        tag=primary_tag,
        owner_user_id=owner.id,
        owner_email=owner.email,
        reason=f"Low confidence, routing to {primary_tag} owner",
        fallback=False,
    )

def _admin_fallback(self, reason: str) -> RouteTarget:
    """Return admin as fallback route target."""
    return RouteTarget(
        tag="system",
        owner_user_id=None,
        owner_email="admin@company.com",  # From config
        reason=f"{reason} - routing to admin",
        fallback=True,
    )
```

---

## Model Validation

### Problem (from review)
Model availability check not defined.

### Solution: Allowlist with Runtime Check

```python
# Configuration
MODEL_ALLOWLIST = {
    "llama3.2": {"context_window": 8192, "max_response": 1024},
    "qwen3:8b": {"context_window": 32768, "max_response": 2048},
    "deepseek-r1:32b": {"context_window": 65536, "max_response": 4096},
}

async def validate_model(self, model: str) -> str:
    """
    Validate model is allowed and available.

    Args:
        model: Requested model name

    Returns:
        Validated model name

    Raises:
        ModelNotAllowedError: Model not in allowlist
        ModelUnavailableError: Model not pulled in Ollama
    """
    # 1. Check allowlist
    if model not in MODEL_ALLOWLIST:
        raise ModelNotAllowedError(
            f"Model '{model}' not in allowlist. "
            f"Available: {list(MODEL_ALLOWLIST.keys())}"
        )

    # 2. Check Ollama availability
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ollama_url}/api/tags",
                timeout=5.0
            )
            available_models = [m["name"] for m in response.json().get("models", [])]

            # Check for exact match or with :latest suffix
            if model not in available_models and f"{model}:latest" not in available_models:
                raise ModelUnavailableError(
                    f"Model '{model}' not available in Ollama. "
                    f"Run: ollama pull {model}"
                )
    except httpx.RequestError as e:
        raise LLMConnectionError(f"Cannot connect to Ollama: {e}")

    return model
```

---

## Error Handling (Service Layer)

### Problem (from review)
Mixed HTTP status codes with service exceptions.

### Solution: Pure Exception Hierarchy

```python
class RAGServiceError(Exception):
    """Base exception for RAG service errors."""
    pass

class LLMConnectionError(RAGServiceError):
    """Cannot connect to Ollama."""
    pass

class LLMTimeoutError(RAGServiceError):
    """LLM response timed out."""
    pass

class ModelNotAllowedError(RAGServiceError):
    """Requested model not in allowlist."""
    pass

class ModelUnavailableError(RAGServiceError):
    """Model not available in Ollama."""
    pass

class ContextError(RAGServiceError):
    """Failed to retrieve or format context."""
    pass

class TokenBudgetExceededError(RAGServiceError):
    """Query + history exceeds available token budget."""
    pass
```

**API layer is responsible for mapping exceptions to HTTP responses.**

---

## Service Interface

### RAGService Class

```python
class RAGService:
    """Orchestrates RAG pipeline: retrieve → prompt → generate → evaluate."""

    def __init__(
        self,
        vector_service: VectorService,
        ollama_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
        temperature: float = 0.1,
        confidence_threshold: int = 60,
        timeout_seconds: int = 30,
        min_similarity_score: float = 0.3,
        max_chunks_per_doc: int = 3,
        chunk_overlap_threshold: float = 0.9,
    ):
        """Initialize RAG service with dependencies."""

    async def generate(self, request: RAGRequest, db: Session) -> RAGResponse:
        """
        Generate RAG response for a query.

        Pipeline:
        1. Validate model (if override requested)
        2. Retrieve quality-filtered context from VectorService
        3. Apply token budget (truncate history/context as needed)
        4. Build prompt with SourceIds for deterministic citations
        5. Call LLM for response
        6. Extract citations via SourceId regex
        7. Calculate hybrid confidence score
        8. Determine action (CITE or ROUTE)
        9. If ROUTE, look up tag owner

        Args:
            request: RAGRequest with query and user context
            db: Database session for tag owner lookup

        Returns:
            RAGResponse with answer, confidence, citations, action

        Raises:
            RAGServiceError subclasses for various failure modes
        """

    async def health_check(self) -> dict:
        """Check Ollama connectivity and model availability."""

    # Internal methods
    async def _get_quality_context(self, ...) -> list[SearchResult]
    def _build_context_with_source_ids(self, ...) -> str
    def _truncate_history(self, ...) -> str
    def _extract_citations(self, answer: str, context_map: dict) -> list[Citation]
    async def _evaluate_confidence(self, ...) -> ConfidenceScore
    def _calculate_coverage(self, answer: str, context: str) -> float
    async def _get_route_target(self, ...) -> RouteTarget
```

---

## Pipeline Flow (Revised)

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGService.generate()                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Validate Model                                                │
│    validate_model(request.model) → raises if invalid            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Retrieve Quality Context                                      │
│    _get_quality_context(query, user_tags, tenant_id)            │
│    - Over-fetch 3x candidates                                    │
│    - Filter by min_similarity_score (0.3)                       │
│    - Deduplicate (>90% similar)                                 │
│    - Limit per document (3 max)                                 │
│    - Return top-k                                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Apply Token Budget                                            │
│    TokenBudget.allocate(system, history, context)               │
│    - Reserve system prompt tokens                               │
│    - Truncate history (oldest first)                            │
│    - Drop lowest-scored chunks if over budget                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Build Prompt with SourceIds                                   │
│    _build_context_with_source_ids(chunks)                       │
│    Format: [SourceId: doc-id:chunk-idx]                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Generate Response                                             │
│    ChatOllama.invoke(messages) → answer                         │
│    Timeout: 30 seconds                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Extract Citations (Deterministic)                             │
│    Regex: \[SourceId:\s*([a-zA-Z0-9-]+:\d+)\]                   │
│    Map to SearchResult metadata                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Calculate Hybrid Confidence                                   │
│    - Retrieval score (30%): avg similarity                      │
│    - Coverage score (40%): answer↔context overlap               │
│    - LLM score (30%): self-assessment                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Determine Action                                              │
│    confidence >= 60 → CITE, grounded=True                       │
│    confidence < 60  → ROUTE, lookup tag owner                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Return RAGResponse                                            │
│    answer, confidence, citations, action, route_to, ...         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing Requirements

### Unit Tests

| Test | Description |
|------|-------------|
| test_build_context_with_source_ids | SourceIds formatted correctly |
| test_extract_citations_valid | Citations extracted from answer |
| test_extract_citations_invalid | Unknown SourceIds handled gracefully |
| test_calculate_coverage | Coverage score computed correctly |
| test_calculate_confidence_hybrid | Weights applied correctly |
| test_truncate_history_overflow | Oldest messages dropped |
| test_truncate_history_keeps_recent | Most recent preserved |
| test_deduplicate_chunks | Near-duplicates removed |
| test_limit_per_document | Max 3 chunks per doc |
| test_validate_model_allowed | Valid model passes |
| test_validate_model_not_allowed | Invalid model raises |
| test_get_route_target_with_owner | Owner found and returned |
| test_get_route_target_fallback | Admin fallback when no owner |

### Integration Tests

| Test | Description |
|------|-------------|
| test_generate_full_pipeline | End-to-end with real Ollama |
| test_generate_low_confidence_routes | Routes when confidence low |
| test_generate_no_context | Handles empty retrieval results |
| test_generate_token_budget | Large history truncated correctly |
| test_health_check | Ollama connectivity verified |
| test_model_override | Custom model selection works |

---

## Implementation Issues (Revised)

### Issue 013: RAG Service Configuration
**Complexity:** TRIVIAL
- Add RAG config fields to config.py (token budgets, thresholds)
- Add MODEL_ALLOWLIST configuration
- Document environment variables

### Issue 014: RAG Service Core
**Complexity:** MODERATE
- Implement RAGService class with quality retrieval
- Implement token budget management
- Implement SourceId-based citation extraction
- Add prompt templates

### Issue 015: Confidence Scoring
**Complexity:** SIMPLE
- Implement hybrid confidence calculation
- Implement coverage scoring
- Implement routing logic with tag owner lookup

### Issue 016: RAG Service Tests
**Complexity:** SIMPLE
- Unit tests for all methods
- Integration tests with Ollama
- Mock tests for CI (no Ollama)

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-28 | Initial specification |
| 2.0 | 2026-01-28 | Revised per engineering review: deterministic citations, hybrid confidence, token budgeting, retrieval quality controls, removed DIRECT path, clarified tag owner lookup, pure exception handling |
