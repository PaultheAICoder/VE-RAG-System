# RAG Service Specification

**Version:** 3.0
**Date:** January 28, 2026
**Status:** FINAL
**Depends On:** VectorService (specs/VECTOR_SERVICE.md), Ollama (llama3.2, qwen3:8b)

---

## Overview

The RAG Service orchestrates retrieval-augmented generation by combining context from VectorService with LLM responses from Ollama. It handles prompt construction, response generation, confidence scoring, citation formatting, and routing decisions.

**Key Principles:**
- **Context-grounded only**: All substantive answers MUST be grounded in retrieved context
- Uses VectorService for context retrieval (access control + tenant isolation enforced there)
- Deterministic citation mapping via SourceIds (no LLM-dependent parsing)
- Hybrid confidence scoring (retrieval signals + LLM evaluation)
- Routes to human when confidence is low (<60%)
- Explicit token budgeting and truncation strategies
- All operations are async
- Stateless service (session state managed by Chat API)

**Cross-References:**
- VectorService tenant filtering: `specs/VECTOR_SERVICE.md` (tenant_id passed to all queries)
- VectorService score semantics: Qdrant cosine similarity, normalized 0.0-1.0

---

## File Location

```
ai_ready_rag/
├── services/
│   ├── __init__.py
│   ├── vector_service.py    # Dependency (see VECTOR_SERVICE.md)
│   └── rag_service.py       # This specification
```

---

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| langchain-ollama | LLM client | >=0.3.0 |
| VectorService | Context retrieval | Internal |

**Note:** Token counting uses Ollama's native tokenizer via API, not tiktoken (model-specific accuracy).

---

## Configuration

Environment variables (loaded via config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| CHAT_MODEL | llama3.2 | Default LLM model |
| RAG_TEMPERATURE | 0.1 | LLM temperature (lower = more deterministic) |
| RAG_TIMEOUT_SECONDS | 30 | LLM response timeout |
| RAG_CONFIDENCE_THRESHOLD | 60 | Min confidence for CITE action (%) |
| RAG_ADMIN_EMAIL | admin@company.com | Fallback routing email |

### Token Budget Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| RAG_MAX_CONTEXT_TOKENS | 3000 | Max tokens for retrieved context |
| RAG_MAX_HISTORY_TOKENS | 1000 | Max tokens for chat history |
| RAG_MAX_RESPONSE_TOKENS | 1024 | Max tokens for LLM response |
| RAG_SYSTEM_PROMPT_TOKENS | 500 | Reserved for system prompt |

**Model Context Windows:**
| Model | Context Window | Available for RAG |
|-------|----------------|-------------------|
| llama3.2 | 8192 | 8192 - 500 (system) = 7692 |
| qwen3:8b | 32768 | 32768 - 500 = 32268 |
| deepseek-r1:32b | 65536 | 65536 - 500 = 65036 |

### Retrieval Quality Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| RAG_MIN_SIMILARITY_SCORE | 0.3 | Minimum cosine similarity (Qdrant 0-1 scale) |
| RAG_MAX_CHUNKS_PER_DOC | 3 | Max chunks from single document |
| RAG_TOTAL_CONTEXT_CHUNKS | 5 | Total chunks to include |
| RAG_DEDUP_CANDIDATES_CAP | 15 | Max candidates for dedup (3x chunks) |
| RAG_CHUNK_OVERLAP_THRESHOLD | 0.9 | Jaccard similarity for dedup |

---

## ID Formats

### document_id Format

**Format:** UUIDv4 string (lowercase, with hyphens)
**Pattern:** `[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}`
**Example:** `550e8400-e29b-41d4-a716-446655440000`

### SourceId Format

**Format:** `{document_id}:{chunk_index}`
**Pattern:** `[a-f0-9-]{36}:\d+`
**Example:** `550e8400-e29b-41d4-a716-446655440000:3`

### Citation Extraction Regex

```python
SOURCEID_PATTERN = r'\[SourceId:\s*([a-f0-9-]{36}:\d+)\]'
```

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
    user_tags: list[str]                # User's access tags (for VectorService filtering)
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
    grounded: bool                      # True iff len(citations) > 0
```

**grounded Definition:** `grounded = True` if and only if at least one valid SourceId was found in the answer and successfully mapped to a citation. This is independent of confidence score.

### ConfidenceScore

```python
@dataclass
class ConfidenceScore:
    """Hybrid confidence scoring with breakdown."""
    overall: int                        # 0-100 final score
    retrieval_score: float              # Average similarity of top chunks (0-1)
    coverage_score: float               # Keyword overlap between answer and context (0-1)
    llm_score: int                      # LLM self-assessment (0-100)
```

**Calculation:**
```
overall = (retrieval_score * 30) + (coverage_score * 40) + (llm_score * 0.3)
```

### Citation

```python
@dataclass
class Citation:
    """Deterministic source citation."""
    source_id: str                      # Format: "{document_id}:{chunk_index}"
    document_id: str                    # UUIDv4
    document_name: str
    chunk_index: int
    page_number: int | None             # None displayed as "N/A"
    section: str | None                 # None displayed as "N/A"
    relevance_score: float              # 0.0-1.0 from vector search
    snippet: str                        # First 200 chars, "..." if truncated
    snippet_full: str                   # Max 1000 chars (for UI hover only)
```

**Snippet Rules:**
- `snippet`: First 200 characters of chunk_text. Append "..." if truncated.
- `snippet_full`: First 1000 characters. For UI hover/expand only, NOT for logging/telemetry.
- `page_number`/`section`: Display as "N/A" if None.

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

## Zero-Context Handling

When VectorService returns zero results (empty retrieval):

```python
INSUFFICIENT_CONTEXT_RESPONSE = RAGResponse(
    answer="I don't have enough information in the available documents to answer this question. Please contact the relevant team for assistance.",
    confidence=ConfidenceScore(overall=0, retrieval_score=0.0, coverage_score=0.0, llm_score=0),
    citations=[],
    action="ROUTE",
    route_to=RouteTarget(
        tag="system",
        owner_user_id=None,
        owner_email=settings.rag_admin_email,
        reason="No relevant documents found",
        fallback=True,
    ),
    model_used=model,
    context_chunks_used=0,
    context_tokens_used=0,
    generation_time_ms=elapsed_ms,
    grounded=False,
)
```

**Pipeline short-circuits at step 2 if zero results.**

---

## Prompt Templates

### RAG System Prompt (with context)

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

STRICT RULES:
1. ONLY use information from the provided CONTEXT sections below
2. If the context doesn't contain enough information, respond with: "I don't have enough information in the available documents to answer this question."
3. NEVER use external knowledge or make assumptions beyond the context
4. Cite sources using the exact SourceId provided (e.g., [SourceId: 550e8400-e29b-41d4-a716-446655440000:0])
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

---

## Confidence Scoring (Hybrid)

### Calculation with Zero-Guard

```python
def calculate_confidence(
    retrieval_results: list[SearchResult],
    answer: str,
    context: str,
    llm_score: int,
) -> ConfidenceScore:
    """
    Combine multiple signals for robust confidence.

    Returns zero confidence if no retrieval results.
    """
    # Guard: zero results = zero confidence
    if not retrieval_results:
        return ConfidenceScore(
            overall=0,
            retrieval_score=0.0,
            coverage_score=0.0,
            llm_score=0,
        )

    # 1. Retrieval score: average similarity of used chunks
    retrieval_score = sum(r.score for r in retrieval_results) / len(retrieval_results)

    # 2. Coverage score: keyword overlap between answer and context
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
import re
from collections import Counter

# Common English stopwords (expand as needed)
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'this', 'that', 'these', 'those', 'i',
    'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours',
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
}

def extract_key_terms(text: str) -> set[str]:
    """
    Extract significant terms from text.

    Method:
    1. Lowercase and extract alphanumeric words
    2. Filter stopwords
    3. Filter short words (< 3 chars)
    """
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) >= 3}

def calculate_coverage(answer: str, context: str) -> float:
    """
    Estimate how much of the answer is grounded in context.

    Returns 0.0-1.0 representing fraction of answer terms found in context.
    """
    answer_terms = extract_key_terms(answer)
    context_terms = extract_key_terms(context)

    if not answer_terms:
        return 0.0

    matches = answer_terms & context_terms
    return len(matches) / len(answer_terms)
```

### Confidence Thresholds

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 80-100 | High confidence, well-grounded | CITE |
| 60-79 | Moderate confidence | CITE |
| 40-59 | Low confidence | ROUTE |
| 0-39 | Very low confidence | ROUTE |

---

## Token Budget Management

### Token Counting via Ollama

```python
async def count_tokens_ollama(self, text: str, model: str) -> int:
    """
    Count tokens using Ollama's native tokenizer.

    More accurate than tiktoken for non-OpenAI models.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": text,
                    "options": {"num_predict": 0},  # Don't generate, just tokenize
                },
                timeout=5.0,
            )
            data = response.json()
            return data.get("prompt_eval_count", len(text) // 4)  # Fallback estimate
    except Exception:
        # Fallback: estimate 1 token per 4 characters
        return len(text) // 4
```

### Token Budget Allocation

```python
class TokenBudget:
    """Manages token allocation for RAG prompt."""

    def __init__(self, model: str, settings: Settings):
        self.model = model
        self.max_context = settings.rag_max_context_tokens
        self.max_history = settings.rag_max_history_tokens
        self.max_response = settings.rag_max_response_tokens
        self.system_reserve = settings.rag_system_prompt_tokens
        self.context_window = MODEL_LIMITS[model]["context_window"]

    def allocate(
        self,
        system_prompt: str,
        chat_history: list[ChatMessage],
        context_chunks: list[SearchResult],
        count_fn: Callable[[str], int],
    ) -> tuple[str, str, list[SearchResult]]:
        """
        Allocate tokens and truncate as needed.

        Priority (highest to lowest):
        1. System prompt (fixed, never truncate)
        2. Most recent chat history (truncate oldest first)
        3. Highest-relevance context chunks (drop lowest-scored first)

        Raises:
            TokenBudgetExceededError: If system + response exceeds context window

        Returns:
            (system_prompt, truncated_history, filtered_chunks)
        """
        # Calculate available budget
        system_tokens = count_fn(system_prompt)
        available = self.context_window - system_tokens - self.max_response

        # Guard: negative budget is an error
        if available < 0:
            raise TokenBudgetExceededError(
                f"System prompt ({system_tokens}) + response reserve ({self.max_response}) "
                f"exceeds context window ({self.context_window})"
            )

        # Allocate history (cap at max_history or 1/3 of available)
        history_budget = min(self.max_history, available // 3)
        history_str, history_tokens = self._truncate_history(
            chat_history, history_budget, count_fn
        )
        available -= history_tokens

        # Allocate context (remaining budget, cap at max_context)
        context_budget = min(self.max_context, available)
        final_chunks = self._fit_context(context_chunks, context_budget, count_fn)

        return system_prompt, history_str, final_chunks

    def _truncate_history(
        self,
        history: list[ChatMessage],
        max_tokens: int,
        count_fn: Callable[[str], int],
    ) -> tuple[str, int]:
        """Truncate chat history, keeping most recent messages."""
        if not history:
            return "", 0

        included = []
        total_tokens = 0

        for msg in reversed(history):
            msg_text = f"{msg.role}: {msg.content}"
            msg_tokens = count_fn(msg_text)
            if total_tokens + msg_tokens <= max_tokens:
                included.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break

        formatted = "\n".join(f"{m.role}: {m.content}" for m in included)
        return formatted, total_tokens

    def _fit_context(
        self,
        chunks: list[SearchResult],
        max_tokens: int,
        count_fn: Callable[[str], int],
    ) -> list[SearchResult]:
        """Fit context chunks within token budget, dropping lowest-scored first."""
        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        included = []
        total_tokens = 0

        for chunk in sorted_chunks:
            chunk_tokens = count_fn(chunk.chunk_text)
            if total_tokens + chunk_tokens <= max_tokens:
                included.append(chunk)
                total_tokens += chunk_tokens

        return included
```

---

## Retrieval Quality Controls

### Quality Pipeline

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
    1. Retrieve 3x candidates from VectorService (capped at 15)
    2. Filter by minimum similarity score (0.3 on Qdrant 0-1 scale)
    3. Deduplicate near-identical chunks (Jaccard > 0.9)
    4. Limit chunks per document (max 3)
    5. Return top-k by score
    """
    # 1. Over-fetch candidates (capped to limit dedup cost)
    candidate_limit = min(max_chunks * 3, self.dedup_candidates_cap)

    candidates = await self.vector_service.search(
        query=query,
        user_tags=user_tags,
        tenant_id=tenant_id,  # Passed to VectorService for tenant isolation
        limit=candidate_limit,
        score_threshold=0.0,  # Filter later for more control
    )

    # Early exit: no candidates
    if not candidates:
        return []

    # 2. Filter by minimum similarity (Qdrant cosine is 0-1)
    filtered = [c for c in candidates if c.score >= self.min_similarity_score]

    # 3. Deduplicate (Jaccard similarity > 0.9)
    deduped = self._deduplicate_chunks(filtered)

    # 4. Limit per document
    limited = self._limit_per_document(deduped, max_per_doc=self.max_chunks_per_doc)

    # 5. Return top-k
    return limited[:max_chunks]
```

### Deduplication (Capped O(n²))

```python
def _deduplicate_chunks(self, chunks: list[SearchResult]) -> list[SearchResult]:
    """
    Remove near-duplicate chunks using Jaccard similarity.

    Complexity: O(n²) where n is capped at dedup_candidates_cap (default 15).
    At n=15, this is 105 comparisons - acceptable.
    """
    result = []
    for chunk in chunks:
        is_duplicate = False
        chunk_words = set(chunk.chunk_text.lower().split())

        for existing in result:
            existing_words = set(existing.chunk_text.lower().split())
            # Jaccard similarity
            intersection = len(chunk_words & existing_words)
            union = len(chunk_words | existing_words)
            similarity = intersection / union if union > 0 else 0

            if similarity > self.chunk_overlap_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(chunk)

    return result

def _limit_per_document(
    self,
    chunks: list[SearchResult],
    max_per_doc: int,
) -> list[SearchResult]:
    """Limit chunks per document to avoid single-source dominance."""
    doc_counts: dict[str, int] = {}
    result = []

    for chunk in chunks:
        count = doc_counts.get(chunk.document_id, 0)
        if count < max_per_doc:
            result.append(chunk)
            doc_counts[chunk.document_id] = count + 1

    return result
```

---

## Model Validation

### Allowlist Configuration

```python
MODEL_LIMITS = {
    "llama3.2": {"context_window": 8192, "max_response": 1024},
    "qwen3:8b": {"context_window": 32768, "max_response": 2048},
    "deepseek-r1:32b": {"context_window": 65536, "max_response": 4096},
}

async def validate_model(self, model: str) -> str:
    """
    Validate model is allowed and available.

    Raises:
        ModelNotAllowedError: Model not in allowlist
        ModelUnavailableError: Model not pulled in Ollama
    """
    if model not in MODEL_LIMITS:
        raise ModelNotAllowedError(
            f"Model '{model}' not in allowlist. "
            f"Available: {list(MODEL_LIMITS.keys())}"
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ollama_url}/api/tags",
                timeout=5.0
            )
            available = [m["name"].split(":")[0] for m in response.json().get("models", [])]

            if model not in available:
                raise ModelUnavailableError(
                    f"Model '{model}' not available. Run: ollama pull {model}"
                )
    except httpx.RequestError as e:
        raise LLMConnectionError(f"Cannot connect to Ollama: {e}")

    return model
```

---

## Tag Owner Lookup

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
    # Handle empty context
    if not context_results:
        return self._admin_fallback("No context available")

    # 1. Find primary tag from context
    tag_counts: dict[str, int] = {}
    for result in context_results:
        for tag in getattr(result, 'tags', []):
            if tag != 'public':  # Skip public tag for routing
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        return self._admin_fallback("No specific tags in context")

    primary_tag = max(tag_counts, key=tag_counts.get)

    # 2. Look up tag owner
    tag_record = db.query(Tag).filter(Tag.name == primary_tag).first()

    if not tag_record or not tag_record.owner_user_id:
        return self._admin_fallback(f"No owner for tag '{primary_tag}'")

    # 3. Get owner details
    owner = db.query(User).filter(User.id == tag_record.owner_user_id).first()

    if not owner:
        return self._admin_fallback(f"Owner not found for tag '{primary_tag}'")

    return RouteTarget(
        tag=primary_tag,
        owner_user_id=owner.id,
        owner_email=owner.email,
        reason=f"Routing to {primary_tag} owner",
        fallback=False,
    )

def _admin_fallback(self, reason: str) -> RouteTarget:
    """Return admin as fallback route target."""
    return RouteTarget(
        tag="system",
        owner_user_id=None,
        owner_email=self.settings.rag_admin_email,
        reason=f"{reason} - routing to admin",
        fallback=True,
    )
```

---

## Error Handling (Service Layer)

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
    """System prompt + response reserve exceeds context window."""
    pass
```

**API layer is responsible for mapping exceptions to HTTP responses.**

---

## Service Interface

```python
class RAGService:
    """Orchestrates RAG pipeline: retrieve → prompt → generate → evaluate."""

    def __init__(
        self,
        vector_service: VectorService,
        settings: Settings,
        ollama_url: str | None = None,
        default_model: str | None = None,
    ):
        self.vector_service = vector_service
        self.settings = settings
        self.ollama_url = ollama_url or settings.ollama_base_url
        self.default_model = default_model or settings.chat_model
        self.min_similarity_score = settings.rag_min_similarity_score
        self.max_chunks_per_doc = settings.rag_max_chunks_per_doc
        self.chunk_overlap_threshold = settings.rag_chunk_overlap_threshold
        self.dedup_candidates_cap = settings.rag_dedup_candidates_cap
        self.confidence_threshold = settings.rag_confidence_threshold

    async def generate(self, request: RAGRequest, db: Session) -> RAGResponse:
        """Generate RAG response for a query."""
        # Implementation follows pipeline in next section

    async def health_check(self) -> dict:
        """Check Ollama connectivity and model availability."""

    # Internal methods documented in Pipeline section
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGService.generate()                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Validate Model                                                │
│    validate_model(request.model or default)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Retrieve Quality Context                                      │
│    get_quality_context(query, user_tags, tenant_id)             │
│    ├─ If empty → return INSUFFICIENT_CONTEXT_RESPONSE           │
│    └─ Else continue                                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Apply Token Budget                                            │
│    TokenBudget.allocate(system, history, context)               │
│    ├─ If TokenBudgetExceededError → raise                       │
│    └─ Else continue with truncated inputs                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Build Prompt with SourceIds                                   │
│    Format: [SourceId: {uuid}:{index}]                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Generate Response                                             │
│    ChatOllama.invoke(messages) → answer                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Extract Citations (Deterministic)                             │
│    Regex: \[SourceId:\s*([a-f0-9-]{36}:\d+)\]                   │
│    Map to SearchResult metadata                                 │
│    Set grounded = len(citations) > 0                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Calculate Hybrid Confidence                                   │
│    retrieval (30%) + coverage (40%) + LLM (30%)                 │
│    Guard: zero results = zero confidence                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Determine Action                                              │
│    confidence >= 60 → CITE                                      │
│    confidence < 60  → ROUTE (lookup tag owner)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Return RAGResponse                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing Requirements

### Unit Tests

| Test | Description |
|------|-------------|
| test_build_context_with_source_ids | SourceIds match UUID:index format |
| test_extract_citations_valid | Citations extracted from answer |
| test_extract_citations_invalid_format | Malformed SourceIds ignored |
| test_extract_citations_unknown_id | Unknown SourceIds logged, not included |
| test_calculate_coverage | Term overlap computed correctly |
| test_calculate_coverage_empty_answer | Returns 0.0 |
| test_calculate_confidence_zero_results | Returns overall=0 |
| test_calculate_confidence_hybrid | Weights applied correctly |
| test_truncate_history_overflow | Oldest messages dropped |
| test_truncate_history_keeps_recent | Most recent preserved |
| test_token_budget_exceeded | Raises TokenBudgetExceededError |
| test_deduplicate_chunks | Near-duplicates removed |
| test_deduplicate_respects_cap | Max 15 candidates processed |
| test_limit_per_document | Max 3 chunks per doc |
| test_validate_model_allowed | Valid model passes |
| test_validate_model_not_allowed | Invalid model raises |
| test_get_route_target_with_owner | Owner found and returned |
| test_get_route_target_no_owner | Admin fallback |
| test_get_route_target_empty_context | Admin fallback |
| test_grounded_true_with_citations | grounded=True when citations exist |
| test_grounded_false_no_citations | grounded=False when no citations |

### Integration Tests

| Test | Description |
|------|-------------|
| test_generate_full_pipeline | End-to-end with real Ollama |
| test_generate_zero_context | Returns INSUFFICIENT_CONTEXT_RESPONSE |
| test_generate_low_confidence_routes | Routes when confidence < 60 |
| test_generate_token_budget | Large history truncated correctly |
| test_health_check | Ollama connectivity verified |
| test_model_override | Custom model selection works |
| test_tenant_isolation | tenant_id passed to VectorService |

---

## Implementation Issues

### Issue 013: RAG Service Configuration
**Complexity:** TRIVIAL
- Add RAG config fields to config.py
- Add MODEL_LIMITS configuration
- Add STOPWORDS constant

### Issue 014: RAG Service Core
**Complexity:** MODERATE
- Implement RAGService class
- Implement quality retrieval pipeline
- Implement token budget management
- Implement SourceId-based citation extraction
- Add prompt templates

### Issue 015: Confidence and Routing
**Complexity:** SIMPLE
- Implement hybrid confidence calculation
- Implement coverage scoring with extract_key_terms
- Implement routing logic with tag owner lookup
- Handle zero-context case

### Issue 016: RAG Service Tests
**Complexity:** MODERATE
- Unit tests for all methods (20+ tests)
- Integration tests with Ollama
- Mock VectorService for CI testing

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-28 | Initial specification |
| 2.0 | 2026-01-28 | Revised per engineering review round 1 |
| 3.0 | 2026-01-28 | FINAL - Addressed round 2: zero-division guard, UUID document_id format, strict SourceId regex, Ollama tokenizer, negative budget check, grounded definition, extract_key_terms implementation, removed unused query classifier, dedup cap, config mapping, snippet limits, score semantics confirmed |
