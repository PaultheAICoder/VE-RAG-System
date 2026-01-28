# RAG Service Specification

**Version:** 1.0
**Date:** January 28, 2026
**Status:** Draft
**Depends On:** VectorService, Ollama (llama3.2, qwen3:8b)

---

## Overview

The RAG Service orchestrates retrieval-augmented generation by combining context from VectorService with LLM responses from Ollama. It handles prompt construction, response generation, confidence scoring, citation formatting, and routing decisions.

**Key Principles:**
- Uses VectorService for context retrieval (access control already enforced)
- Supports multiple LLM models (configurable)
- Generates confidence scores for response quality
- Routes to human when confidence is low (<60%)
- Produces citations with source references
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
| VectorService | Context retrieval | Internal |

---

## Configuration

Environment variables (loaded via config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| CHAT_MODEL | llama3.2 | Default LLM model |
| RAG_TEMPERATURE | 0.1 | LLM temperature (lower = more deterministic) |
| RAG_MAX_TOKENS | 2048 | Maximum response tokens |
| RAG_CONTEXT_CHUNKS | 5 | Number of chunks to include in context |
| RAG_CONFIDENCE_THRESHOLD | 60 | Minimum confidence for direct response (%) |
| RAG_TIMEOUT_SECONDS | 30 | LLM response timeout |

---

## Data Models

### RAGRequest

```python
@dataclass
class RAGRequest:
    """Input for RAG generation."""
    query: str                      # User's question
    user_tags: list[str]            # User's access tags (for filtering)
    tenant_id: str = "default"      # Multi-tenant isolation
    chat_history: list[dict] | None = None  # Previous messages for context
    model: str | None = None        # Override default model
    max_context_chunks: int = 5     # Limit context size
```

### RAGResponse

```python
@dataclass
class RAGResponse:
    """Output from RAG generation."""
    answer: str                     # Generated response text
    confidence: int                 # 0-100 confidence score
    citations: list[Citation]       # Source references
    action: Literal["CITE", "ROUTE"]  # Response action
    route_to: str | None            # Tag owner if ROUTE action
    model_used: str                 # Actual model used
    context_chunks: int             # Number of chunks used
    generation_time_ms: float       # Response time
```

### Citation

```python
@dataclass
class Citation:
    """Source citation for response."""
    document_id: str
    document_name: str
    chunk_index: int
    page_number: int | None
    section: str | None
    relevance_score: float          # 0.0-1.0 from vector search
    snippet: str                    # Relevant text excerpt (truncated)
```

---

## Prompt Templates

### RAG Prompt (with context)

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Always cite your sources using [Source: document_name, page X] format
4. Be concise but thorough
5. If you're uncertain, indicate your level of confidence

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}
"""

RAG_USER_PROMPT = "{query}"
```

### Direct Response Prompt (no retrieval needed)

```python
DIRECT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the following question directly.
If you don't know the answer, say so. Do not make up information."""

DIRECT_USER_PROMPT = "{query}"
```

### Confidence Evaluation Prompt

```python
CONFIDENCE_PROMPT = """Rate the confidence of this answer on a scale of 0-100.

QUESTION: {query}
CONTEXT PROVIDED: {context_summary}
ANSWER GIVEN: {answer}

Consider:
- Does the answer directly address the question?
- Is the answer fully supported by the context?
- Are there any gaps or uncertainties?

Respond with ONLY a number between 0 and 100."""
```

### Query Router Prompt

```python
ROUTER_PROMPT = """Determine if this query requires document retrieval.

QUERY: {query}

Respond with exactly one of:
- RETRIEVE: Query needs information from documents
- DIRECT: Query can be answered without documents (greetings, clarifications, general knowledge)

Response:"""
```

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
        max_tokens: int = 2048,
        confidence_threshold: int = 60,
        timeout_seconds: int = 30,
    ):
        """Initialize RAG service with dependencies."""

    async def generate(self, request: RAGRequest) -> RAGResponse:
        """
        Generate RAG response for a query.

        Pipeline:
        1. Route query (RETRIEVE or DIRECT)
        2. If RETRIEVE: get context from VectorService
        3. Build prompt with context and history
        4. Call LLM for response
        5. Evaluate confidence
        6. Format citations
        7. Determine action (CITE or ROUTE)

        Args:
            request: RAGRequest with query and user context

        Returns:
            RAGResponse with answer, confidence, citations, action
        """

    async def route_query(self, query: str) -> Literal["RETRIEVE", "DIRECT"]:
        """
        Determine if query needs document retrieval.

        Returns:
            "RETRIEVE" for document-based queries
            "DIRECT" for simple queries (greetings, clarifications)
        """

    async def evaluate_confidence(
        self,
        query: str,
        answer: str,
        context_summary: str,
    ) -> int:
        """
        Evaluate answer confidence using LLM.

        Returns:
            Integer 0-100 representing confidence percentage
        """

    async def health_check(self) -> dict:
        """
        Check Ollama connectivity and model availability.

        Returns:
            {"healthy": bool, "model": str, "latency_ms": float}
        """

    def _build_context(self, search_results: list[SearchResult]) -> str:
        """Format search results into context string for prompt."""

    def _build_chat_history(self, history: list[dict] | None) -> str:
        """Format chat history for prompt inclusion."""

    def _format_citations(self, search_results: list[SearchResult]) -> list[Citation]:
        """Convert search results to citation format."""

    def _determine_action(
        self,
        confidence: int,
        user_tags: list[str],
    ) -> tuple[Literal["CITE", "ROUTE"], str | None]:
        """
        Determine response action based on confidence.

        Returns:
            ("CITE", None) if confidence >= threshold
            ("ROUTE", tag_owner) if confidence < threshold
        """
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
│ 1. Route Query                                                   │
│    route_query(query) → RETRIEVE | DIRECT                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │   RETRIEVE    │       │    DIRECT     │
            └───────────────┘       └───────────────┘
                    │                       │
                    ▼                       │
┌─────────────────────────────────────┐     │
│ 2. Get Context                      │     │
│    VectorService.search(            │     │
│      query, user_tags, limit=5      │     │
│    )                                │     │
└─────────────────────────────────────┘     │
                    │                       │
                    ▼                       │
┌─────────────────────────────────────┐     │
│ 3. Build Prompt                     │     │
│    - Format context chunks          │     │
│    - Include chat history           │     │
│    - Apply RAG_SYSTEM_PROMPT        │     │
└─────────────────────────────────────┘     │
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Generate Response                                             │
│    ChatOllama.invoke(messages) → answer                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Evaluate Confidence                                           │
│    evaluate_confidence(query, answer, context) → 0-100          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Determine Action                                              │
│    confidence >= 60 → CITE                                       │
│    confidence < 60  → ROUTE to tag owner                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Format Response                                               │
│    RAGResponse(answer, confidence, citations, action, ...)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Context Formatting

### Chunk Format in Prompt

```
[Document: {document_name}]
[Page: {page_number}] [Section: {section}]
---
{chunk_text}
---

[Document: {document_name}]
...
```

### Citation Format in Response

The LLM is instructed to cite sources inline:

```
Based on the HR policy [Source: hr-handbook.pdf, page 12], employees are entitled to...
```

The service then parses these citations and maps them to the actual document metadata.

---

## Confidence Scoring

### Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 80-100 | High confidence, well-supported | CITE |
| 60-79 | Moderate confidence, mostly supported | CITE |
| 40-59 | Low confidence, partial support | ROUTE |
| 0-39 | Very low confidence, insufficient context | ROUTE |

### Routing Logic

When confidence < threshold (default 60%):
1. Identify the most relevant tag from context documents
2. Set `route_to` = tag owner (lookup from tag metadata)
3. Set `action` = "ROUTE"
4. Include message suggesting user contact the tag owner

---

## Error Handling

### Exception Hierarchy

```python
class RAGServiceError(Exception):
    """Base exception for RAG service errors."""
    pass

class LLMError(RAGServiceError):
    """Failed to generate LLM response."""
    pass

class ContextError(RAGServiceError):
    """Failed to retrieve or format context."""
    pass

class ConfidenceError(RAGServiceError):
    """Failed to evaluate confidence."""
    pass
```

### Error Responses

| Error | HTTP Status | User Message |
|-------|-------------|--------------|
| LLM timeout | 504 | "Response generation timed out. Please try again." |
| LLM unavailable | 503 | "AI service temporarily unavailable." |
| No context found | 200 | Normal response with low confidence, suggests routing |
| Invalid model | 400 | "Requested model not available." |

---

## Model Selection

### Available Models

| Model | Use Case | Speed | Quality |
|-------|----------|-------|---------|
| llama3.2 | General queries | Fast | Good |
| qwen3:8b | Complex reasoning | Medium | Better |
| deepseek-r1:32b | Deep analysis | Slow | Best |

### Model Override

Users can request specific models via `RAGRequest.model`. The service validates the model is available before use.

---

## Testing Requirements

### Unit Tests

| Test | Description |
|------|-------------|
| test_route_query_retrieve | Document queries route to RETRIEVE |
| test_route_query_direct | Greetings route to DIRECT |
| test_build_context | Context formatted correctly |
| test_build_chat_history | History formatted correctly |
| test_format_citations | Citations extracted and formatted |
| test_determine_action_cite | High confidence returns CITE |
| test_determine_action_route | Low confidence returns ROUTE |

### Integration Tests

| Test | Description |
|------|-------------|
| test_generate_with_context | Full pipeline with document retrieval |
| test_generate_direct | Direct response without retrieval |
| test_generate_low_confidence | Routes when confidence low |
| test_health_check | Ollama connectivity verified |
| test_model_override | Custom model selection works |

---

## Implementation Issues

### Issue 013: RAG Service Configuration
**Complexity:** TRIVIAL
- Add RAG config fields to config.py
- Add environment variable documentation

### Issue 014: RAG Service Core
**Complexity:** SIMPLE
- Implement RAGService class
- Implement generate() pipeline
- Add prompt templates

### Issue 015: Confidence Scoring
**Complexity:** SIMPLE
- Implement evaluate_confidence()
- Implement determine_action()
- Add routing logic

### Issue 016: RAG Service Tests
**Complexity:** SIMPLE
- Unit tests for all methods
- Integration tests with Ollama
- Mock tests for CI (no Ollama)

---

## Usage Example

```python
from ai_ready_rag.services import VectorService, RAGService
from ai_ready_rag.services.rag_service import RAGRequest

# Initialize services
vector_service = VectorService()
await vector_service.initialize()

rag_service = RAGService(vector_service=vector_service)

# Generate response
request = RAGRequest(
    query="What is the vacation policy?",
    user_tags=["hr", "employee"],
    tenant_id="acme-corp",
)

response = await rag_service.generate(request)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}%")
print(f"Action: {response.action}")
for citation in response.citations:
    print(f"  - {citation.document_name}, page {citation.page_number}")
```

---

## Future Enhancements

1. **Streaming responses** - Stream tokens as generated
2. **Multi-query synthesis** - Combine results from query expansion
3. **Caching** - Cache common query responses
4. **Feedback loop** - Track citation clicks to improve ranking
5. **Custom prompts** - Per-tenant prompt customization

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-28 | Initial specification |
