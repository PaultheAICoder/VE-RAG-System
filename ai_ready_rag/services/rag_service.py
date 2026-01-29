"""RAG Service for retrieval-augmented generation.

Orchestrates: retrieve -> prompt -> generate -> evaluate.
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx
from langchain_ollama import ChatOllama
from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.core.exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
    ModelNotAllowedError,
    ModelUnavailableError,
    RAGServiceError,
    TokenBudgetExceededError,
)
from ai_ready_rag.services.factory import get_vector_service
from ai_ready_rag.services.rag_constants import MODEL_LIMITS, STOPWORDS
from ai_ready_rag.services.vector_service import SearchResult

logger = logging.getLogger(__name__)

# SourceId extraction pattern: [SourceId: {uuid}:{index}]
SOURCEID_PATTERN = r"\[SourceId:\s*([a-f0-9-]{36}:\d+)\]"


# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """Single message in chat history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime | None = None


@dataclass
class RAGRequest:
    """Input for RAG generation."""

    query: str
    user_tags: list[str]
    tenant_id: str = "default"
    chat_history: list[ChatMessage] | None = None
    model: str | None = None
    max_context_chunks: int = 5


@dataclass
class ConfidenceScore:
    """Hybrid confidence scoring with breakdown."""

    overall: int  # 0-100 final score
    retrieval_score: float  # Average similarity of top chunks (0-1)
    coverage_score: float  # Keyword overlap between answer and context (0-1)
    llm_score: int  # LLM self-assessment (0-100)


@dataclass
class Citation:
    """Deterministic source citation."""

    source_id: str  # Format: "{document_id}:{chunk_index}"
    document_id: str
    document_name: str
    chunk_index: int
    page_number: int | None
    section: str | None
    relevance_score: float  # 0.0-1.0 from vector search
    snippet: str  # First 200 chars, "..." if truncated
    snippet_full: str  # Max 1000 chars (for UI hover only)


@dataclass
class RouteTarget:
    """Routing destination when confidence is low."""

    tag: str
    owner_user_id: str | None
    owner_email: str | None
    reason: str
    fallback: bool  # True if no owner found, route to admin


@dataclass
class RAGResponse:
    """Output from RAG generation."""

    answer: str
    confidence: ConfidenceScore
    citations: list[Citation]
    action: Literal["CITE", "ROUTE"]
    route_to: RouteTarget | None
    model_used: str
    context_chunks_used: int
    context_tokens_used: int
    generation_time_ms: float
    grounded: bool  # True iff len(citations) > 0


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def extract_key_terms(text: str) -> set[str]:
    """Extract significant terms from text.

    Method:
    1. Lowercase and extract alphanumeric words
    2. Filter stopwords
    3. Filter short words (< 3 chars)
    """
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) >= 3}


def calculate_coverage(answer: str, context: str) -> float:
    """Estimate how much of the answer is grounded in context.

    Returns 0.0-1.0 representing fraction of answer terms found in context.
    """
    answer_terms = extract_key_terms(answer)
    context_terms = extract_key_terms(context)

    if not answer_terms:
        return 0.0

    matches = answer_terms & context_terms
    return len(matches) / len(answer_terms)


# -----------------------------------------------------------------------------
# Token Budget Management
# -----------------------------------------------------------------------------


class TokenBudget:
    """Manages token allocation for RAG prompt."""

    def __init__(self, model: str, settings: Settings):
        """Initialize token budget with model limits and settings.

        Args:
            model: Model name (must be in MODEL_LIMITS)
            settings: Application settings with token budget config
        """
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
        """Allocate tokens and truncate as needed.

        Priority (highest to lowest):
        1. System prompt (fixed, never truncate)
        2. Most recent chat history (truncate oldest first)
        3. Highest-relevance context chunks (drop lowest-scored first)

        Args:
            system_prompt: The system prompt text
            chat_history: List of chat messages
            context_chunks: Retrieved context chunks
            count_fn: Function to count tokens in text

        Returns:
            Tuple of (system_prompt, truncated_history, filtered_chunks)

        Raises:
            TokenBudgetExceededError: If system + response exceeds context window
        """
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
        history_str, history_tokens = self._truncate_history(chat_history, history_budget, count_fn)
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
        """Truncate chat history, keeping most recent messages.

        Args:
            history: List of chat messages
            max_tokens: Maximum tokens for history
            count_fn: Token counting function

        Returns:
            Tuple of (formatted_history, total_tokens)
        """
        if not history:
            return "", 0

        included: list[ChatMessage] = []
        total_tokens = 0

        # Iterate from most recent to oldest
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
        """Fit context chunks within token budget, dropping lowest-scored first.

        Args:
            chunks: List of search results
            max_tokens: Maximum tokens for context
            count_fn: Token counting function

        Returns:
            List of chunks that fit within budget
        """
        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        included: list[SearchResult] = []
        total_tokens = 0

        for chunk in sorted_chunks:
            chunk_tokens = count_fn(chunk.chunk_text)
            if total_tokens + chunk_tokens <= max_tokens:
                included.append(chunk)
                total_tokens += chunk_tokens

        return included


# -----------------------------------------------------------------------------
# RAG Service
# -----------------------------------------------------------------------------


class RAGService:
    """Orchestrates RAG pipeline: retrieve -> prompt -> generate -> evaluate.

    Uses factory pattern to obtain vector service based on ENV_PROFILE.
    """

    def __init__(
        self,
        settings: Settings,
        vector_service: "SearchResult | None" = None,
        ollama_url: str | None = None,
        default_model: str | None = None,
    ):
        """Initialize RAG service.

        Args:
            settings: Application settings
            vector_service: Optional VectorService override (uses factory if None)
            ollama_url: Optional Ollama URL override
            default_model: Optional default model override
        """
        self.settings = settings
        self._vector_service = vector_service
        self.ollama_url = ollama_url or settings.ollama_base_url
        self.default_model = default_model or settings.chat_model
        self.min_similarity_score = settings.rag_min_similarity_score
        self.max_chunks_per_doc = settings.rag_max_chunks_per_doc
        self.chunk_overlap_threshold = settings.rag_chunk_overlap_threshold
        self.dedup_candidates_cap = settings.rag_dedup_candidates_cap
        self.confidence_threshold = settings.rag_confidence_threshold

    @property
    def vector_service(self):
        """Get vector service, creating via factory if needed."""
        if self._vector_service is None:
            self._vector_service = get_vector_service(self.settings)
        return self._vector_service

    async def validate_model(self, model: str) -> str:
        """Validate model is allowed and available.

        Args:
            model: Model name to validate

        Returns:
            The validated model name

        Raises:
            ModelNotAllowedError: Model not in allowlist
            ModelUnavailableError: Model not pulled in Ollama
            LLMConnectionError: Cannot connect to Ollama
        """
        if model not in MODEL_LIMITS:
            raise ModelNotAllowedError(
                f"Model '{model}' not in allowlist. Available: {list(MODEL_LIMITS.keys())}"
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags", timeout=5.0)
                models_data = response.json().get("models", [])
                available = [m["name"].split(":")[0] for m in models_data]

                # Also check full model name with tag
                available_full = [m["name"] for m in models_data]

                if model not in available and model not in available_full:
                    raise ModelUnavailableError(
                        f"Model '{model}' not available. Run: ollama pull {model}"
                    )
        except httpx.RequestError as e:
            raise LLMConnectionError(f"Cannot connect to Ollama: {e}") from e

        return model

    async def get_quality_context(
        self,
        query: str,
        user_tags: list[str],
        tenant_id: str,
        max_chunks: int = 5,
    ) -> list[SearchResult]:
        """Retrieve context with quality controls.

        Pipeline:
        1. Retrieve 3x candidates from VectorService (capped at 15)
        2. Filter by minimum similarity score (0.3 on Qdrant 0-1 scale)
        3. Deduplicate near-identical chunks (Jaccard > 0.9)
        4. Limit chunks per document (max 3)
        5. Return top-k by score

        Args:
            query: User's question
            user_tags: User's access tags
            tenant_id: Tenant identifier
            max_chunks: Maximum chunks to return

        Returns:
            List of quality-filtered search results
        """
        candidate_limit = min(max_chunks * 3, self.dedup_candidates_cap)

        candidates = await self.vector_service.search(
            query=query,
            user_tags=user_tags,
            tenant_id=tenant_id,
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

    def _deduplicate_chunks(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Remove near-duplicate chunks using Jaccard similarity.

        Complexity: O(n^2) where n is capped at dedup_candidates_cap (default 15).
        At n=15, this is 105 comparisons - acceptable.

        Args:
            chunks: List of search results

        Returns:
            Deduplicated list
        """
        result: list[SearchResult] = []
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
        """Limit chunks per document to avoid single-source dominance.

        Args:
            chunks: List of search results
            max_per_doc: Maximum chunks per document

        Returns:
            Limited list
        """
        doc_counts: dict[str, int] = {}
        result: list[SearchResult] = []

        for chunk in chunks:
            count = doc_counts.get(chunk.document_id, 0)
            if count < max_per_doc:
                result.append(chunk)
                doc_counts[chunk.document_id] = count + 1

        return result

    async def count_tokens_ollama(self, text: str, model: str) -> int:
        """Count tokens using Ollama's native tokenizer.

        More accurate than tiktoken for non-OpenAI models.

        Args:
            text: Text to count tokens for
            model: Model name for tokenizer

        Returns:
            Token count (falls back to char estimate on error)
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

    def _build_context_prompt(self, chunks: list[SearchResult]) -> str:
        """Format chunks with SourceIds for the prompt.

        Args:
            chunks: List of search results

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        formatted_chunks = []
        for chunk in chunks:
            source_id = f"{chunk.document_id}:{chunk.chunk_index}"
            page = chunk.page_number if chunk.page_number else "N/A"
            section = chunk.section if chunk.section else "N/A"

            formatted = f"""[SourceId: {source_id}]
[Document: {chunk.document_name}]
[Page: {page}] [Section: {section}]
---
{chunk.chunk_text}
---"""
            formatted_chunks.append(formatted)

        return "\n\n".join(formatted_chunks)

    def _extract_citations(
        self,
        answer: str,
        chunks: list[SearchResult],
    ) -> list[Citation]:
        """Extract citations from answer using SourceId pattern.

        Args:
            answer: LLM-generated answer
            chunks: Context chunks used for generation

        Returns:
            List of citations found in answer
        """
        source_ids = re.findall(SOURCEID_PATTERN, answer)

        # Build lookup map from chunks
        chunk_map: dict[str, SearchResult] = {}
        for chunk in chunks:
            sid = f"{chunk.document_id}:{chunk.chunk_index}"
            chunk_map[sid] = chunk

        citations: list[Citation] = []
        seen_ids: set[str] = set()

        for source_id in source_ids:
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)

            if source_id not in chunk_map:
                logger.warning(f"Unknown SourceId in answer: {source_id}")
                continue

            chunk = chunk_map[source_id]

            # Create snippet (first 200 chars)
            snippet = chunk.chunk_text[:200]
            if len(chunk.chunk_text) > 200:
                snippet += "..."

            # Create full snippet (first 1000 chars)
            snippet_full = chunk.chunk_text[:1000]
            if len(chunk.chunk_text) > 1000:
                snippet_full += "..."

            citations.append(
                Citation(
                    source_id=source_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    relevance_score=chunk.score,
                    snippet=snippet,
                    snippet_full=snippet_full,
                )
            )

        return citations

    def _calculate_confidence(
        self,
        retrieval_results: list[SearchResult],
        answer: str,
        context: str,
        llm_score: int,
    ) -> ConfidenceScore:
        """Combine multiple signals for robust confidence.

        Args:
            retrieval_results: Search results used
            answer: LLM-generated answer
            context: Combined context text
            llm_score: LLM self-assessment (0-100)

        Returns:
            ConfidenceScore with breakdown
        """
        # Guard: zero results = zero confidence
        if not retrieval_results:
            return ConfidenceScore(
                overall=0,
                retrieval_score=0.0,
                coverage_score=0.0,
                llm_score=0,
            )

        retrieval_score = sum(r.score for r in retrieval_results) / len(retrieval_results)
        coverage_score = calculate_coverage(answer, context)
        llm_normalized = llm_score / 100

        # Weighted combination: retrieval (30%) + coverage (40%) + LLM (30%)
        overall = int((retrieval_score * 30) + (coverage_score * 40) + (llm_normalized * 30))

        return ConfidenceScore(
            overall=min(100, max(0, overall)),
            retrieval_score=retrieval_score,
            coverage_score=coverage_score,
            llm_score=llm_score,
        )

    def _admin_fallback(self, reason: str) -> RouteTarget:
        """Return admin as fallback route target.

        Args:
            reason: Reason for routing

        Returns:
            RouteTarget pointing to admin
        """
        return RouteTarget(
            tag="system",
            owner_user_id=None,
            owner_email=self.settings.rag_admin_email,
            reason=f"{reason} - routing to admin",
            fallback=True,
        )

    async def get_route_target(
        self,
        context_results: list[SearchResult],
        db: Session,
    ) -> RouteTarget:
        """Determine routing target based on context tags.

        Lookup chain:
        1. Find most common tag in retrieved documents
        2. Look up tag in `tags` table
        3. If tag.owner_id exists, get user email
        4. If no owner, fallback to system admin

        Args:
            context_results: Search results with tags
            db: Database session for lookups

        Returns:
            RouteTarget with owner info or admin fallback
        """
        from ai_ready_rag.db.models import Tag, User

        # Handle empty context
        if not context_results:
            return self._admin_fallback("No context available")

        # 1. Find primary tag from context
        tag_counts: dict[str, int] = {}
        for result in context_results:
            for tag in getattr(result, "tags", []):
                if tag != "public":  # Skip public tag for routing
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if not tag_counts:
            return self._admin_fallback("No specific tags in context")

        primary_tag = max(tag_counts, key=tag_counts.get)

        # 2. Look up tag owner
        tag_record = db.query(Tag).filter(Tag.name == primary_tag).first()

        if not tag_record or not tag_record.owner_id:
            return self._admin_fallback(f"No owner for tag '{primary_tag}'")

        # 3. Get owner details
        owner = db.query(User).filter(User.id == tag_record.owner_id).first()

        if not owner:
            return self._admin_fallback(f"Owner not found for tag '{primary_tag}'")

        return RouteTarget(
            tag=primary_tag,
            owner_user_id=owner.id,
            owner_email=owner.email,
            reason=f"Routing to {primary_tag} owner",
            fallback=False,
        )

    def _insufficient_context_response(
        self,
        model: str,
        elapsed_ms: float,
    ) -> RAGResponse:
        """Create response for zero-context case.

        Args:
            model: Model name for response metadata
            elapsed_ms: Elapsed time in milliseconds

        Returns:
            RAGResponse with zero confidence and admin routing
        """
        return RAGResponse(
            answer="I don't have enough information in the available documents to answer this question. Please contact the relevant team for assistance.",
            confidence=ConfidenceScore(
                overall=0,
                retrieval_score=0.0,
                coverage_score=0.0,
                llm_score=0,
            ),
            citations=[],
            action="ROUTE",
            route_to=RouteTarget(
                tag="system",
                owner_user_id=None,
                owner_email=self.settings.rag_admin_email,
                reason="No relevant documents found",
                fallback=True,
            ),
            model_used=model,
            context_chunks_used=0,
            context_tokens_used=0,
            generation_time_ms=elapsed_ms,
            grounded=False,
        )

    async def generate(self, request: RAGRequest, db: Session) -> RAGResponse:
        """Generate RAG response for a query.

        Pipeline:
        1. Validate model
        2. Retrieve quality context (short-circuit if zero results)
        3. Apply token budget
        4. Build prompt with SourceIds
        5. Generate response
        6. Extract citations (deterministic)
        7. Calculate hybrid confidence (retrieval + coverage + LLM)
        8. Determine action (CITE if >= threshold, ROUTE with tag owner lookup)
        9. Return RAGResponse

        Args:
            request: RAG request with query and context
            db: Database session (for routing lookup)

        Returns:
            RAGResponse with answer, citations, and metadata

        Raises:
            ModelNotAllowedError: Model not in allowlist
            ModelUnavailableError: Model not available in Ollama
            LLMConnectionError: Cannot connect to Ollama
            LLMTimeoutError: LLM response timed out
            TokenBudgetExceededError: Token budget exceeded
            RAGServiceError: Other generation errors
        """
        start_time = time.perf_counter()

        # 1. Validate model
        model = request.model or self.default_model
        await self.validate_model(model)

        # 2. Retrieve quality context
        context_chunks = await self.get_quality_context(
            query=request.query,
            user_tags=request.user_tags,
            tenant_id=request.tenant_id,
            max_chunks=request.max_context_chunks,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Early exit: no context (zero-context short-circuit)
        if not context_chunks:
            return self._insufficient_context_response(model, elapsed_ms)

        # 3. Apply token budget
        chat_history = request.chat_history or []
        context_str = self._build_context_prompt(context_chunks)
        history_str = (
            "\n".join(f"{m.role}: {m.content}" for m in chat_history)
            if chat_history
            else "No previous conversation."
        )

        system_prompt = RAG_SYSTEM_PROMPT.format(
            context=context_str,
            chat_history=history_str,
        )

        token_budget = TokenBudget(model, self.settings)

        def count_fn(text: str) -> int:
            # Synchronous fallback - use char estimate
            return len(text) // 4

        try:
            _, truncated_history, final_chunks = token_budget.allocate(
                system_prompt=system_prompt,
                chat_history=chat_history,
                context_chunks=context_chunks,
                count_fn=count_fn,
            )
        except TokenBudgetExceededError:
            raise

        # 4-5. Generate response
        try:
            llm = ChatOllama(
                base_url=self.ollama_url,
                model=model,
                temperature=self.settings.rag_temperature,
                timeout=self.settings.rag_timeout_seconds,
            )

            messages = [
                ("system", system_prompt),
                ("human", request.query),
            ]

            response = await llm.ainvoke(messages)
            answer = response.content

        except Exception as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"LLM response timed out: {e}") from e
            raise RAGServiceError(f"LLM generation failed: {e}") from e

        # 6. Extract citations
        citations = self._extract_citations(answer, final_chunks)
        grounded = len(citations) > 0

        # 7. Calculate hybrid confidence
        # Note: llm_score=50 placeholder until LLM self-assessment is integrated
        context_for_coverage = "\n".join(c.chunk_text for c in final_chunks)
        confidence = self._calculate_confidence(
            retrieval_results=final_chunks,
            answer=answer,
            context=context_for_coverage,
            llm_score=50,  # TODO: Integrate CONFIDENCE_PROMPT call
        )

        # 8. Determine action based on confidence threshold
        if confidence.overall >= self.confidence_threshold:
            action: Literal["CITE", "ROUTE"] = "CITE"
            route_to = None
        else:
            action = "ROUTE"
            route_to = await self.get_route_target(final_chunks, db)

        # Calculate context tokens used
        context_tokens_used = sum(len(c.chunk_text) // 4 for c in final_chunks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=answer,
            confidence=confidence,
            citations=citations,
            action=action,
            route_to=route_to,
            model_used=model,
            context_chunks_used=len(final_chunks),
            context_tokens_used=context_tokens_used,
            generation_time_ms=elapsed_ms,
            grounded=grounded,
        )

    async def health_check(self) -> dict:
        """Check Ollama connectivity and model availability.

        Returns:
            Dict with health status information
        """
        result = {
            "ollama_healthy": False,
            "ollama_latency_ms": None,
            "default_model_available": False,
            "available_models": [],
        }

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                response.raise_for_status()

            result["ollama_latency_ms"] = (time.perf_counter() - start) * 1000
            result["ollama_healthy"] = True

            models_data = response.json().get("models", [])
            result["available_models"] = [m["name"] for m in models_data]

            # Check if default model is available
            available_base = [m["name"].split(":")[0] for m in models_data]
            result["default_model_available"] = (
                self.default_model in available_base
                or self.default_model in result["available_models"]
            )

        except Exception as e:
            logger.warning(f"RAG health check failed: {e}")

        return result
