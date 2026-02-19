"""RAG Service for retrieval-augmented generation.

Orchestrates: retrieve -> prompt -> generate -> evaluate.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
from ai_ready_rag.services.cache_service import CacheEntry, CacheService
from ai_ready_rag.services.factory import get_vector_service
from ai_ready_rag.services.rag_constants import (
    MODEL_LIMITS,
    ROUTER_PROMPT,
    ROUTING_DIRECT,
    ROUTING_RETRIEVE,
    STOPWORDS,
)
from ai_ready_rag.services.settings_service import get_rag_setting
from ai_ready_rag.services.vector_service import SearchResult

if TYPE_CHECKING:
    from ai_ready_rag.services.protocols import VectorServiceProtocol

logger = logging.getLogger(__name__)

# SourceId extraction pattern: [SourceId: {uuid}:{index}]
# Accept [SourceId: uuid:chunk] or (SourceId: uuid:chunk) - LLMs sometimes use parens
SOURCEID_PATTERN = r"[\[\(]SourceId:\s*([a-f0-9-]{36}:\d+)[\]\)]"
# Fallback: LLM sometimes omits chunk index, writing just the UUID
SOURCEID_PATTERN_DOCONLY = r"[\[\(]SourceId:\s*([a-f0-9-]{36})[\]\)]"


# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. /no_think
You answer questions using ONLY the provided context documents. Never use outside knowledge.
Every factual statement MUST include a citation copied exactly from the context block header.
Example: "The policy limit is $1,000,000 [SourceId: a1b2c3d4-e5f6-7890-abcd-ef1234567890:5]"
If the context does not contain the answer, respond: NOT FOUND IN PROVIDED DOCUMENTS"""

RAG_USER_TEMPLATE = """CONTEXT DOCUMENTS:
{context}

PREVIOUS CONVERSATION:
{chat_history}

Based ONLY on the context documents above, answer this question. Include [SourceId: ...] citations for every fact.

Question: {query}"""

CONFIDENCE_PROMPT = """Rate how well this answer is grounded in the context. /no_think

CONTEXT:
{context_summary}

QUESTION: {query}

ANSWER: {answer}

Score 0-100. If the answer references specific facts, names, or numbers that appear in the context, score 70+. Only score below 30 if the answer is completely fabricated with no basis in the context.

Respond with ONLY a number."""


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
    user_tags: list[str] | None  # None = no tag filtering (system admin bypass)
    tenant_id: str = "default"
    chat_history: list[ChatMessage] | None = None
    model: str | None = None
    max_context_chunks: int | None = None  # None = use database setting
    is_warming: bool = False  # True when called from warming worker (skip access tracking)


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
    routing_decision: Literal["RETRIEVE", "DIRECT"] | None = None  # Query routing result


@dataclass
class EvalRAGResult:
    """RAG output enriched with retrieval details for evaluation."""

    response: RAGResponse
    retrieved_contexts: list[str]  # chunk_text from each SearchResult
    retrieval_scores: list[float]  # score from each SearchResult


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


@dataclass
class QueryIntent:
    """Detected intent from user query for tag-based boosting."""

    preferred_tags: list[str]  # Tags to boost e.g. ["stage:policy"]
    intent_label: str | None  # For logging e.g. "active_policy"
    confidence: float  # 0.0-1.0
    forms_eligible: bool = False  # True when intent matches structured form data
    intent_labels: list[str] | None = None  # All matched labels e.g. ["active_policy", "gl"]


# Insurance domain intent patterns: (compiled_regex, preferred_tags, label)
# Checked in order; multiple can match and accumulate tags.
INTENT_PATTERNS: list[tuple[re.Pattern, list[str], str]] = [
    # Stage patterns
    (
        re.compile(
            r"(?:current|active|bound|in[- ]?force)\b.*?"
            r"(?:polic|coverage|limit|deductible|premium)",
            re.IGNORECASE,
        ),
        ["stage:policy", "doctype:policy"],
        "active_policy",
    ),
    (
        re.compile(
            r"(?:policy)\s+(?:limit|deductible|premium|number|period|effective)",
            re.IGNORECASE,
        ),
        ["stage:policy", "doctype:policy"],
        "policy_detail",
    ),
    (
        re.compile(r"\b(?:quote[ds]?|proposal[s]?|proposed)\b", re.IGNORECASE),
        ["stage:quote", "doctype:quote"],
        "quote",
    ),
    (
        re.compile(r"\b(?:bind|binder|bound|binding)\b", re.IGNORECASE),
        ["stage:bind"],
        "bind",
    ),
    (
        re.compile(r"\b(?:submit|submission|application)\b", re.IGNORECASE),
        ["stage:submission"],
        "submission",
    ),
    (
        re.compile(r"\b(?:renewal|renew)\b", re.IGNORECASE),
        ["stage:quote"],
        "renewal",
    ),
    # Doctype patterns
    (
        re.compile(r"\b(?:endorsement|rider|amendment)\b", re.IGNORECASE),
        ["doctype:endorsement"],
        "endorsement",
    ),
    (
        re.compile(r"\b(?:certificate|cert|coi)\b", re.IGNORECASE),
        ["doctype:certificate"],
        "certificate",
    ),
    (
        re.compile(r"(?:loss\s+run|claims\s+history)", re.IGNORECASE),
        ["doctype:loss_run"],
        "loss_run",
    ),
    (
        re.compile(r"(?:declaration|dec\s+page)", re.IGNORECASE),
        ["doctype:coverage_summary"],
        "dec_page",
    ),
    # Topic patterns
    (
        re.compile(r"\b(?:earthquake|quake|seismic)\b", re.IGNORECASE),
        ["topic:earthquake"],
        "earthquake",
    ),
    (
        re.compile(r"\b(?:property|building|structure)\b", re.IGNORECASE),
        ["topic:property"],
        "property",
    ),
    (
        re.compile(r"(?:general\s+liab|(?<!\w)gl(?!\w)|commercial\s+general)", re.IGNORECASE),
        ["topic:gl"],
        "gl",
    ),
    (
        re.compile(r"(?:workers?\s+comp|(?<!\w)wc(?!\w))", re.IGNORECASE),
        ["topic:wc"],
        "workers_comp",
    ),
    (
        re.compile(r"(?:d\s*&\s*o|directors\s+and\s+officers)", re.IGNORECASE),
        ["topic:do"],
        "do",
    ),
    (
        re.compile(r"(?:crime|fidelity|employee\s+dishonesty)", re.IGNORECASE),
        ["topic:crime"],
        "crime",
    ),
    (
        re.compile(r"(?:epli|employment\s+practices)", re.IGNORECASE),
        ["topic:epli"],
        "epli",
    ),
    (
        re.compile(r"\b(?:umbrella|excess)\b", re.IGNORECASE),
        ["topic:umbrella"],
        "umbrella",
    ),
]


def detect_query_intent(query: str) -> QueryIntent:
    """Detect intent from user query using regex patterns.

    Scans query against INTENT_PATTERNS. Multiple patterns can match,
    accumulating preferred_tags (deduplicated). Fast regex-only, no LLM.

    Args:
        query: User's natural language query.

    Returns:
        QueryIntent with accumulated preferred_tags and primary label.
    """
    preferred_tags: list[str] = []
    labels: list[str] = []
    seen_tags: set[str] = set()

    for pattern, tags, label in INTENT_PATTERNS:
        if pattern.search(query):
            for tag in tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    preferred_tags.append(tag)
            labels.append(label)

    from ai_ready_rag.services.forms_query_service import FORMS_ELIGIBLE_INTENTS

    forms_eligible = any(label in FORMS_ELIGIBLE_INTENTS for label in labels)

    return QueryIntent(
        preferred_tags=preferred_tags,
        intent_label=labels[0] if labels else None,
        confidence=min(1.0, len(labels) * 0.5) if labels else 0.0,
        forms_eligible=forms_eligible,
        intent_labels=labels if labels else None,
    )


_TAG_CACHE: dict[str, list[tuple[str, str, re.Pattern]]] = {}
_TAG_CACHE_TIME: float = 0.0
TAG_CACHE_TTL = 300  # 5 minutes


def _get_tag_patterns(db: Session) -> list[tuple[str, str, re.Pattern]]:
    """Load tags from DB and build regex patterns. Cached 5 min."""
    global _TAG_CACHE_TIME
    now = time.time()
    cache_key = "all"
    if cache_key in _TAG_CACHE and (now - _TAG_CACHE_TIME) < TAG_CACHE_TTL:
        return _TAG_CACHE[cache_key]

    from ai_ready_rag.db.models.user import Tag

    tags = db.query(Tag).all()

    patterns: list[tuple[str, str, re.Pattern]] = []
    for tag in tags:
        namespace, _, value = tag.name.partition(":")

        match_terms: set[str] = set()
        if tag.display_name:
            match_terms.add(tag.display_name.lower())
        if value:
            match_terms.add(value.replace("-", " ").replace("_", " ").lower())

        # For year tags like "2025-2026", also match individual years
        if namespace == "year" and "-" in value:
            for part in value.split("-"):
                if len(part) == 4 and part.isdigit():
                    match_terms.add(part)

        for term in match_terms:
            if len(term) < 2:
                continue
            escaped = re.escape(term)
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            patterns.append((tag.name, namespace, pattern))

    _TAG_CACHE[cache_key] = patterns
    _TAG_CACHE_TIME = now
    return patterns


def enrich_intent_with_tags(intent: QueryIntent, query: str, db: Session) -> QueryIntent:
    """Add preferred_tags by matching query text against DB tags.

    Scans all tags' display_name/value against query using word-boundary
    regex. Matched tags are added to intent.preferred_tags (deduped).
    Does not remove existing preferred_tags from regex detection.
    """
    patterns = _get_tag_patterns(db)
    seen = set(intent.preferred_tags)

    for tag_name, _namespace, pattern in patterns:
        if tag_name in seen:
            continue
        if pattern.search(query):
            intent.preferred_tags.append(tag_name)
            seen.add(tag_name)

    return intent


def extract_key_terms(text: str) -> set[str]:
    """Extract significant terms from text.

    Method:
    1. Lowercase and extract alphanumeric words
    2. Filter stopwords
    3. Filter short words (< 3 chars)
    """
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) >= 3}


def expand_query(question: str, db: Session | None = None) -> list[str]:
    """Expand query with related search terms for better recall.

    Business domain-specific expansions based on common query patterns.
    Now includes database-managed synonyms with word-boundary matching.

    Args:
        question: Original user question
        db: Optional database session for synonym lookup

    Returns:
        List of query variants (original + expansions)
    """
    import json

    queries = [question]
    query_tokens = tokenize_query(question)

    # 1. Check database synonyms (if db provided) - BIDIRECTIONAL
    if db is not None:
        synonyms = get_cached_synonyms(db)
        for syn in synonyms:
            syn_list = json.loads(syn.synonyms)

            # Forward match: term in query -> add synonyms
            if matches_keyword(syn.term, query_tokens):
                queries.extend(syn_list)
                logger.debug(f"Synonym expansion (forward): '{syn.term}' -> {syn_list}")

            # Reverse match: any synonym in query -> add term + other synonyms
            else:
                for synonym in syn_list:
                    if matches_keyword(synonym, query_tokens):
                        # Add the term
                        queries.append(syn.term)
                        # Add other synonyms (excluding the matched one)
                        other_synonyms = [s for s in syn_list if s != synonym]
                        queries.extend(other_synonyms)
                        logger.debug(
                            f"Synonym expansion (reverse): '{synonym}' -> "
                            f"['{syn.term}'] + {other_synonyms}"
                        )
                        break  # Only expand once per synonym group

    # 2. Existing hardcoded patterns (fallback)
    q_lower = question.lower()

    # Customer/client queries
    if "customer" in q_lower or "client" in q_lower:
        queries.append("customer company revenue sales account")

    # Ranking queries
    if any(w in q_lower for w in ["top", "best", "biggest", "largest", "highest"]):
        queries.append("highest revenue largest sales top ranking")

    # Employee queries
    if any(w in q_lower for w in ["employee", "staff", "who works", "team"]):
        queries.append("employee directory staff name department role")

    # Financial queries
    if any(w in q_lower for w in ["budget", "financial", "cost", "expense", "revenue"]):
        queries.append("budget expense revenue financial quarterly annual")

    # Vendor/supplier queries
    if "vendor" in q_lower or "supplier" in q_lower:
        queries.append("vendor supplier company contact procurement")

    # Inventory/product queries
    if any(w in q_lower for w in ["inventory", "stock", "product", "sku"]):
        queries.append("inventory product SKU quantity stock warehouse")

    # Policy/procedure queries
    if "policy" in q_lower or "procedure" in q_lower:
        queries.append("policy procedure guideline rule compliance")

    # Data/table queries
    if any(w in q_lower for w in ["table", "data", "spreadsheet", "report"]):
        queries.append("table data row column statistics report")

    return queries


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
# Synonym Cache and Helper Functions
# -----------------------------------------------------------------------------

# Module-level cache for synonym data
_synonym_cache: list | None = None
_synonym_cache_version: int = 0
_synonym_cache_time: datetime | None = None


def invalidate_synonym_cache() -> None:
    """Invalidate synonym cache (call on CRUD operations)."""
    global _synonym_cache, _synonym_cache_version, _synonym_cache_time
    _synonym_cache = None
    _synonym_cache_time = None
    _synonym_cache_version += 1
    logger.debug(f"Synonym cache invalidated (version: {_synonym_cache_version})")


def get_cached_synonyms(db: Session) -> list:
    """Get synonyms with in-memory caching (60s TTL + explicit invalidation).

    Args:
        db: Database session for querying synonyms

    Returns:
        List of enabled QuerySynonym records
    """
    global _synonym_cache, _synonym_cache_time
    from ai_ready_rag.db.models import QuerySynonym

    now = datetime.utcnow()
    cache_expired = (
        _synonym_cache is None
        or _synonym_cache_time is None
        or (now - _synonym_cache_time).total_seconds() > 60
    )

    if cache_expired:
        _synonym_cache = db.query(QuerySynonym).filter(QuerySynonym.enabled.is_(True)).all()
        _synonym_cache_time = now
        logger.debug(f"Synonym cache refreshed: {len(_synonym_cache)} entries")

    return _synonym_cache or []


# Module-level cache for curated Q&A data
_qa_cache: list | None = None
_qa_cache_version: int = 0
_qa_token_index: dict[str, set[str]] | None = None
_qa_lookup: dict | None = None
_qa_keywords_by_id: dict[str, list[str]] | None = None
_qa_cache_time: datetime | None = None


def invalidate_qa_cache() -> None:
    """Invalidate curated Q&A cache (call on CRUD operations)."""
    global \
        _qa_cache, \
        _qa_cache_version, \
        _qa_token_index, \
        _qa_lookup, \
        _qa_keywords_by_id, \
        _qa_cache_time
    _qa_cache = None
    _qa_token_index = None
    _qa_lookup = None
    _qa_keywords_by_id = None
    _qa_cache_time = None
    _qa_cache_version += 1
    logger.debug(f"Q&A cache invalidated (version: {_qa_cache_version})")


def get_cached_qa(db: Session) -> tuple[dict[str, set[str]], dict, dict[str, list[str]]]:
    """Get Q&A with in-memory caching (60s TTL + explicit invalidation).

    Returns:
        token_index: Maps single token -> set of candidate qa_ids
        qa_lookup: Maps qa_id -> CuratedQA object (ordered by priority)
        keywords_by_id: Maps qa_id -> list of original keywords for verification
    """
    global _qa_token_index, _qa_lookup, _qa_keywords_by_id, _qa_cache_time
    import json

    from ai_ready_rag.db.models import CuratedQA

    now = datetime.utcnow()
    cache_expired = (
        _qa_token_index is None
        or _qa_cache_time is None
        or (now - _qa_cache_time).total_seconds() > 60
    )

    if cache_expired:
        # Load all enabled Q&A pairs ordered by priority
        qa_pairs = (
            db.query(CuratedQA)
            .filter(CuratedQA.enabled.is_(True))
            .order_by(CuratedQA.priority.desc())
            .all()
        )

        # Build token index and keyword lookup
        _qa_token_index = {}
        _qa_lookup = {}
        _qa_keywords_by_id = {}

        for qa in qa_pairs:
            _qa_lookup[qa.id] = qa
            keywords = json.loads(qa.keywords)
            _qa_keywords_by_id[qa.id] = keywords

            # Index each token from each keyword for fast candidate lookup
            for kw in keywords:
                for token in tokenize_query(kw):
                    if token not in _qa_token_index:
                        _qa_token_index[token] = set()
                    _qa_token_index[token].add(qa.id)

        _qa_cache_time = now
        logger.debug(
            f"Q&A cache refreshed: {len(_qa_lookup)} entries, {len(_qa_token_index)} tokens indexed"
        )

    return _qa_token_index or {}, _qa_lookup or {}, _qa_keywords_by_id or {}


def check_curated_qa(query: str, db: Session) -> str | None:
    """Check if query matches any curated Q&A pair by keyword.

    Uses two-phase matching:
    1. Token index lookup for candidate Q&A pairs (fast filter)
    2. Full keyword verification using matches_keyword() (accurate match)

    This handles multi-word keywords like "paid time off" correctly.

    Args:
        query: User's query string
        db: Database session

    Returns:
        Matching qa_id string (highest priority), or None to proceed with RAG
    """
    token_index, qa_lookup, keywords_by_id = get_cached_qa(db)
    query_tokens = tokenize_query(query)

    # Phase 1: Find candidate Q&A IDs (any token match)
    candidate_qa_ids: set[str] = set()
    for token in query_tokens:
        if token in token_index:
            candidate_qa_ids.update(token_index[token])

    if not candidate_qa_ids:
        return None

    # Phase 2: Verify full keyword match for each candidate
    # Iterate in priority order (qa_lookup preserves insertion order from query)
    for qa_id in qa_lookup:
        if qa_id not in candidate_qa_ids:
            continue

        # Check if ANY keyword fully matches the query
        keywords = keywords_by_id.get(qa_id, [])
        for keyword in keywords:
            if matches_keyword(keyword, query_tokens):
                logger.info(f"Curated Q&A match: '{keyword}' -> qa_id={qa_id}")
                return qa_id

    return None


def tokenize_query(query: str) -> set[str]:
    """Extract normalized word tokens from query.

    Extracts alphanumeric words, converts to lowercase, and filters
    tokens shorter than 2 characters.

    Args:
        query: Input text to tokenize

    Returns:
        Set of normalized word tokens
    """
    words = re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
    return {w for w in words if len(w) >= 2}


def matches_keyword(keyword: str, query_tokens: set[str]) -> bool:
    """Check if keyword matches query tokens using word-boundary matching.

    For multi-word keywords, all tokens must be present in query_tokens.
    This prevents false positives like "pto" matching "photo".

    Args:
        keyword: The keyword/phrase to match (e.g., "pto" or "paid time off")
        query_tokens: Set of tokens from the user query

    Returns:
        True if all keyword tokens are present in query_tokens
    """
    keyword_tokens = tokenize_query(keyword)
    return keyword_tokens.issubset(query_tokens)


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
        vector_service: VectorServiceProtocol | None = None,
        cache_service: CacheService | None = None,
        default_model: str | None = None,
    ):
        """Initialize RAG service.

        Args:
            settings: Application settings (used for non-tunable config like URLs)
            vector_service: VectorService instance (uses factory if None for backward compat)
            cache_service: CacheService instance (lazy-created if None)
            default_model: Optional default model override
        """
        self.settings = settings
        self._vector_service = vector_service
        self.ollama_url = settings.ollama_base_url
        self.default_model = default_model or settings.chat_model
        # Non-tunable settings from config (not exposed in UI)
        self.max_chunks_per_doc = settings.rag_max_chunks_per_doc
        self.chunk_overlap_threshold = settings.rag_chunk_overlap_threshold
        self.dedup_candidates_cap = settings.rag_dedup_candidates_cap
        self.enable_hallucination_check = settings.rag_enable_hallucination_check
        # Cache service (explicit or lazy-loaded for backward compat)
        self._cache_service: CacheService | None = cache_service
        # Live evaluation queue (injected post-construction from main.py)
        self._live_eval_queue = None
        # Forms query service (lazy — only if forms DB exists)
        self._forms_query_service = None
        forms_db = getattr(settings, "forms_db_path", None)
        if forms_db and Path(forms_db).exists():
            from ai_ready_rag.services.forms_query_service import FormsQueryService

            self._forms_query_service = FormsQueryService(settings)

    # -------------------------------------------------------------------------
    # Dynamic settings from database (single source of truth)
    # -------------------------------------------------------------------------

    @property
    def min_similarity_score(self) -> float:
        """Get min similarity score from database."""
        return get_rag_setting("retrieval_min_score", self.settings.rag_min_similarity_score)

    @property
    def confidence_threshold(self) -> int:
        """Get confidence threshold from database."""
        return get_rag_setting("llm_confidence_threshold", self.settings.rag_confidence_threshold)

    @property
    def enable_query_expansion(self) -> bool:
        """Get query expansion setting from database."""
        return get_rag_setting(
            "retrieval_enable_expansion", self.settings.rag_enable_query_expansion
        )

    @property
    def rag_temperature(self) -> float:
        """Get LLM temperature from database."""
        return get_rag_setting("llm_temperature", self.settings.rag_temperature)

    @property
    def rag_max_response_tokens(self) -> int:
        """Get max response tokens from database, capped to model limit."""
        requested = get_rag_setting(
            "llm_max_response_tokens", self.settings.rag_max_response_tokens
        )
        return self._get_effective_max_tokens(requested)

    def _get_effective_max_tokens(self, requested: int) -> int:
        """Cap requested tokens to current model's limit.

        Args:
            requested: User-requested max tokens

        Returns:
            Effective max tokens (capped to model limit if necessary)
        """
        model = self._get_current_chat_model()
        model_limit = MODEL_LIMITS.get(model, {}).get("max_response")

        # Fallback for unknown models: use a safe default
        if model_limit is None:
            model_limit = 2048
            logger.debug(f"Unknown model '{model}', using default limit {model_limit}")

        if requested > model_limit:
            logger.warning(
                f"Requested {requested} tokens exceeds {model} limit ({model_limit}). "
                f"Capping to {model_limit}."
            )
            return model_limit
        return requested

    def _get_current_chat_model(self) -> str:
        """Get current chat model from database settings or default.

        Returns:
            Current chat model name
        """
        from ai_ready_rag.db.database import SessionLocal
        from ai_ready_rag.services.settings_service import SettingsService

        db = SessionLocal()
        try:
            service = SettingsService(db)
            model = service.get("chat_model")
            return model if model else self.default_model
        finally:
            db.close()

    @property
    def retrieval_top_k(self) -> int:
        """Get top-k retrieval count from database."""
        return get_rag_setting("retrieval_top_k", self.settings.rag_total_context_chunks)

    def _get_live_sample_rate(self) -> float:
        """Get live evaluation sample rate from AdminSetting."""
        rate = get_rag_setting("eval_live_sample_rate", 0.0)
        try:
            return max(0.0, min(1.0, float(rate)))
        except (TypeError, ValueError):
            return 0.0

    def _maybe_queue_for_evaluation(
        self,
        request: RAGRequest,
        response: RAGResponse,
        final_chunks: list,
    ) -> None:
        """Probabilistically enqueue a live query for evaluation."""
        import random

        if request.is_warming:
            return
        if self._live_eval_queue is None:
            return

        sample_rate = self._get_live_sample_rate()
        if sample_rate <= 0:
            return
        if random.random() > sample_rate:
            return

        contexts = [c.chunk_text for c in final_chunks] if final_chunks else []
        self._live_eval_queue.enqueue(
            query=request.query,
            answer=response.answer,
            contexts=contexts,
            model_used=response.model,
            generation_time_ms=response.generation_time_ms,
        )

    @property
    def vector_service(self):
        """Get vector service, creating via factory if needed."""
        if self._vector_service is None:
            self._vector_service = get_vector_service(self.settings)
        return self._vector_service

    @property
    def cache(self) -> CacheService | None:
        """Get cache service, creating lazily if needed."""
        if self._cache_service is None:
            from ai_ready_rag.db.database import SessionLocal

            # Pass session factory, not a session instance
            # CacheService will create fresh sessions for each operation
            self._cache_service = CacheService(db_factory=SessionLocal)
        return self._cache_service

    def _cache_entry_to_response(self, entry: CacheEntry, elapsed_ms: float) -> RAGResponse:
        """Convert CacheEntry to RAGResponse.

        Args:
            entry: Cached response entry
            elapsed_ms: Time elapsed for cache lookup

        Returns:
            RAGResponse constructed from cached data
        """
        citations = [
            Citation(
                source_id=src.get("document_id", "") + ":0",
                document_id=src.get("document_id", ""),
                document_name=src.get("document_name", ""),
                chunk_index=0,
                page_number=src.get("page_number"),
                section=src.get("section"),
                relevance_score=1.0,  # Cached, no new retrieval
                snippet=src.get("excerpt", "")[:200],
                snippet_full=src.get("excerpt", "")[:1000],
            )
            for src in entry.sources
        ]

        return RAGResponse(
            answer=entry.answer,
            confidence=ConfidenceScore(
                overall=entry.confidence_overall,
                retrieval_score=entry.confidence_retrieval,
                coverage_score=entry.confidence_coverage,
                llm_score=entry.confidence_llm,
            ),
            citations=citations,
            action="CITE" if entry.confidence_overall >= self.confidence_threshold else "ROUTE",
            route_to=None,
            model_used=entry.model_used,
            context_chunks_used=len(citations),
            context_tokens_used=0,  # Not tracked in cache
            generation_time_ms=elapsed_ms,
            grounded=len(citations) > 0,
            routing_decision=None,  # Not tracked in cache
        )

    async def _get_or_embed_query(self, query: str) -> list[float]:
        """Get embedding from cache or compute new one.

        Args:
            query: The query text

        Returns:
            Query embedding vector
        """
        if self.cache:
            cached_embedding = await self.cache.get_embedding(query)
            if cached_embedding:
                return cached_embedding

        # Compute embedding via vector service
        return await self.vector_service.embed(query)

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
        user_tags: list[str] | None,
        tenant_id: str,
        max_chunks: int = 8,
    ) -> list[SearchResult]:
        """Retrieve context with quality controls and query expansion.

        Pipeline:
        1. Expand query into variants (if enabled)
        2. Retrieve candidates for each variant
        3. Interleave results for diversity
        4. Filter by minimum similarity score (0.3 on Qdrant 0-1 scale)
        5. Deduplicate near-identical chunks (Jaccard > 0.9)
        6. Limit chunks per document (max 3)
        7. Return top-k by score

        Args:
            query: User's question
            user_tags: User's access tags. None = no tag filtering
                (system admin bypass), empty list = public only.
            tenant_id: Tenant identifier
            max_chunks: Maximum chunks to return

        Returns:
            List of quality-filtered search results
        """
        # 0. Detect intent (fast regex, no LLM)
        intent = detect_query_intent(query)

        # 1. Expand query if enabled + forms lookup (share DB session)
        from ai_ready_rag.db.database import SessionLocal

        forms_results: list[SearchResult] = []
        db_session = SessionLocal()
        try:
            # Enrich intent with dynamic tag matching from DB
            intent = enrich_intent_with_tags(intent, query, db_session)
            if intent.preferred_tags:
                logger.info(f"Intent: {intent.intent_label} -> {intent.preferred_tags}")

            if self.enable_query_expansion:
                queries = expand_query(query, db=db_session)
                logger.debug(f"Query expansion: {len(queries)} variants")
            else:
                queries = [query]

            # 1.5 Query forms DB for structured context
            if intent.forms_eligible and self._forms_query_service:
                try:
                    forms_results = self._forms_query_service.query_forms(
                        intent=intent,
                        user_tags=user_tags,
                        db=db_session,
                        max_results=min(4, max_chunks // 2),
                    )
                    if forms_results:
                        logger.info(
                            f"Forms query: {len(forms_results)} results "
                            f"for intent={intent.intent_label}"
                        )
                except Exception as e:
                    logger.warning(f"Forms query failed, continuing with vector only: {e}")
        finally:
            db_session.close()

        candidate_limit = min(max_chunks * 3, self.dedup_candidates_cap)

        # 2. Retrieve candidates for each query variant
        query_results: list[list[SearchResult]] = []
        for q in queries:
            candidates = await self.vector_service.search(
                query=q,
                user_tags=user_tags,
                tenant_id=tenant_id,
                limit=candidate_limit,
                score_threshold=0.0,  # Filter later for more control
                preferred_tags=intent.preferred_tags or None,
            )
            if candidates:
                query_results.append(candidates)

        # Early exit: no results from any query (unless forms provided results)
        if not query_results and not forms_results:
            return []

        # 3. Interleave results for diversity (Spark pattern)
        seen_content: set[int] = set()
        all_candidates: list[SearchResult] = []

        for i in range(candidate_limit):
            for results in query_results:
                if i < len(results):
                    chunk = results[i]
                    # Hash first 200 chars for dedup
                    content_hash = hash(chunk.chunk_text[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_candidates.append(chunk)

        # 3.1 Prepend forms results (structured data ranks above vector results)
        if forms_results:
            all_candidates = forms_results + all_candidates

        # 3.5 Apply recency boost (before score filter so boosted scores are filtered correctly)
        if self.recency_weight > 0:
            all_candidates = self._apply_recency_boost(all_candidates)

        # 3.6 Apply intent boost (after recency, before score filter)
        if self.intent_boost_weight > 0 and intent.preferred_tags:
            all_candidates = self._apply_intent_boost(all_candidates, intent)

        # 4. Filter by minimum similarity (Qdrant cosine is 0-1)
        filtered = [c for c in all_candidates if c.score >= self.min_similarity_score]

        # 5. Deduplicate (Jaccard similarity > 0.9)
        deduped = self._deduplicate_chunks(filtered)

        # 6. Limit per document
        limited = self._limit_per_document(deduped, max_per_doc=self.max_chunks_per_doc)

        # 7. Return top-k
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

    @property
    def recency_weight(self) -> float:
        """Get recency boost weight from settings."""
        return self.settings.rag_recency_weight

    @property
    def intent_boost_weight(self) -> float:
        """Get intent boost weight from database settings."""
        return get_rag_setting(
            "retrieval_intent_boost_weight", self.settings.rag_intent_boost_weight
        )

    def _apply_recency_boost(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Boost scores of newer documents using year tags + content dates.

        Args:
            chunks: List of search results to rerank.

        Returns:
            Reranked list with recency-boosted scores.
        """
        weight = self.recency_weight

        # Find the max year across all candidates to normalize
        max_year = 0
        chunk_years: list[int | None] = []

        for chunk in chunks:
            year = self._extract_year(chunk)
            chunk_years.append(year)
            if year and year > max_year:
                max_year = year

        if max_year == 0:
            return chunks  # No year data, skip reranking

        for chunk, year in zip(chunks, chunk_years, strict=True):
            if year is None:
                recency_score = 0.5  # Neutral
            else:
                # Newer = higher. Each year difference = 0.2 penalty
                years_behind = max_year - year
                recency_score = max(0.0, 1.0 - years_behind * 0.2)
            chunk.score = (1 - weight) * chunk.score + weight * recency_score

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def _extract_year(self, chunk: SearchResult) -> int | None:
        """Extract document year from tags (primary) or content (fallback).

        Args:
            chunk: Search result to extract year from.

        Returns:
            Year as integer, or None if not found.
        """
        # Primary: year tag (e.g., "year:2025-2026" → take end year)
        if chunk.tags:
            for tag in chunk.tags:
                if tag.startswith("year:"):
                    match = re.search(r"(\d{4})$", tag)
                    if match:
                        return int(match.group(1))

        # Fallback: scan chunk_text for years (first 500 chars)
        years = re.findall(r"\b(20[2-3]\d)\b", chunk.chunk_text[:500])
        if years:
            return max(int(y) for y in years)

        return None

    def _apply_intent_boost(
        self, chunks: list[SearchResult], intent: QueryIntent
    ) -> list[SearchResult]:
        """Boost scores of chunks whose tags match detected intent.

        Follows the same pattern as _apply_recency_boost. Blends original
        score with tag-overlap score using configurable weight.

        Args:
            chunks: List of search results to rerank.
            intent: Detected query intent with preferred_tags.

        Returns:
            Reranked list with intent-boosted scores.
        """
        weight = self.intent_boost_weight
        preferred_set = set(intent.preferred_tags)
        num_preferred = len(preferred_set)

        if num_preferred == 0:
            return chunks

        for chunk in chunks:
            chunk_tags = set(chunk.tags or [])
            overlap = len(chunk_tags & preferred_set) / num_preferred
            chunk.score = (1 - weight) * chunk.score + weight * overlap

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    async def evaluate_hallucination(
        self,
        query: str,
        answer: str,
        context_summary: str,
        model: str,
    ) -> int:
        """Evaluate answer for hallucinations using LLM self-assessment.

        Ported from Spark's working configuration.

        Args:
            query: Original user question
            answer: LLM-generated answer
            context_summary: Summary of context used
            model: Model to use for evaluation

        Returns:
            Score 0-100 (higher = better grounded)
        """
        if not self.enable_hallucination_check:
            return 50  # Default neutral score

        try:
            prompt = CONFIDENCE_PROMPT.format(
                context_summary=context_summary[:8000],  # Enough to cover most retrieved chunks
                query=query,
                answer=answer[:2000],  # Reasonable limit for eval
            )

            # Use raw Ollama API instead of ChatOllama to avoid
            # thinking-token stripping bug in langchain-ollama
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 2048},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                score_text = resp.json()["message"]["content"].strip()

            # Extract numeric score
            match = re.search(r"\d+", score_text)
            if not match:
                logger.warning(
                    f"[RAG] LLM self-assessment returned non-numeric: '{score_text[:100]}'"
                )
                return 50  # Fallback to neutral

            score = int(match.group())
            score = min(100, max(0, score))

            logger.info(
                f"[RAG] Hallucination check: score={score} | Raw response: '{score_text[:200]}'"
            )

            return score

        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return 50  # Fallback to neutral

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

        # Build lookup maps from chunks
        chunk_map: dict[str, SearchResult] = {}
        doc_map: dict[str, SearchResult] = {}  # doc_id -> first (highest-scored) chunk
        for chunk in chunks:
            sid = f"{chunk.document_id}:{chunk.chunk_index}"
            chunk_map[sid] = chunk
            if chunk.document_id not in doc_map:
                doc_map[chunk.document_id] = chunk

        # Fallback: if LLM used doc-only IDs (no :chunk), find them too
        if not source_ids:
            doc_only_ids = re.findall(SOURCEID_PATTERN_DOCONLY, answer)
            for doc_id in doc_only_ids:
                if doc_id in doc_map:
                    chunk = doc_map[doc_id]
                    source_ids.append(f"{doc_id}:{chunk.chunk_index}")

        citations: list[Citation] = []
        seen_ids: set[str] = set()
        seen_doc_ids: set[str] = set()

        for source_id in source_ids:
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)

            if source_id in chunk_map:
                chunk = chunk_map[source_id]
            else:
                # Fallback: LLM cited correct doc but wrong chunk index
                doc_id = source_id.split(":")[0] if ":" in source_id else source_id
                if doc_id in doc_map and doc_id not in seen_doc_ids:
                    chunk = doc_map[doc_id]
                    logger.info(f"Citation fallback: {source_id} -> {doc_id}:{chunk.chunk_index}")
                else:
                    logger.warning(f"Unknown SourceId in answer: {source_id}")
                    continue

            # Deduplicate by document to avoid multiple citations to same doc
            if chunk.document_id in seen_doc_ids:
                continue
            seen_doc_ids.add(chunk.document_id)

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

    async def run_router(self, query: str, model: str) -> str:
        """Classify query as RETRIEVE or DIRECT using LLM.

        Args:
            query: User's question
            model: Model to use for classification

        Returns:
            ROUTING_RETRIEVE or ROUTING_DIRECT
        """
        try:
            prompt = ROUTER_PROMPT.format(question=query)

            # Use raw Ollama API to avoid ChatOllama thinking-token bug
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 256},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                decision = resp.json()["message"]["content"].strip().upper()

            # Parse response - look for DIRECT or RETRIEVE
            if ROUTING_DIRECT in decision:
                logger.info(f"Router: DIRECT for query: {query[:50]}...")
                return ROUTING_DIRECT
            else:
                # Default to RETRIEVE when in doubt
                logger.info(f"Router: RETRIEVE for query: {query[:50]}...")
                return ROUTING_RETRIEVE

        except Exception as e:
            logger.warning(f"Router failed, defaulting to RETRIEVE: {e}")
            return ROUTING_RETRIEVE

    def _generate_direct_response(
        self,
        answer: str,
        model: str,
        elapsed_ms: float,
    ) -> RAGResponse:
        """Create response for DIRECT routing (no retrieval).

        Args:
            answer: LLM-generated answer
            model: Model name for response metadata
            elapsed_ms: Elapsed time in milliseconds

        Returns:
            RAGResponse with no citations and grounded=False
        """
        return RAGResponse(
            answer=answer,
            confidence=ConfidenceScore(
                overall=50,  # Neutral confidence for direct responses
                retrieval_score=0.0,
                coverage_score=0.0,
                llm_score=50,
            ),
            citations=[],
            action="CITE",  # CITE action but with no citations
            route_to=None,
            model_used=model,
            context_chunks_used=0,
            context_tokens_used=0,
            generation_time_ms=elapsed_ms,
            grounded=False,
            routing_decision=ROUTING_DIRECT,
        )

    def _insufficient_context_response(
        self,
        model: str,
        elapsed_ms: float,
        routing_decision: str | None = None,
    ) -> RAGResponse:
        """Create response for zero-context case.

        Args:
            model: Model name for response metadata
            elapsed_ms: Elapsed time in milliseconds
            routing_decision: Query routing decision if applicable

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
            routing_decision=routing_decision,
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
        from ai_ready_rag.services.settings_service import SettingsService

        # 1. Validate model
        model = request.model or self.default_model
        await self.validate_model(model)

        # 0. Check curated Q&A first (highest priority - admin-approved answers)
        matched_qa_id = check_curated_qa(request.query, db)
        if matched_qa_id:
            from ai_ready_rag.db.models import CuratedQA

            curated_qa = db.query(CuratedQA).filter(CuratedQA.id == matched_qa_id).first()
            if curated_qa:
                # Update access tracking (skip during warming to reduce DB writes)
                if not request.is_warming:
                    curated_qa.access_count += 1
                    curated_qa.last_accessed_at = datetime.utcnow()
                    db.commit()

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Build citation from source_reference
                citation = Citation(
                    source_id=f"curated:{curated_qa.id}",
                    document_id=curated_qa.id,
                    document_name=curated_qa.source_reference,
                    chunk_index=0,
                    page_number=None,
                    section=curated_qa.source_reference,
                    relevance_score=1.0,
                    snippet=curated_qa.source_reference,
                    snippet_full=curated_qa.source_reference,
                )

                return RAGResponse(
                    answer=curated_qa.answer,
                    confidence=ConfidenceScore(
                        overall=curated_qa.confidence,
                        retrieval_score=1.0,
                        coverage_score=1.0,
                        llm_score=curated_qa.confidence,
                    ),
                    citations=[citation],
                    action="CITE",
                    route_to=None,
                    model_used="curated",
                    context_chunks_used=0,
                    context_tokens_used=0,
                    generation_time_ms=elapsed_ms,
                    grounded=True,
                    routing_decision=None,
                )
            # else: Q&A deleted between cache hit and re-fetch; fall through to RAG

        # 1.25 Check cache first (if enabled)
        if self.cache and self.cache.enabled:
            try:
                # Get or compute embedding for semantic cache lookup
                # This enables semantic matching even for new query variations
                query_embedding = await self._get_or_embed_query(request.query)
                cached = await self.cache.get(
                    query=request.query,
                    user_tags=request.user_tags,
                    query_embedding=query_embedding,
                )
                if cached:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(f"Cache HIT for query: {request.query[:50]}...")
                    return self._cache_entry_to_response(cached, elapsed_ms)
            except Exception as e:
                logger.warning(f"Cache lookup failed, proceeding without cache: {e}")

        # 1.5 Run query router (if retrieve_and_direct mode enabled)
        routing_decision: str | None = None
        settings_service = SettingsService(db)
        query_routing_mode = settings_service.get("query_routing_mode") or "retrieve_only"

        if query_routing_mode == "retrieve_and_direct":
            routing_decision = await self.run_router(request.query, model)
            if routing_decision == ROUTING_DIRECT:
                # Skip retrieval, generate direct response
                try:
                    llm = ChatOllama(
                        base_url=self.ollama_url,
                        model=model,
                        temperature=self.rag_temperature,
                        timeout=self.settings.rag_timeout_seconds,
                    )
                    response = await llm.ainvoke([("human", request.query)])
                    answer = response.content
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    return self._generate_direct_response(answer, model, elapsed_ms)
                except Exception as e:
                    if "timeout" in str(e).lower():
                        raise LLMTimeoutError(f"LLM response timed out: {e}") from e
                    raise RAGServiceError(f"LLM generation failed: {e}") from e
        else:
            # Default mode: always retrieve (routing_decision stays None)
            routing_decision = ROUTING_RETRIEVE

        # Run core retrieval+generation pipeline
        response, final_chunks = await self._run_rag_pipeline(
            request, db, model, start_time, routing_decision
        )

        # 9. Store in cache before returning (if enabled and high enough confidence)
        if self.cache and self.cache.enabled:
            try:
                # Get or compute embedding for cache storage
                query_embedding = await self._get_or_embed_query(request.query)
                await self.cache.put(request.query, query_embedding, response)
                await self.cache.put_embedding(request.query, query_embedding)
            except Exception as e:
                logger.warning(f"[RAG] Failed to cache response: {e}")

        # 10. Maybe sample for live evaluation
        try:
            self._maybe_queue_for_evaluation(request, response, final_chunks)
        except Exception as e:
            logger.warning(f"[RAG] Failed to queue for live evaluation: {e}")

        return response

    async def generate_for_eval(self, request: RAGRequest, db: Session) -> EvalRAGResult:
        """Run RAG pipeline for evaluation, returning contexts alongside the response.

        Unlike generate(), this method:
        - Skips curated Q&A lookup
        - Skips cache lookup/store
        - Skips direct routing
        - Always retrieves context
        """
        start_time = time.perf_counter()
        model = request.model or self.default_model
        await self.validate_model(model)
        routing_decision = ROUTING_RETRIEVE

        response, final_chunks = await self._run_rag_pipeline(
            request, db, model, start_time, routing_decision
        )
        return EvalRAGResult(
            response=response,
            retrieved_contexts=[c.chunk_text for c in final_chunks],
            retrieval_scores=[c.score for c in final_chunks],
        )

    async def _run_rag_pipeline(
        self,
        request: RAGRequest,
        db: Session,
        model: str,
        start_time: float,
        routing_decision: str | None,
    ) -> tuple[RAGResponse, list[SearchResult]]:
        """Core retrieval+generation pipeline shared by generate() and generate_for_eval().

        Returns (RAGResponse, list[SearchResult]) so callers can access
        the raw search results for evaluation or caching.
        """
        # 2. Retrieve quality context (use DB setting if not specified in request)
        max_chunks = request.max_context_chunks or self.retrieval_top_k
        context_chunks = await self.get_quality_context(
            query=request.query,
            user_tags=request.user_tags,
            tenant_id=request.tenant_id,
            max_chunks=max_chunks,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Early exit: no context (zero-context short-circuit)
        if not context_chunks:
            return self._insufficient_context_response(model, elapsed_ms, routing_decision), []

        # 3. Apply token budget
        chat_history = request.chat_history or []
        system_prompt = RAG_SYSTEM_PROMPT

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

        # Build prompts AFTER token budget allocation (uses trimmed chunks)
        context_str = self._build_context_prompt(final_chunks)
        history_str = truncated_history if truncated_history else "No previous conversation."
        user_prompt = RAG_USER_TEMPLATE.format(
            context=context_str,
            chat_history=history_str,
            query=request.query,
        )

        # 4-5. Generate response
        logger.debug(
            f"[RAG] Query: {request.query[:100]} | "
            f"Context chunks: {len(final_chunks)} | "
            f"Context chars: {len(context_str)} | "
            f"History entries: {len(chat_history)}"
        )
        for i, c in enumerate(final_chunks):
            logger.debug(
                f"[RAG] Chunk {i}: doc={c.document_name} "
                f"sid={c.document_id}:{c.chunk_index} "
                f"score={c.score:.3f} tags={c.tags}"
            )

        try:
            llm = ChatOllama(
                base_url=self.ollama_url,
                model=model,
                temperature=self.rag_temperature,
                timeout=self.settings.rag_timeout_seconds,
            )

            messages = [
                ("system", system_prompt),
                ("human", user_prompt),
            ]

            response = await llm.ainvoke(messages)
            answer = response.content

            logger.debug(
                f"[RAG] LLM response length: {len(answer)} chars | First 200 chars: {answer[:200]}"
            )

        except Exception as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"LLM response timed out: {e}") from e
            raise RAGServiceError(f"LLM generation failed: {e}") from e

        # 6. Extract citations
        citations = self._extract_citations(answer, final_chunks)
        grounded = len(citations) > 0

        # 7. Calculate hybrid confidence with LLM self-assessment
        context_for_coverage = "\n".join(c.chunk_text for c in final_chunks)

        # Strip any leaked thinking artifacts from the answer before evaluation
        # qwen3 sometimes leaks thinking into visible content
        clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        # Evaluate hallucination (Spark feature)
        llm_score = await self.evaluate_hallucination(
            query=request.query,
            answer=clean_answer,
            context_summary=context_for_coverage,
            model=model,
        )

        confidence = self._calculate_confidence(
            retrieval_results=final_chunks,
            answer=answer,
            context=context_for_coverage,
            llm_score=llm_score,
        )

        # Citation guard: If context was provided but LLM cited nothing, cap confidence
        # and replace answer with standard message (don't show generic responses)
        if len(final_chunks) > 0 and len(citations) == 0:
            logger.warning(
                f"[RAG] No citations despite {len(final_chunks)} context chunks - "
                f"replacing generic answer and capping confidence to 30%"
            )
            answer = (
                "I found relevant documents but was unable to extract a specific answer. "
                "Please try rephrasing your question or contact your administrator for assistance."
            )
            confidence = ConfidenceScore(
                overall=min(confidence.overall, 30),
                retrieval_score=confidence.retrieval_score,
                coverage_score=confidence.coverage_score,
                llm_score=confidence.llm_score,
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

        rag_response = RAGResponse(
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
            routing_decision=routing_decision,
        )

        return rag_response, final_chunks

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
