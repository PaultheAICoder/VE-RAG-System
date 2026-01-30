"""RAG Service constants.

Model limits and stopwords for coverage scoring.
"""

# Model context windows and response limits
# Used by TokenBudget for allocation decisions
MODEL_LIMITS: dict[str, dict[str, int]] = {
    "llama3.2": {"context_window": 8192, "max_response": 1024},
    "llama3.2:latest": {"context_window": 8192, "max_response": 1024},
    "qwen3:8b": {"context_window": 32768, "max_response": 2048},
    "deepseek-r1:32b": {"context_window": 65536, "max_response": 4096},
}

# Common English stopwords for coverage scoring
# Used in extract_key_terms() to filter non-significant words
STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
    }
)

# Query routing constants
ROUTING_DIRECT = "DIRECT"
ROUTING_RETRIEVE = "RETRIEVE"

# Router prompt for classifying queries (ported from Spark app.py)
ROUTER_PROMPT = """You are a routing agent. Analyze the user's question and decide if it requires retrieving information from documents.

Respond with ONLY one of these two words:
- RETRIEVE: if the question asks about company data, business information, customers, employees, financials, products, policies, procedures, or anything that would be in business documents. Questions containing "our", "we", "company" almost always need RETRIEVE.
- DIRECT: ONLY for general knowledge questions completely unrelated to the business (like "what is the capital of France")

When in doubt, choose RETRIEVE.

Question: {question}

Your routing decision (RETRIEVE or DIRECT):"""
