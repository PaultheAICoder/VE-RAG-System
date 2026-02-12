"""Generate document summaries via Ollama for RAG indexing.

Produces a structured summary chunk for each document, enabling the retriever
to identify the correct document before pulling granular chunks.
"""

import logging
import re

import httpx

from ai_ready_rag.services.processing_service import ChunkInfo

logger = logging.getLogger(__name__)

# Timeout for Ollama summary generation (generous for large context)
SUMMARY_TIMEOUT_SECONDS = 120


class SummaryGenerator:
    """Generate document summaries via Ollama for RAG indexing."""

    def __init__(self, ollama_url: str, model: str = "qwen3:8b"):
        self.ollama_url = ollama_url
        self.model = model

    async def generate(
        self,
        chunks: list[ChunkInfo],
        filename: str,
        document_id: str,
    ) -> tuple[ChunkInfo, dict] | None:
        """Generate a summary chunk from sampled document chunks.

        Args:
            chunks: List of ChunkInfo objects from the document.
            filename: Original filename for context.
            document_id: Document identifier (for logging).

        Returns:
            Tuple of (ChunkInfo for summary, extra metadata dict) or None on failure.
        """
        if not chunks:
            logger.warning(f"No chunks to summarize for document {document_id}")
            return None

        sampled = self._sample_chunks(chunks)
        word_count = sum(len(c.text.split()) for c in chunks)
        prompt = self._build_prompt(filename, sampled, len(chunks), word_count)

        try:
            async with httpx.AsyncClient(timeout=SUMMARY_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                response_text = response.json().get("response", "").strip()

            if not response_text:
                logger.warning(f"Empty summary response for document {document_id}")
                return None

            parsed = self._parse_response(response_text)

            summary_text = f"DOCUMENT SUMMARY: {parsed['text']}"
            summary_chunk = ChunkInfo(
                text=summary_text,
                chunk_index=-1,
                page_number=None,
                section="Document Summary",
                token_count=len(summary_text) // 4,
            )
            summary_metadata = {
                "is_summary": True,
                "document_type": parsed["document_type"],
                "carrier": parsed["carrier"],
                "policy_period": parsed["policy_period"],
            }

            logger.info(
                f"Generated summary for {document_id}: "
                f"type={parsed['document_type']}, carrier={parsed['carrier']}"
            )
            return summary_chunk, summary_metadata

        except httpx.TimeoutException:
            logger.warning(f"Ollama timed out generating summary for {document_id}")
            return None
        except Exception as e:
            logger.warning(f"Summary generation failed for {document_id}: {e}")
            return None

    def _sample_chunks(self, chunks: list[ChunkInfo]) -> list[ChunkInfo]:
        """Sample up to 9 chunks: first 3, middle 3, last 3.

        For documents with <= 9 chunks, returns all chunks.
        """
        if len(chunks) <= 9:
            return list(chunks)

        first = chunks[:3]
        mid_start = len(chunks) // 2 - 1
        middle = chunks[mid_start : mid_start + 3]
        last = chunks[-3:]

        return first + middle + last

    def _build_prompt(
        self,
        filename: str,
        sampled_chunks: list[ChunkInfo],
        chunk_count: int,
        word_count: int,
    ) -> str:
        """Build the LLM prompt for summary generation."""
        chunk_texts = "\n\n".join(
            f"[Chunk {c.chunk_index}]: {c.text[:500]}" for c in sampled_chunks
        )

        return f"""You are summarizing a document for a retrieval-augmented generation (RAG) system.
Your summary will be embedded and used to match user questions to the right document.

Document: {filename}
Total chunks: {chunk_count}
Total words: {word_count:,}

Here are representative samples from the document:

{chunk_texts}

Produce the following (be precise, use exact numbers/names from the document):

DOCUMENT_TYPE: (one of: policy, quote, submission, loss_run, financial, sov, certificate, application, other)
CARRIER: (carrier or market name, or "unknown")
POLICY_PERIOD: (e.g., "12/01/2024-12/01/2025", or "unknown")
SUMMARY: (one paragraph, 100-200 words) Describe what this document contains, its key coverages/limits/deductibles if applicable, and what types of questions it could answer. Include specific numbers, names, and dates from the content."""

    def _parse_response(self, response_text: str) -> dict:
        """Parse structured fields from the LLM response.

        Returns:
            Dict with keys: text, document_type, carrier, policy_period.
            Falls back to full response as summary text if parsing fails.
        """
        result = {
            "text": "",
            "document_type": "other",
            "carrier": "unknown",
            "policy_period": "unknown",
        }

        # Extract DOCUMENT_TYPE
        dt_match = re.search(r"DOCUMENT_TYPE:\s*(.+)", response_text, re.IGNORECASE)
        if dt_match:
            dt_value = dt_match.group(1).strip().lower().rstrip(".")
            valid_types = {
                "policy",
                "quote",
                "submission",
                "loss_run",
                "financial",
                "sov",
                "certificate",
                "application",
                "other",
            }
            result["document_type"] = dt_value if dt_value in valid_types else "other"

        # Extract CARRIER
        carrier_match = re.search(r"CARRIER:\s*(.+)", response_text, re.IGNORECASE)
        if carrier_match:
            result["carrier"] = carrier_match.group(1).strip().rstrip(".")

        # Extract POLICY_PERIOD
        period_match = re.search(r"POLICY_PERIOD:\s*(.+)", response_text, re.IGNORECASE)
        if period_match:
            result["policy_period"] = period_match.group(1).strip().rstrip(".")

        # Extract SUMMARY
        summary_match = re.search(r"SUMMARY:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            result["text"] = summary_match.group(1).strip()
        else:
            # Fallback: use entire response as summary text
            result["text"] = response_text
            result["document_type"] = "other"

        return result
