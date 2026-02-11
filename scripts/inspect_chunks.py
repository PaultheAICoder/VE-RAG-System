#!/usr/bin/env python3
"""Inspect vector chunks for a document.

Shows what data was extracted and optionally asks Ollama for recommended questions.

Usage:
    # Summary view (default) — first 200 chars per chunk
    python scripts/inspect_chunks.py DOCUMENT_ID

    # Detail view — full chunk text
    python scripts/inspect_chunks.py DOCUMENT_ID --detail

    # Ask Ollama for recommended questions
    python scripts/inspect_chunks.py DOCUMENT_ID --recommend

    # Combine: full detail + recommendations
    python scripts/inspect_chunks.py DOCUMENT_ID --detail --recommend

    # Limit chunks shown
    python scripts/inspect_chunks.py DOCUMENT_ID --limit 10
"""

import argparse
import sys
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_ready_rag.config import get_settings


def get_chunks(qdrant_url: str, document_id: str) -> list[dict]:
    """Fetch all chunks for a document from Qdrant."""
    client = QdrantClient(url=qdrant_url)

    points, _ = client.scroll(
        collection_name="documents",
        scroll_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )

    chunks = []
    for p in points:
        payload = p.payload or {}
        chunks.append(
            {
                "index": payload.get("chunk_index", 0),
                "text": payload.get("chunk_text", ""),
                "page": payload.get("page_number"),
                "section": payload.get("section"),
                "tags": payload.get("tags", []),
                "document_name": payload.get("document_name", ""),
            }
        )

    return sorted(chunks, key=lambda c: c["index"])


def print_summary(chunks: list[dict], preview_chars: int = 200) -> None:
    """Print summary view — chunk index, word count, preview."""
    total_words = sum(len(c["text"].split()) for c in chunks)
    total_chars = sum(len(c["text"]) for c in chunks)
    doc_name = chunks[0]["document_name"] if chunks else "Unknown"
    tags = chunks[0]["tags"] if chunks else []

    print(f"Document:    {doc_name}")
    print(f"Tags:        {', '.join(tags) if tags else 'none'}")
    print(f"Chunks:      {len(chunks)}")
    print(f"Total words: {total_words:,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Avg words/chunk: {total_words // len(chunks) if chunks else 0}")
    print()

    # Chunk size distribution
    sizes = [len(c["text"]) for c in chunks]
    print(
        f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes) // 2]}"
    )
    print("=" * 80)

    for c in chunks:
        words = len(c["text"].split())
        page_str = f"p.{c['page']}" if c["page"] else ""
        section_str = c["section"] or ""
        meta = " | ".join(filter(None, [page_str, section_str]))
        meta_display = f" [{meta}]" if meta else ""

        preview = c["text"][:preview_chars].replace("\n", " ")
        if len(c["text"]) > preview_chars:
            preview += "..."

        print(f"\n  [{c['index']:3d}] ({words:4d} words){meta_display}")
        print(f"        {preview}")


def print_detail(chunks: list[dict], limit: int | None = None) -> None:
    """Print full chunk text."""
    display = chunks[:limit] if limit else chunks

    for c in display:
        words = len(c["text"].split())
        page_str = f"page={c['page']}" if c["page"] else ""
        section_str = f"section={c['section']}" if c["section"] else ""
        meta = " | ".join(filter(None, [page_str, section_str]))

        print(f"\n{'=' * 80}")
        print(f"Chunk {c['index']} | {words} words | {len(c['text'])} chars | {meta}")
        print("-" * 80)
        print(c["text"])

    if limit and limit < len(chunks):
        print(f"\n... showing {limit} of {len(chunks)} chunks (use --limit to adjust)")


def recommend_questions(chunks: list[dict], settings) -> None:
    """Ask Ollama to suggest questions based on the document content."""
    # Build a content summary from evenly-spaced chunk samples
    # Sample ~12 chunks to stay within context window and keep latency reasonable
    sample_size = min(12, len(chunks))
    step = max(1, len(chunks) // sample_size)
    sampled = [chunks[i] for i in range(0, len(chunks), step)][:sample_size]

    doc_name = chunks[0]["document_name"] if chunks else "Unknown"
    content_summary = "\n\n".join(f"[Chunk {c['index']}]: {c['text'][:300]}" for c in sampled)

    prompt = f"""You are analyzing a document that has been chunked for a RAG system.
Document: {doc_name}
Total chunks: {len(chunks)}
Total words: {sum(len(c["text"].split()) for c in chunks):,}

Here are representative samples from the document:

{content_summary}

Based on this content, suggest 10 specific questions that a user could ask this RAG system.
Focus on questions that:
1. Can be answered from the document content
2. Cover different sections/topics in the document
3. Range from simple factual to analytical
4. Would be useful for someone who needs information from this document

Format: Number each question on its own line. No other text."""

    print("\n" + "=" * 80)
    print("RECOMMENDED QUESTIONS (via Ollama)")
    print("=" * 80)

    ollama_url = settings.ollama_base_url
    model = settings.chat_model

    print(f"Model: {model}")
    print(f"Sampling {len(sampled)} of {len(chunks)} chunks...\n")

    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            response.raise_for_status()
            answer = response.json().get("response", "")
            print(answer.strip())
    except httpx.TimeoutException:
        print("ERROR: Ollama request timed out. Is the model loaded?")
        print(f"  Try: ollama pull {model}")
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to Ollama at {ollama_url}")
        print("  Is Ollama running? Try: ollama serve")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect vector chunks for a document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tip: Use scripts/list_documents.py to find document IDs",
    )
    parser.add_argument("document_id", help="Document ID (UUID)")
    parser.add_argument("--detail", action="store_true", help="Show full chunk text")
    parser.add_argument(
        "--recommend", action="store_true", help="Ask Ollama for recommended questions"
    )
    parser.add_argument("--limit", type=int, help="Limit number of chunks shown (detail mode)")
    parser.add_argument(
        "--preview", type=int, default=200, help="Preview chars per chunk in summary (default: 200)"
    )
    args = parser.parse_args()

    settings = get_settings()
    chunks = get_chunks(settings.qdrant_url, args.document_id)

    if not chunks:
        print(f"No chunks found for document {args.document_id}")
        print("Check the ID with: python scripts/list_documents.py --format ids")
        sys.exit(1)

    if args.detail:
        print_detail(chunks, limit=args.limit)
    else:
        print_summary(chunks, preview_chars=args.preview)

    if args.recommend:
        recommend_questions(chunks, settings)


if __name__ == "__main__":
    main()
