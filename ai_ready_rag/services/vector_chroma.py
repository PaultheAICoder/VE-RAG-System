"""Chroma vector service implementation for laptop development."""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_ready_rag.services.vector_service import SearchResult

logger = logging.getLogger(__name__)


class ChromaVectorService:
    """Lightweight vector service using Chroma for local development.

    Implements VectorServiceProtocol for laptop profile.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        embedding_dimension: int = 768,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.embedding_dimension = embedding_dimension

        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaVectorService initialized: {self.collection_name}")

    @property
    def collection(self) -> chromadb.Collection:
        """Get collection, initializing if needed."""
        if self._collection is None:
            raise RuntimeError("ChromaVectorService not initialized. Call initialize() first.")
        return self._collection

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embedding_model, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embedding_model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    async def add_document(
        self,
        document_id: str,
        document_name: str,
        chunks: list[str],
        tags: list[str],
        uploaded_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Index document chunks to Chroma."""
        if not chunks:
            return 0

        # Delete existing vectors for this document
        await self.delete_document(document_id)

        # Generate embeddings
        embeddings = await self._embed_batch(chunks)

        # Prepare data for Chroma
        ids = [f"{document_id}:{i}" for i in range(len(chunks))]
        metadatas = []
        for i, _chunk in enumerate(chunks):
            meta = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
                "tags": ",".join(tags),  # Chroma stores as string
                "uploaded_by": uploaded_by,
            }
            if metadata:
                meta.update({k: str(v) for k, v in metadata.items() if v is not None})
            metadatas.append(meta)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")
        return len(chunks)

    async def search(
        self,
        query: str,
        user_tags: list[str],
        tenant_id: str = "default",
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Search vectors filtered by user tags."""
        # Generate query embedding
        query_embedding = await self._embed(query)

        # Build where clause for tag filtering
        # Chroma doesn't support array contains, so we use string matching
        where_clause = None
        if user_tags:
            # Match any document where tags string contains any of user's tags
            where_clause = {"$or": [{"tags": {"$contains": tag}} for tag in user_tags]}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,  # Get more to filter by score
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Chroma returns distances (lower is better), convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Convert distance to similarity

                if score < score_threshold:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                text = results["documents"][0][i] if results["documents"] else ""

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata.get("document_id", ""),
                        document_name=metadata.get("document_name", ""),
                        chunk_text=text,
                        chunk_index=int(metadata.get("chunk_index", 0)),
                        score=score,
                        page_number=int(metadata["page_number"])
                        if metadata.get("page_number")
                        else None,
                        section=metadata.get("section"),
                    )
                )

        # Sort by score and limit
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:limit]

    async def delete_document(self, document_id: str) -> int:
        """Delete all vectors for a document."""
        # Get existing IDs for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} vectors for document {document_id}")
            return len(results["ids"])

        return 0

    async def update_document_tags(self, document_id: str, tags: list[str]) -> int:
        """Update tags for all vectors of a document."""
        # Get existing vectors
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            return 0

        # Update metadatas with new tags
        new_metadatas = []
        for meta in results["metadatas"]:
            meta["tags"] = ",".join(tags)
            new_metadatas.append(meta)

        # Update in collection
        self.collection.update(
            ids=results["ids"],
            metadatas=new_metadatas,
        )

        logger.info(f"Updated tags for {len(results['ids'])} vectors of document {document_id}")
        return len(results["ids"])

    async def health_check(self) -> dict[str, Any]:
        """Check Chroma health."""
        try:
            count = self.collection.count()
            return {
                "healthy": True,
                "backend": "chroma",
                "details": {
                    "collection": self.collection_name,
                    "vector_count": count,
                    "persist_dir": self.persist_dir,
                },
            }
        except Exception as e:
            return {
                "healthy": False,
                "backend": "chroma",
                "details": {"error": str(e)},
            }

    async def clear_collection(self) -> bool:
        """Clear all vectors from collection."""
        try:
            # Delete and recreate collection
            if self._client:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            logger.warning(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
