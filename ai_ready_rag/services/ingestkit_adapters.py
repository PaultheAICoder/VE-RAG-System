"""Adapter classes that bridge ingestkit backends to VE-RAG's infrastructure.

Each adapter satisfies an ingestkit Protocol interface while injecting VE-RAG's
access-control metadata (tags, document_id, tenant_id) and using VE-RAG's
configured services (Qdrant collection, Ollama models).

Adapters:
- VERagVectorStoreAdapter: Writes to VE-RAG's Qdrant collection with matching payload schema
- VERagEmbeddingAdapter: Delegates to ingestkit's OllamaEmbedding (same Ollama server)
- VERagLLMAdapter: Delegates to ingestkit's OllamaLLM (same Ollama server)
- ExcelStructuredDB: Delegates to ingestkit's SQLiteStructuredDB (separate DB file)
- VERagFormDBAdapter: SQLite wrapper implementing FormDBBackend protocol (ingestkit-forms)
- VERagLayoutFingerprinter: LayoutFingerprinter protocol (ingestkit-forms)
- VERagOCRAdapter: OCRBackend protocol (Tesseract/PaddleOCR) (ingestkit-forms)
- VERagVLMAdapter: VLMBackend protocol (Ollama VLM) (ingestkit-forms)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

if TYPE_CHECKING:
    from ingestkit_core.models import ChunkPayload

logger = logging.getLogger(__name__)


class VERagVectorStoreAdapter:
    """Adapts ingestkit's VectorStoreBackend protocol to VE-RAG's Qdrant collection.

    Merges VE-RAG access-control fields (tags, document_id, tenant_id, etc.) into
    every Qdrant point payload so that existing search filters and lifecycle
    operations (delete_document, update_document_tags) work on ingestkit-written points.

    Uses sync QdrantClient because ingestkit's pipeline is synchronous.
    """

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection_name: str,
        embedding_dimension: int,
        document_id: str,
        document_name: str,
        tags: list[str],
        uploaded_by: str,
        tenant_id: str = "default",
    ) -> None:
        self._client = QdrantClient(url=qdrant_url, timeout=30.0)
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension
        self._document_id = document_id
        self._document_name = document_name
        self._tags = tags
        self._uploaded_by = uploaded_by
        self._tenant_id = tenant_id
        self._uploaded_at = datetime.now(UTC).isoformat()

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Create the collection if it does not already exist."""
        name = self._resolve_collection(collection)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", name, vector_size)

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        """Upsert chunks with VE-RAG's payload schema merged in.

        Each point payload contains:
        - VE-RAG fields: chunk_id, document_id, document_name, chunk_index,
          chunk_text, tags, tenant_id, uploaded_by, uploaded_at, page_number, section
        - ingestkit provenance fields (prefixed with ingestkit_*): source_format,
          ingestion_method, parser_version, ingest_key, sheet_name, etc.
        """
        if not chunks:
            return 0

        name = self._resolve_collection(collection)
        points = []

        for chunk in chunks:
            meta = chunk.metadata
            chunk_index = meta.chunk_index

            payload = {
                # VE-RAG standard fields (must match vector_service.py schema)
                "chunk_id": chunk.id,
                "document_id": self._document_id,
                "document_name": self._document_name,
                "chunk_index": chunk_index,
                "chunk_text": chunk.text,
                "tags": self._tags,
                "tenant_id": self._tenant_id,
                "uploaded_by": self._uploaded_by,
                "uploaded_at": self._uploaded_at,
                "page_number": None,  # Excel files don't have page numbers
                "section": meta.section_title,
                # ingestkit provenance fields
                "ingestkit_source_format": meta.source_format,
                "ingestkit_ingestion_method": meta.ingestion_method,
                "ingestkit_parser_version": meta.parser_version,
                "ingestkit_ingest_key": meta.ingest_key,
                "ingestkit_chunk_hash": meta.chunk_hash,
                "ingestkit_source_uri": meta.source_uri,
                "ingestkit_ingest_run_id": meta.ingest_run_id,
            }

            # Add Excel-specific fields if present
            if meta.table_name:
                payload["ingestkit_table_name"] = meta.table_name
            if meta.row_count is not None:
                payload["ingestkit_row_count"] = meta.row_count
            if meta.columns:
                payload["ingestkit_columns"] = meta.columns

            # Add sheet_name if the metadata has it (ChunkMetadata extends BaseChunkMetadata)
            if hasattr(meta, "sheet_name"):
                payload["ingestkit_sheet_name"] = meta.sheet_name

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.vector,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=name, points=points)
        logger.info(
            "Upserted %d ingestkit chunks for document %s to '%s'",
            len(points),
            self._document_id,
            name,
        )
        return len(points)

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        """Create a payload index on the specified field."""
        type_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
        }
        schema_type = type_map.get(field_type)
        if schema_type is None:
            raise ValueError(f"Unsupported field_type '{field_type}'")

        name = self._resolve_collection(collection)
        self._client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema_type,
        )

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete points by their IDs."""
        if not ids:
            return 0

        name = self._resolve_collection(collection)
        self._client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    def delete_by_filter(self, collection: str, field: str, value: str) -> None:
        """Delete points matching a payload filter (e.g., by ingest_key)."""
        name = self._resolve_collection(collection)
        self._client.delete(
            collection_name=name,
            points_selector=Filter(
                must=[FieldCondition(key=field, match=MatchValue(value=value))],
            ),
        )
        logger.info(
            "Deleted ingestkit points where %s=%s from '%s'",
            field,
            value,
            name,
        )

    def _resolve_collection(self, collection: str) -> str:
        """Use VE-RAG's collection name, ignoring ingestkit's default_collection."""
        return self._collection_name


def create_embedding_adapter(
    *,
    ollama_url: str,
    embedding_model: str,
    embedding_dimension: int,
    backend_timeout: float = 30.0,
):
    """Create an ingestkit OllamaEmbedding that uses VE-RAG's Ollama settings.

    Returns an OllamaEmbedding instance (satisfies EmbeddingBackend protocol).
    """
    from ingestkit_excel.backends.ollama import OllamaEmbedding
    from ingestkit_excel.config import ExcelProcessorConfig

    config = ExcelProcessorConfig(backend_timeout_seconds=backend_timeout)
    return OllamaEmbedding(
        base_url=ollama_url,
        model=embedding_model,
        embedding_dimension=embedding_dimension,
        config=config,
    )


def create_llm_adapter(
    *,
    ollama_url: str,
    backend_timeout: float = 30.0,
):
    """Create an ingestkit OllamaLLM that uses VE-RAG's Ollama settings.

    Returns an OllamaLLM instance (satisfies LLMBackend protocol).
    """
    from ingestkit_excel.backends.ollama import OllamaLLM
    from ingestkit_excel.config import ExcelProcessorConfig

    config = ExcelProcessorConfig(backend_timeout_seconds=backend_timeout)
    return OllamaLLM(base_url=ollama_url, config=config)


def create_structured_db(*, db_path: str):
    """Create an ingestkit SQLiteStructuredDB for Excel table storage.

    Returns an SQLiteStructuredDB instance (satisfies StructuredDBBackend protocol).
    Uses a separate SQLite file from VE-RAG's app database.
    """
    from ingestkit_excel.backends.sqlite import SQLiteStructuredDB

    return SQLiteStructuredDB(db_path=db_path)


# ---------------------------------------------------------------------------
# ingestkit-forms adapters
# ---------------------------------------------------------------------------


class VERagFormDBAdapter:
    """Thin SQLite wrapper implementing ingestkit_forms.FormDBBackend protocol.

    Uses a separate SQLite file (forms_data.db) from VE-RAG's main app DB.
    All table names are validated through ingestkit-forms' validate_table_name()
    to avoid regex duplication and drift.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = self._validate_path(db_path)
        self._ensure_db()

    @staticmethod
    def _validate_path(db_path: str) -> str:
        """Normalize and validate DB path is under data/ directory."""
        resolved = Path(db_path).resolve()
        allowed_parent = Path("./data").resolve()
        if not str(resolved).startswith(str(allowed_parent)):
            raise ValueError(f"forms_db_path must be under data/: {db_path}")
        return str(resolved)

    def _ensure_db(self) -> None:
        """Create the DB file and parent directories if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

    @staticmethod
    def check_table_name(name: str) -> None:
        """Validate table name using ingestkit-forms' validate_table_name.

        Raises ValueError if the name is not a safe SQL identifier.
        Uses ingestkit-forms' canonical regex to avoid duplication and drift.
        """
        from ingestkit_forms import validate_table_name

        error = validate_table_name(name)
        if error is not None:
            logger.error("forms.cleanup.unsafe_identifier", extra={"table": name})
            raise ValueError(f"Unsafe table identifier rejected: {name!r} — {error}")

    def execute_sql(self, sql: str, params: tuple | None = None) -> None:
        """Execute a SQL statement (CREATE TABLE, ALTER TABLE, INSERT OR REPLACE)."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(sql, params or ())
            conn.commit()
        finally:
            conn.close()

    def get_table_columns(self, table_name: str) -> list[str]:
        """Return column names of an existing table."""
        self.check_table_name(table_name)
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(f"PRAGMA table_info([{table_name}])")
            return [row[1] for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_rows(self, table_name: str, column: str, values: list[str]) -> int:
        """Delete rows where ``column`` IN ``values``. Returns count deleted."""
        self.check_table_name(table_name)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                f"DELETE FROM [{table_name}] WHERE [{column}] IN ({placeholders})",
                values,
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def table_exists(self, table_name: str) -> bool:
        """Return True if the table exists in the database."""
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def get_connection_uri(self) -> str:
        """Return the database connection URI."""
        return f"sqlite:///{self._db_path}"


class VERagLayoutFingerprinter:
    """LayoutFingerprinter protocol adapter using ingestkit-forms' built-in impl.

    Wraps compute_layout_fingerprint_from_file() to satisfy the protocol's
    compute_fingerprint(file_path) -> list[bytes] method.

    A renderer callable is required for non-image formats (PDF, XLSX).
    The renderer signature is: (file_path: str, dpi: int) -> list[PIL.Image.Image]
    """

    def __init__(self, config, renderer=None) -> None:
        self._config = config
        self._renderer = renderer

    def compute_fingerprint(self, file_path: str) -> list[bytes]:
        """Compute per-page layout fingerprints for a document."""
        from ingestkit_forms import compute_layout_fingerprint_from_file

        result = compute_layout_fingerprint_from_file(
            file_path,
            self._config,
            renderer=self._renderer,
        )
        # compute_layout_fingerprint_from_file returns concatenated bytes for all
        # pages. The protocol expects list[bytes] with one element per page.
        if isinstance(result, list):
            return result
        # Split concatenated bytes into per-page chunks
        page_size = self._config.fingerprint_grid_rows * self._config.fingerprint_grid_cols
        if page_size > 0 and len(result) > page_size:
            return [result[i : i + page_size] for i in range(0, len(result), page_size)]
        return [result]


class VERagOCRAdapter:
    """OCR adapter implementing ingestkit_forms.OCRBackend protocol.

    Delegates to Tesseract (default) or PaddleOCR based on settings.
    """

    def __init__(self, engine: str = "tesseract") -> None:
        self._engine = engine

    def ocr_region(
        self,
        image_bytes: bytes,
        language: str = "en",
        config: str | None = None,
        timeout: float | None = None,
    ):
        """Run OCR on a cropped image region."""

        if self._engine == "tesseract":
            return self._tesseract_ocr(image_bytes, language, config)
        elif self._engine == "paddleocr":
            return self._paddleocr_ocr(image_bytes, language)
        raise ValueError(f"Unknown OCR engine: {self._engine}")

    def engine_name(self) -> str:
        return self._engine

    def _tesseract_ocr(self, image_bytes: bytes, language: str, config: str | None):
        """Run Tesseract OCR on image bytes."""
        import io

        from ingestkit_forms import OCRRegionResult
        from PIL import Image

        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is required for Tesseract OCR. Install with: pip install pytesseract"
            ) from exc

        img = Image.open(io.BytesIO(image_bytes))
        # Map language codes: ingestkit uses 'en', tesseract uses 'eng'
        lang_map = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu"}
        tess_lang = lang_map.get(language, language)
        tess_config = config or "--psm 6"
        data = pytesseract.image_to_data(
            img, lang=tess_lang, config=tess_config, output_type=pytesseract.Output.DICT
        )
        texts = []
        confidences = []
        for i, text in enumerate(data["text"]):
            if text.strip():
                texts.append(text)
                conf = data["conf"][i]
                confidences.append(max(0.0, float(conf)) / 100.0)

        full_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRRegionResult(
            text=full_text,
            confidence=avg_conf,
            char_confidences=confidences if confidences else None,
            engine="tesseract",
        )

    def _paddleocr_ocr(self, image_bytes: bytes, language: str):
        """Run PaddleOCR on image bytes."""
        import io

        import numpy as np
        from ingestkit_forms import OCRRegionResult
        from PIL import Image

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is required for PaddleOCR engine. Install with: pip install paddleocr"
            ) from exc

        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        lang_map = {"en": "en", "es": "es", "fr": "fr", "de": "german"}
        paddle_lang = lang_map.get(language, language)
        ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
        result = ocr.ocr(img_array, cls=True)

        texts = []
        confidences = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                texts.append(text)
                confidences.append(float(conf))

        full_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRRegionResult(
            text=full_text,
            confidence=avg_conf,
            char_confidences=None,
            engine="paddleocr",
        )


class VERagVLMAdapter:
    """VLM adapter implementing ingestkit_forms.VLMBackend protocol.

    Calls Ollama API with a multimodal model (e.g., qwen2.5-vl:7b).
    Only created when forms_vlm_enabled=True.
    """

    def __init__(self, ollama_url: str, model: str = "qwen2.5-vl:7b") -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model

    def extract_field(
        self,
        image_bytes: bytes,
        field_type: str,
        field_name: str,
        extraction_hint: str | None = None,
        timeout: float | None = None,
    ):
        """Run VLM extraction on a cropped field image."""
        import base64

        import httpx
        from ingestkit_forms import VLMFieldResult

        b64_image = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            f"Extract the {field_type} value for the field '{field_name}' "
            f"from this form image. Return ONLY the extracted value, nothing else."
        )
        if extraction_hint:
            prompt += f" Hint: {extraction_hint}"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
        }
        req_timeout = timeout or 30.0
        resp = httpx.post(
            f"{self._ollama_url}/api/generate",
            json=payload,
            timeout=req_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        value = data.get("response", "").strip()
        # Checkboxes: interpret as boolean
        if field_type == "checkbox":
            value = value.lower() in ("yes", "true", "checked", "x", "1")

        return VLMFieldResult(
            value=value if value else None,
            confidence=0.7,  # VLM doesn't provide confidence; use conservative default
            model=self._model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Ping Ollama API to check VLM model is loaded."""
        import httpx

        try:
            resp = httpx.get(f"{self._ollama_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(self._model) for m in models)
        except Exception:
            return False
