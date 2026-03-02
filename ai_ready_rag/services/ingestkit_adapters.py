"""Adapter classes that bridge ingestkit backends to VE-RAG's infrastructure.

Each adapter satisfies an ingestkit Protocol interface while injecting VE-RAG's
access-control metadata (tags, document_id, tenant_id) and using VE-RAG's
configured services (Qdrant collection, Claude CLI, PostgreSQL).

Adapters:
- VERagVectorStoreAdapter: Writes to VE-RAG's pgvector chunk_vectors table
- VERagEmbeddingAdapter: Delegates to ingestkit's OllamaEmbedding (same Ollama server)
- VERagClaudeLLM: LLMBackend using Claude CLI (`claude -p`) — primary LLM for all ingestkit
- VERagPostgresStructuredDB: StructuredDBBackend over PostgreSQL (replaces SQLite for Excel)
- VERagPostgresFormDB: FormDBBackend over PostgreSQL (replaces SQLite for forms)
- VERagLayoutFingerprinter: LayoutFingerprinter protocol (ingestkit-forms)
- VERagOCRAdapter: OCRBackend protocol (Tesseract/PaddleOCR) (ingestkit-forms)
- VERagVLMAdapter: VLMBackend protocol (Ollama VLM) (ingestkit-forms)
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestkit_core.models import ChunkPayload

logger = logging.getLogger(__name__)


class VERagVectorStoreAdapter:
    """Adapts ingestkit's VectorStoreBackend protocol to VE-RAG's pgvector store.

    Writes chunks directly to the chunk_vectors table using psycopg2, merging
    VE-RAG access-control fields (tags, document_id, tenant_id, etc.) into every
    row so RAG search and lifecycle operations work on ingestkit-written chunks.

    The `collection` parameter accepted by ingestkit protocol methods is ignored —
    all chunks go to the shared chunk_vectors table (managed by Alembic migrations).
    """

    def __init__(
        self,
        *,
        database_url: str,
        embedding_dimension: int,
        document_id: str,
        document_name: str,
        tags: list[str],
        uploaded_by: str,
        tenant_id: str = "default",
    ) -> None:
        self._database_url = database_url
        self._embedding_dimension = embedding_dimension
        self._document_id = document_id
        self._document_name = document_name
        self._tags = tags
        self._uploaded_by = uploaded_by
        self._tenant_id = tenant_id
        self._uploaded_at = datetime.now(UTC).isoformat()

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """No-op — chunk_vectors table is managed by Alembic migrations."""

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        """Insert chunks into chunk_vectors with VE-RAG metadata merged in.

        Each row contains:
        - VE-RAG fields: document_id, document_name, chunk_index, chunk_text,
          tags, tenant_id, uploaded_by, uploaded_at, page_number, section
        - ingestkit provenance fields (stored in metadata_ JSON under ingestkit_*)
        """
        if not chunks:
            return 0

        import psycopg2

        conn = psycopg2.connect(self._database_url)
        try:
            with conn.cursor() as cur:
                # Re-index: remove existing chunks for this document first
                cur.execute(
                    "DELETE FROM chunk_vectors WHERE document_id = %s",
                    (self._document_id,),
                )

                for chunk in chunks:
                    meta = chunk.metadata
                    chunk_index = meta.chunk_index

                    # chunk_vectors.id must be a valid UUID string
                    try:
                        chunk_id = str(uuid.UUID(chunk.id))
                    except ValueError:
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))

                    metadata: dict = {
                        "tags": self._tags,
                        "document_name": self._document_name,
                        "uploaded_by": self._uploaded_by,
                        "uploaded_at": self._uploaded_at,
                        "page_number": None,
                        "section": meta.section_title,
                        "ingestkit_source_format": meta.source_format,
                        "ingestkit_ingestion_method": meta.ingestion_method,
                        "ingestkit_parser_version": meta.parser_version,
                        "ingestkit_ingest_key": meta.ingest_key,
                        "ingestkit_chunk_hash": meta.chunk_hash,
                        "ingestkit_source_uri": meta.source_uri,
                        "ingestkit_ingest_run_id": meta.ingest_run_id,
                    }
                    if meta.table_name:
                        metadata["ingestkit_table_name"] = meta.table_name
                    if meta.row_count is not None:
                        metadata["ingestkit_row_count"] = meta.row_count
                    if meta.columns:
                        metadata["ingestkit_columns"] = meta.columns
                    if hasattr(meta, "sheet_name"):
                        metadata["ingestkit_sheet_name"] = meta.sheet_name

                    vector = chunk.vector  # pre-computed by VERagEmbeddingAdapter
                    if vector:
                        vector_str = f"[{','.join(str(x) for x in vector)}]"
                        cur.execute(
                            "INSERT INTO chunk_vectors "
                            "(id, document_id, chunk_index, chunk_text, metadata_, "
                            "tenant_id, embedding, vector_embedding) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS vector))",
                            (
                                chunk_id,
                                self._document_id,
                                chunk_index,
                                chunk.text,
                                json.dumps(metadata),
                                self._tenant_id,
                                json.dumps(vector),
                                vector_str,
                            ),
                        )
                    else:
                        cur.execute(
                            "INSERT INTO chunk_vectors "
                            "(id, document_id, chunk_index, chunk_text, metadata_, "
                            "tenant_id, embedding) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (
                                chunk_id,
                                self._document_id,
                                chunk_index,
                                chunk.text,
                                json.dumps(metadata),
                                self._tenant_id,
                                None,
                            ),
                        )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        logger.info(
            "ingestkit.pgvector.upsert: %d chunks for document %s",
            len(chunks),
            self._document_id,
        )
        return len(chunks)

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        """No-op — pgvector indexes are managed by Alembic migrations."""

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete chunks by their IDs."""
        if not ids:
            return 0

        import psycopg2

        conn = psycopg2.connect(self._database_url)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunk_vectors WHERE id = ANY(%s)", (ids,))
                deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted

    def delete_by_filter(self, collection: str, field: str, value: str) -> None:
        """Delete chunks matching a metadata JSON field value."""
        import psycopg2

        conn = psycopg2.connect(self._database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunk_vectors "
                    "WHERE document_id = %s AND metadata_::jsonb @> %s::jsonb",
                    (self._document_id, json.dumps({field: value})),
                )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            "ingestkit.pgvector.delete_by_filter: %s=%s for document %s",
            field,
            value,
            self._document_id,
        )


class _SanitizingEmbeddingAdapter:
    """Wraps an OllamaEmbedding with two safety behaviours:

    1. None / empty strings → single space (Ollama rejects null input).
    2. Texts that exceed the model's context window → split on newline
       boundaries, embed each segment, return mean-pooled vector.
       This preserves all content without truncation.

    nomic-embed-text on Ollama has a ~4 096-char context window; we keep a
    conservative margin of 3 800 characters per segment.
    """

    # Safe character limit per segment (conservative below the ~4096 limit)
    _MAX_CHARS = 3800

    def __init__(self, inner):
        self._inner = inner

    # ------------------------------------------------------------------
    # EmbeddingBackend protocol
    # ------------------------------------------------------------------

    def embed(self, texts, **kwargs) -> list[list[float]]:
        # Tolerate bare-string callers (API contract violation in rechunk path)
        if isinstance(texts, str):
            texts = [texts]

        # Phase 1 — expand each input into ≥1 segments
        all_segments: list[str] = []
        counts: list[int] = []

        for t in texts:
            if not isinstance(t, str) or not t.strip():
                all_segments.append(" ")
                counts.append(1)
            else:
                segs = self._split_text(t)
                all_segments.extend(segs)
                counts.append(len(segs))

        # Phase 2 — single batch embed call (all segments together)
        all_vectors = self._inner.embed(all_segments, **kwargs)

        # Phase 3 — mean-pool segments back to one vector per original input
        result: list[list[float]] = []
        idx = 0
        for count in counts:
            vecs = all_vectors[idx : idx + count]
            if count == 1:
                result.append(vecs[0])
            else:
                dim = len(vecs[0])
                mean = [sum(v[i] for v in vecs) / count for i in range(dim)]
                result.append(mean)
            idx += count

        return result

    def dimension(self) -> int:
        return self._inner.dimension()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """Split text on newline boundaries so each segment ≤ _MAX_CHARS.

        Keeps logical lines together; never cuts mid-line.  A line longer
        than _MAX_CHARS on its own is emitted as a single over-length
        segment (the model will truncate internally, but that is better
        than losing the line entirely).
        """
        if len(text) <= self._MAX_CHARS:
            return [text]

        lines = text.split("\n")
        segments: list[str] = []
        current: list[str] = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for the newline separator
            if current_len + line_len > self._MAX_CHARS and current:
                segments.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len

        if current:
            segments.append("\n".join(current))

        return segments or [text[: self._MAX_CHARS]]


def create_embedding_adapter(
    *,
    ollama_url: str,
    embedding_model: str,
    embedding_dimension: int,
    backend_timeout: float = 30.0,
):
    """Create an ingestkit OllamaEmbedding that uses VE-RAG's Ollama settings.

    Returns an EmbeddingBackend-compatible instance wrapped with null sanitization.
    """
    from ingestkit_excel.backends.ollama import OllamaEmbedding
    from ingestkit_excel.config import ExcelProcessorConfig

    config = ExcelProcessorConfig(backend_timeout_seconds=backend_timeout)
    inner = OllamaEmbedding(
        base_url=ollama_url,
        model=embedding_model,
        embedding_dimension=embedding_dimension,
        config=config,
    )
    return _SanitizingEmbeddingAdapter(inner)


def create_llm_adapter():
    """Return a VERagClaudeLLM — Claude CLI is the primary LLM for all ingestkit pipelines."""
    return VERagClaudeLLM()


def create_structured_db(
    *, database_url: str, tags: list[str] | None = None
) -> VERagPostgresStructuredDB:
    """Return a PostgreSQL StructuredDBBackend for ingestkit-excel table storage."""
    return VERagPostgresStructuredDB(database_url=database_url, tags=tags)


# ---------------------------------------------------------------------------
# Claude CLI LLM adapter (primary LLM for all ingestkit pipelines)
# ---------------------------------------------------------------------------


class VERagClaudeLLM:
    """LLMBackend implementation that delegates to Claude via `claude -p`.

    Satisfies ingestkit's LLMBackend protocol (classify + generate).
    The `model` parameter passed by ingestkit is ignored — Claude is always used.
    """

    def classify(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout: float | None = None,
    ) -> dict:
        """Run a classification prompt through Claude and return parsed JSON."""
        system = (
            "You are a precise JSON classifier. "
            "Respond with valid JSON only — no markdown, no explanation."
        )
        result = self._run(system + "\n\n" + prompt, timeout=timeout)
        # Strip any accidental markdown fences
        text = (
            result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        return json.loads(text)

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        """Run a generation prompt through Claude and return the text response."""
        return self._run(prompt, timeout=timeout)

    def _run(self, prompt: str, *, timeout: float | None = None) -> str:
        """Invoke `claude -p` as a subprocess and return stdout."""
        import os

        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout or 120,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude -p exited {result.returncode}: {result.stderr[:200]}")
        return result.stdout.strip()


# ---------------------------------------------------------------------------
# PostgreSQL StructuredDB adapter (ingestkit-excel)
# ---------------------------------------------------------------------------


class VERagPostgresStructuredDB:
    """PostgreSQL StructuredDBBackend for ingestkit-excel table storage.

    Satisfies ingestkit's StructuredDBBackend protocol via structural subtyping.
    Uses a dedicated schema (``excel_tables``) to keep extracted spreadsheet
    tables separate from VE-RAG's application tables.
    """

    _SCHEMA = "excel_tables"

    def __init__(self, *, database_url: str, tags: list[str] | None = None) -> None:
        self._database_url = database_url
        self._access_tags: list[str] = tags or []
        self._ensure_schema()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._database_url)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._SCHEMA}"')
            conn.commit()

    def _qualified(self, table_name: str) -> str:
        return f'"{self._SCHEMA}"."{table_name}"'

    def create_table_from_dataframe(self, table_name: str, df) -> None:
        """Write a DataFrame as a table in the excel_tables schema."""
        from sqlalchemy import create_engine

        engine = create_engine(
            self._database_url,
            connect_args={"options": f"-csearch_path={self._SCHEMA}"},
        )
        try:
            df.to_sql(table_name, engine, if_exists="replace", index=False, schema=self._SCHEMA)
            # Register schema metadata in excel_table_registry for NL2SQL discovery
            self._register_table_in_registry(table_name, df)
        except Exception as exc:
            raise RuntimeError(f"Failed to write table '{table_name}': {exc}") from exc
        finally:
            engine.dispose()

    def _register_table_in_registry(self, table_name: str, df) -> None:
        """Write table schema to excel_table_registry for NL2SQL discovery.

        Uses UPSERT so re-ingesting the same file updates the existing row
        rather than creating a duplicate. Failures are logged but never raised
        (registry write must not break ingest).

        The ``access_tags`` from the document are stored in ``table_metadata``
        JSON so the ExcelTablesService can enforce tag-based access control
        on the NL2SQL path at query time.

        Also performs a dual-write to ``document_table_registry`` (Phase 2 unified
        pipeline). The dual-write is wrapped in its own try/except so it never
        breaks ingest when the new table does not yet exist.
        """
        import uuid as _uuid
        from datetime import datetime

        try:
            columns = [str(c) for c in df.columns]
            column_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            now = datetime.utcnow()
            # Embed access_tags in table_metadata so no schema migration is required.
            table_metadata = json.dumps({"access_tags": self._access_tags})

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO excel_table_registry
                            (id, table_name, schema_name, columns, column_types,
                             tenant_id, row_count, table_metadata, created_at, updated_at)
                        VALUES
                            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (schema_name, table_name)
                        DO UPDATE SET
                            columns = EXCLUDED.columns,
                            column_types = EXCLUDED.column_types,
                            row_count = EXCLUDED.row_count,
                            table_metadata = EXCLUDED.table_metadata,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            str(_uuid.uuid4()),
                            table_name,
                            self._SCHEMA,
                            json.dumps(columns),
                            json.dumps(column_types),
                            getattr(self, "_tenant_id", "default"),
                            len(df),
                            table_metadata,
                            now,
                            now,
                        ),
                    )
                conn.commit()
            logger.info(
                "excel_table_registry.written: table=%s columns=%d rows=%d access_tags=%s",
                table_name,
                len(columns),
                len(df),
                self._access_tags,
            )
        except Exception as exc:
            logger.warning("excel_table_registry.write_failed: %s", exc)

        # Dual-write to document_table_registry (unified pipeline, Phase 2).
        # Wrapped in its own try/except — silently skips if the table does not
        # exist yet (backward compat with DBs that have not run migration 009).
        self._write_to_document_registry(df, table_name, self._SCHEMA)

    def _write_to_document_registry(
        self,
        df,
        table_name: str,
        schema_name: str,
        doc_id: str | None = None,
        doc_name: str | None = None,
    ) -> None:
        """Upsert a row into document_table_registry with row_value_samples.

        Silently skips (logs WARNING) if the table does not exist so that
        deployments without migration 009 continue to function.
        """
        import uuid as _uuid

        from ai_ready_rag.utils.signal_canon import sample_row_values

        try:
            columns = [str(c) for c in df.columns]
            column_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
            row_samples = sample_row_values(df)
            tenant_id = getattr(self, "_tenant_id", "default")

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO document_table_registry
                            (id, tenant_id, table_name, schema_name, source_format,
                             columns, column_types, row_value_samples,
                             document_id, document_name, row_count,
                             created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (tenant_id, schema_name, table_name)
                        DO UPDATE SET
                            columns = EXCLUDED.columns,
                            column_types = EXCLUDED.column_types,
                            row_value_samples = EXCLUDED.row_value_samples,
                            document_id = EXCLUDED.document_id,
                            document_name = EXCLUDED.document_name,
                            row_count = EXCLUDED.row_count,
                            updated_at = NOW()
                        """,
                        (
                            str(_uuid.uuid4()),
                            tenant_id,
                            table_name,
                            schema_name,
                            "excel",
                            json.dumps(columns),
                            json.dumps(column_types),
                            json.dumps(row_samples),
                            doc_id,
                            doc_name,
                            len(df),
                        ),
                    )
                conn.commit()
            logger.info(
                "document_table_registry.written: table=%s schema=%s rows=%d samples_cols=%d",
                table_name,
                schema_name,
                len(df),
                len(row_samples),
            )
        except Exception as exc:
            logger.warning("document_table_registry.write_failed: %s", exc)

    def drop_table(self, table_name: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self._qualified(table_name)}")
            conn.commit()

    def table_exists(self, table_name: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s",
                    (self._SCHEMA, table_name),
                )
                return cur.fetchone() is not None

    def get_table_schema(self, table_name: str) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                    (self._SCHEMA, table_name),
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_connection_uri(self) -> str:
        return self._database_url


# ---------------------------------------------------------------------------
# ingestkit-forms adapters
# ---------------------------------------------------------------------------


class VERagFormDBAdapter:
    """PostgreSQL implementation of ingestkit_forms.FormDBBackend protocol.

    All form extraction tables are stored in the ``forms_data`` schema in
    the shared PostgreSQL database. Table names are validated through
    ingestkit-forms' validate_table_name() to avoid SQL injection.
    """

    _SCHEMA = "forms_data"

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._ensure_schema()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._database_url)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._SCHEMA}"')
            conn.commit()

    @staticmethod
    def check_table_name(name: str) -> None:
        """Validate table name using ingestkit-forms' validate_table_name."""
        from ingestkit_forms import validate_table_name

        error = validate_table_name(name)
        if error is not None:
            logger.error("forms.cleanup.unsafe_identifier", extra={"table": name})
            raise ValueError(f"Unsafe table identifier rejected: {name!r} — {error}")

    def _qualified(self, table_name: str) -> str:
        return f'"{self._SCHEMA}"."{table_name}"'

    # PostgreSQL silently truncates identifiers longer than 63 bytes.
    # When two ACORD field names share the same 63-char prefix, the truncation
    # produces duplicate column names and raises "specified more than once".
    _PG_MAX_IDENTIFIER_LEN = 63

    @staticmethod
    def _deduplicate_identifier(name: str, seen: dict[str, int]) -> str:
        """Truncate ``name`` to PG limit and append a suffix on collision.

        ``seen`` maps a (potentially already-truncated) base name to the next
        available integer suffix.  The dict is mutated in place so the caller
        can accumulate state across multiple column names.
        """
        limit = VERagFormDBAdapter._PG_MAX_IDENTIFIER_LEN
        truncated = name[:limit]
        if truncated not in seen:
            seen[truncated] = 1
            return truncated
        # Collision — find a suffix that fits within the limit
        n = seen[truncated]
        seen[truncated] = n + 1
        suffix = f"_{n}"
        base = name[: limit - len(suffix)]
        candidate = base + suffix
        # Ensure the suffixed name is also tracked to avoid future collisions
        seen.setdefault(candidate, 1)
        return candidate

    @classmethod
    def _sanitize_identifiers(cls, sql: str) -> str:
        """Rewrite any double-quoted identifier longer than 63 chars.

        Parses quoted identifiers in ``sql`` (e.g. ``"F[0].P1[0].SomeLong..."``)
        and replaces colliding truncations with a ``_N`` sequential suffix so that
        PostgreSQL never sees two identifiers that map to the same 63-char prefix.

        Only replaces identifiers that actually exceed the limit to minimise diff.

        Uses a stable mapping so the same original identifier always produces
        the same truncated name within a single SQL statement (critical for
        ``col = EXCLUDED.col`` in ON CONFLICT clauses).
        """
        import re

        seen: dict[str, int] = {}  # truncated_base -> next suffix counter
        resolved: dict[str, str] = {}  # original_name -> final truncated name
        limit = cls._PG_MAX_IDENTIFIER_LEN

        def replace_identifier(m: re.Match) -> str:
            ident = m.group(1)  # content inside the double quotes
            if len(ident) <= limit:
                seen.setdefault(ident, 1)
                return f'"{ident}"'
            # Return cached result if we already resolved this identifier
            if ident in resolved:
                return f'"{resolved[ident]}"'
            new_ident = cls._deduplicate_identifier(ident, seen)
            resolved[ident] = new_ident
            if new_ident != ident:
                logger.debug(
                    "forms.column_truncated: '%s' → '%s'",
                    ident[:40],
                    new_ident,
                )
            return f'"{new_ident}"'

        return re.sub(r'"([^"]*)"', replace_identifier, sql)

    @staticmethod
    def _translate_insert_or_replace(sql: str) -> str:
        """Translate SQLite INSERT OR REPLACE to PostgreSQL INSERT ... ON CONFLICT.

        ingestkit_forms generates:
            INSERT OR REPLACE INTO "table" (col1, col2, ...) VALUES (?, ?, ...)

        PostgreSQL requires:
            INSERT INTO "table" (col1, col2, ...) VALUES (%s, %s, ...)
            ON CONFLICT (_form_id) DO UPDATE SET col1 = EXCLUDED.col1, ...

        The conflict target is always _form_id (ingestkit spec §9.4).
        """
        import re

        m = re.match(
            r"INSERT\s+OR\s+REPLACE\s+INTO\s+(.+?)\s*\((.+?)\)\s*VALUES\s*\((.+?)\)\s*$",
            sql.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return sql  # Not a recognized INSERT OR REPLACE pattern — pass through

        table_ref = m.group(1).strip()
        cols_raw = m.group(2).strip()
        vals_raw = m.group(3).strip()

        # Parse column list (may contain quoted identifiers with commas inside)
        cols = [c.strip() for c in cols_raw.split(",")]

        # Build ON CONFLICT ... DO UPDATE SET for all non-PK columns
        update_cols = [c for c in cols if c.strip('"') != "_form_id"]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        return (
            f"INSERT INTO {table_ref} ({cols_raw}) VALUES ({vals_raw}) "
            f'ON CONFLICT ("_form_id") DO UPDATE SET {set_clause}'
        )

    def execute_sql(self, sql: str, params: tuple | None = None) -> None:
        """Execute a SQL statement (CREATE TABLE, ALTER TABLE, INSERT).

        Long column names are sanitized before execution: PostgreSQL silently
        truncates identifiers to 63 bytes, which causes "specified more than
        once" errors when two ACORD field names share the same 63-char prefix.
        """
        # Translate SQLite-style ? placeholders to psycopg2 %s
        if params:
            sql = sql.replace("?", "%s")
        # Translate SQLite INSERT OR REPLACE to PostgreSQL ON CONFLICT upsert
        sql = self._translate_insert_or_replace(sql)
        sql = self._sanitize_identifiers(sql)
        # Make ALTER TABLE ADD COLUMN idempotent — truncated identifiers may
        # collide with existing columns when a second document triggers schema
        # evolution with the same long field name.
        import re

        sql = re.sub(
            r"ALTER\s+TABLE\s+(.+?)\s+ADD\s+COLUMN\s+(?!IF\s+NOT\s+EXISTS\b)",
            r"ALTER TABLE \1 ADD COLUMN IF NOT EXISTS ",
            sql,
            flags=re.IGNORECASE,
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Set search_path so unqualified table names (from ingestkit_forms
                # generated SQL) resolve to forms_data rather than public.
                cur.execute(f'SET LOCAL search_path TO "{self._SCHEMA}", public')
                cur.execute(sql, params or ())
            conn.commit()

    def get_table_columns(self, table_name: str) -> list[str]:
        """Return column names of an existing table."""
        self.check_table_name(table_name)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                    (self._SCHEMA, table_name),
                )
                return [row[0] for row in cur.fetchall()]

    def delete_rows(self, table_name: str, column: str, values: list[str]) -> int:
        """Delete rows where ``column`` IN ``values``. Returns count deleted."""
        self.check_table_name(table_name)
        if not values:
            return 0
        placeholders = ",".join("%s" for _ in values)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DELETE FROM {self._qualified(table_name)} WHERE "{column}" IN ({placeholders})',
                    values,
                )
                count = cur.rowcount
            conn.commit()
        return count

    def table_exists(self, table_name: str) -> bool:
        """Return True if the table exists in the database."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s",
                    (self._SCHEMA, table_name),
                )
                return cur.fetchone() is not None

    def get_connection_uri(self) -> str:
        """Return the database connection URI."""
        return self._database_url


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


class VERagImageOCRAdapter:
    """OCR adapter implementing ingestkit_image.ImageOCRBackend protocol.

    Uses pytesseract to extract text from full images (not form regions).
    """

    def __init__(self, language: str = "eng", config: str | None = None) -> None:
        self._language = language
        self._config = config

    def ocr_image(
        self,
        image_bytes: bytes,
        language: str | None = None,
        config: str | None = None,
        timeout: float | None = None,
    ):
        """Run OCR on a full image and return an OCRResult."""
        import io

        from ingestkit_image import OCRResult
        from PIL import Image

        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is required for image OCR. Install with: pip install pytesseract"
            ) from exc

        lang = language or self._language
        tess_config = config or self._config or "--psm 6"

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        data = pytesseract.image_to_data(
            img, lang=lang, config=tess_config, output_type=pytesseract.Output.DICT
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
        return OCRResult(
            text=full_text,
            confidence=avg_conf,
            engine="tesseract",
            language=lang,
        )

    def engine_name(self) -> str:
        return "tesseract"


class VERagPDFWidgetAdapter:
    """PDF widget adapter implementing ingestkit_forms.PDFWidgetBackend protocol.

    Uses pypdf to extract AcroForm field values from fillable PDFs.
    """

    def extract_widgets(self, file_path: str, page: int) -> list:
        """Extract all form widgets from the specified page."""
        from ingestkit_forms.models import BoundingBox
        from ingestkit_forms.protocols import WidgetField
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        if page >= len(reader.pages):
            return []

        pdf_page = reader.pages[page]
        page_width = float(pdf_page.mediabox.width)
        page_height = float(pdf_page.mediabox.height)

        widgets = []
        annotations = pdf_page.get("/Annots")
        if not annotations:
            return []

        for annot in annotations:
            annot_obj = annot.get_object() if hasattr(annot, "get_object") else annot
            subtype = annot_obj.get("/Subtype")
            if subtype != "/Widget":
                continue

            field_name = annot_obj.get("/T", "")
            if hasattr(field_name, "__str__"):
                field_name = str(field_name)

            field_value = annot_obj.get("/V")
            if field_value is not None:
                field_value = str(field_value)
                if field_value.startswith("/"):
                    field_value = field_value[1:]

            ft = annot_obj.get("/FT", "/Tx")
            type_map = {"/Tx": "text", "/Btn": "checkbox", "/Ch": "dropdown"}
            field_type = type_map.get(str(ft), "text")

            rect = annot_obj.get("/Rect", [0, 0, 0, 0])
            rect = [float(r) for r in rect]
            x0 = min(rect[0], rect[2]) / page_width
            y0 = 1.0 - max(rect[1], rect[3]) / page_height
            x1 = max(rect[0], rect[2]) / page_width
            y1 = 1.0 - min(rect[1], rect[3]) / page_height
            bbox = BoundingBox(
                x=max(0.0, min(1.0, x0)),
                y=max(0.0, min(1.0, y0)),
                width=max(0.001, min(1.0, x1 - x0)),
                height=max(0.001, min(1.0, y1 - y0)),
            )

            widgets.append(
                WidgetField(
                    field_name=field_name,
                    field_value=field_value,
                    field_type=field_type,
                    bbox=bbox,
                    page=page,
                )
            )

        return widgets

    def has_form_fields(self, file_path: str) -> bool:
        """Check whether the PDF contains any fillable form fields."""
        from pypdf import PdfReader

        try:
            reader = PdfReader(file_path)
            # Check AcroForm at document root (most reliable)
            if reader.trailer.get("/Root"):
                root = reader.trailer["/Root"].get_object()
                acro_form = root.get("/AcroForm")
                if acro_form:
                    acro_obj = (
                        acro_form.get_object() if hasattr(acro_form, "get_object") else acro_form
                    )
                    fields = acro_obj.get("/Fields", [])
                    if len(fields) > 0:
                        return True
            # Fallback: check page annotations
            for pg in reader.pages:
                annots = pg.get("/Annots")
                if annots:
                    resolved = annots.get_object() if hasattr(annots, "get_object") else annots
                    if len(resolved) > 0:
                        return True
            return False
        except Exception:
            return False

    def engine_name(self) -> str:
        return "pypdf"
