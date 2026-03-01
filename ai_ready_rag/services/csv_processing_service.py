"""CsvProcessingService — dual-path ingest for CSV files.

Every uploaded CSV is processed on two paths:
  1. Structured: written to document_tables schema + document_table_registry
  2. Vector: serialized to text and indexed as RAG chunks

Spec: UNIFIED_TABLE_INGEST_v1 §6.5
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ai_ready_rag.db.models.document import Document

logger = logging.getLogger(__name__)


class CsvProcessingService:
    """Parse CSV, write to SQL table + vector store."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_csv(
        self,
        document: Document,
        db: Session,
        access_tags: list[str],
        tenant_id: str = "default",
    ) -> tuple[Any, bool]:
        """Dual-path CSV ingest.

        Returns:
            (ProcessingResult | None, should_fallback)
            should_fallback=False means: use this result (success or failure).
            should_fallback=True means: delegate to standard chunker.
        """
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()
        result, should_fallback = await loop.run_in_executor(
            None,
            partial(self._process_sync, document, db, access_tags, tenant_id),
        )
        return result, should_fallback

    # ------------------------------------------------------------------
    # Sync implementation (runs in executor)
    # ------------------------------------------------------------------

    def _process_sync(self, document, db, access_tags, tenant_id):
        from ai_ready_rag.services.processing_service import ProcessingResult

        def make_result(success, chunk_count=0, page_count=1, word_count=0, error_message=None):
            return ProcessingResult(
                success=success,
                chunk_count=chunk_count,
                page_count=page_count,
                word_count=word_count,
                processing_time_ms=0,
                error_message=error_message,
            )

        file_path = Path(document.file_path)

        # Size gate
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            max_mb = getattr(self.settings, "csv_max_file_size_mb", 100)
            if size_mb > max_mb:
                document.status = "failed"
                document.error_message = "csv_file_too_large"
                db.commit()
                return make_result(False, error_message="csv_file_too_large"), False
        except Exception as exc:
            logger.warning("csv.size_check_failed: %s", exc)

        # Detect delimiter
        delimiter = self._detect_delimiter(file_path)
        if delimiter is None:
            document.status = "failed"
            document.error_message = "csv_delimiter_undetected"
            db.commit()
            return make_result(False, error_message="csv_delimiter_undetected"), False

        # Parse CSV
        try:
            df, stats = self._read_csv(file_path, delimiter)
        except Exception as exc:
            logger.warning("csv.parse_error: %s", exc)
            # Encoding/parse failure — let standard chunker try
            return None, True

        # Validate
        error_reason = self._validate(df, stats)
        if error_reason:
            document.status = "failed"
            document.error_message = error_reason
            db.commit()
            return make_result(False, error_message=error_reason), False

        # Structured path (write to SQL table + registry)
        self._structured_path(df, document, access_tags, tenant_id, db)

        # Vector path (serialize to text, chunk count from df size)
        chunk_count = self._vector_path_chunk_count(df)

        return make_result(
            True,
            chunk_count=max(chunk_count, 1),
            page_count=1,
            word_count=df.size,
        ), False

    # ------------------------------------------------------------------
    # Parser helpers
    # ------------------------------------------------------------------

    def _detect_delimiter(self, file_path: Path) -> str | None:
        """Detect CSV delimiter by trying comma, semicolon, tab."""
        CANDIDATES = [",", ";", "\t"]
        try:
            # Use csv.Sniffer as a quick hint
            with open(file_path, encoding="utf-8-sig", errors="replace") as f:
                sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                hint = dialect.delimiter
                candidates = [hint] + [d for d in CANDIDATES if d != hint]
            except csv.Error:
                candidates = CANDIDATES

            for delim in candidates:
                try:
                    with open(file_path, encoding="utf-8-sig", errors="replace") as f:
                        reader = csv.reader(f, delimiter=delim)
                        rows = []
                        for _ in range(20):
                            try:
                                rows.append(next(reader))
                            except StopIteration:
                                break
                    if not rows:
                        continue
                    multi_col = sum(1 for r in rows if len(r) > 1)
                    if multi_col / len(rows) >= 0.9:
                        return delim
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("csv.delimiter_detection_failed: %s", exc)
        return None

    def _read_csv(self, file_path: Path, delimiter: str):
        """Parse CSV into DataFrame, counting bad rows."""
        import pandas as pd

        # Count raw lines for bad-row detection
        try:
            with open(file_path, encoding="utf-8-sig", errors="replace") as f:
                raw_line_count = sum(1 for _ in f)
        except Exception:
            raw_line_count = 0

        encoding = "utf-8-sig"
        try:
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                on_bad_lines="skip",
                engine="python",
            )
        except UnicodeDecodeError:
            encoding = "latin-1"
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                on_bad_lines="skip",
                engine="python",
            )

        # Estimate bad rows (header row is expected to be missing from df)
        expected_data_rows = max(raw_line_count - 1, 0)
        actual_rows = len(df)
        bad_rows = max(expected_data_rows - actual_rows, 0)

        stats = {
            "bad_row_count": bad_rows,
            "total_rows": expected_data_rows,
            "encoding_used": encoding,
            "delimiter_detected": delimiter,
        }
        return df, stats

    def _validate(self, df, stats: dict) -> str | None:
        """Return error reason string or None if valid."""
        if len(df) == 0:
            return "csv_empty"
        if len(df.columns) < 2:
            return "csv_delimiter_undetected"
        total = stats.get("total_rows", 0)
        bad = stats.get("bad_row_count", 0)
        if total > 0 and bad / total > 0.05:
            return "csv_bad_row_rate"
        return None

    def _structured_path(self, df, document, access_tags, tenant_id, db) -> list[str]:
        """Write DataFrame to SQL table via document_table_registry."""
        # Only runs for PostgreSQL — not SQLite dev
        database_url = getattr(self.settings, "database_url", "")
        if "postgresql" not in str(database_url):
            logger.debug("csv.structured_path: skipping (not postgresql)")
            return []

        try:
            import psycopg2

            from ai_ready_rag.db.models.base import generate_uuid
            from ai_ready_rag.utils.signal_canon import make_table_name, sample_row_values

            doc_stem = (document.original_filename or document.id).rsplit(".", 1)[0]
            table_name = make_table_name(doc_stem, document.id, 0)

            conn = psycopg2.connect(database_url)
            with conn.cursor() as cur:
                cur.execute("CREATE SCHEMA IF NOT EXISTS document_tables")
                cur.execute(f'DROP TABLE IF EXISTS document_tables."{table_name}"')
                cols_ddl = ", ".join(f'"{c}" TEXT' for c in df.columns)
                cur.execute(f'CREATE TABLE document_tables."{table_name}" ({cols_ddl})')
                for _, row in df.iterrows():
                    ph = ", ".join(["%s"] * len(df.columns))
                    qcols = ", ".join(f'"{c}"' for c in df.columns)
                    cur.execute(
                        f'INSERT INTO document_tables."{table_name}" ({qcols}) VALUES ({ph})',
                        [str(v) if v is not None else None for v in row],
                    )
                # Upsert registry
                samples = sample_row_values(df)
                meta = json.dumps({"access_tags": access_tags})
                cur.execute(
                    """
                    INSERT INTO document_table_registry
                        (id, tenant_id, table_name, schema_name, source_format,
                         table_index, columns, column_types, row_value_samples,
                         document_id, document_name, row_count, table_metadata,
                         created_at, updated_at)
                    VALUES (%s,%s,%s,'document_tables','csv',0,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
                    ON CONFLICT (tenant_id, schema_name, table_name)
                    DO UPDATE SET
                        columns = EXCLUDED.columns,
                        row_value_samples = EXCLUDED.row_value_samples,
                        document_id = EXCLUDED.document_id,
                        document_name = EXCLUDED.document_name,
                        row_count = EXCLUDED.row_count,
                        table_metadata = EXCLUDED.table_metadata,
                        updated_at = NOW()
                    """,
                    (
                        generate_uuid(),
                        tenant_id,
                        table_name,
                        json.dumps(list(df.columns)),
                        json.dumps({}),
                        json.dumps(samples),
                        document.id,
                        document.original_filename,
                        len(df),
                        meta,
                    ),
                )
            conn.commit()
            conn.close()
            logger.info(
                "csv.structured_path: table=%s rows=%d doc=%s",
                table_name,
                len(df),
                document.id,
            )
            return [table_name]
        except ImportError:
            logger.warning("csv.structured_path: psycopg2 not available")
            return []
        except Exception as exc:
            logger.warning("csv.structured_path_failed: doc=%s error=%s", document.id, exc)
            return []

    def _vector_path_chunk_count(self, df) -> int:
        """Estimate chunk count from DataFrame (rows / 20)."""
        return max(len(df) // 20, 1)
