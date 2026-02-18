"""Tests for migration CLI (Issue #286).

Tests use mocked AsyncQdrantClient to validate migration logic,
cursor checkpointing, point transformation, and verification gates.
"""

from collections import namedtuple
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models

from ai_ready_rag.cli.migrate_hybrid import (
    MigrationCursor,
    create_target_collection,
    do_migrate,
    gate_latency,
    gate_payload_integrity,
    gate_point_count,
    gate_search_parity,
    gate_sparse_presence,
    run_verification_gates,
    transform_points,
)

CollectionInfo = namedtuple("CollectionInfo", ["name"])


def _collections(*names: str):
    """Create a mock get_collections response with proper .name attributes."""
    return MagicMock(collections=[CollectionInfo(name=n) for n in names])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant_client():
    """Create a mock AsyncQdrantClient with common methods."""
    client = AsyncMock()
    client.get_collections.return_value = _collections("documents")
    client.get_collection.return_value = MagicMock(points_count=50)
    client.scroll.return_value = ([], None)
    client.upsert.return_value = None
    client.retrieve.return_value = []
    client.query_points.return_value = MagicMock(points=[])
    client.create_collection.return_value = None
    client.create_payload_index.return_value = None
    return client


@pytest.fixture
def sample_source_points():
    """Create sample points with unnamed vectors."""
    points = []
    for i in range(10):
        p = MagicMock()
        p.id = f"point-{i}"
        p.vector = [0.1 * i] * 768  # unnamed dense vector
        p.payload = {
            "chunk_text": f"This is chunk {i} with some text content.",
            "document_id": f"doc-{i % 3}",
            "document_name": f"file-{i % 3}.pdf",
            "tags": ["public"],
            "tenant_id": "default",
            "chunk_index": i,
            "sparse_indexed": False,
        }
        points.append(p)
    return points


@pytest.fixture
def mock_sparse_model():
    """Mock sparse embedding model returning deterministic vectors."""
    model = MagicMock()

    def fake_embed(texts):
        results = []
        for _ in texts:
            r = MagicMock()
            r.indices = MagicMock(tolist=lambda: [0, 1, 2])
            r.values = MagicMock(tolist=lambda: [0.5, 0.3, 0.1])
            results.append(r)
        return results

    model.embed = fake_embed
    return model


@pytest.fixture
def cursor_path(tmp_path):
    """Return a temp path for cursor file."""
    return str(tmp_path / ".migrate_cursor")


# ---------------------------------------------------------------------------
# Cursor tests
# ---------------------------------------------------------------------------


class TestMigrationCursor:
    def test_save_load_roundtrip(self, cursor_path):
        cursor = MigrationCursor(
            offset="abc-123", migrated=42, source_count=100, timestamp="2026-02-17T00:00:00Z"
        )
        cursor.save(cursor_path)

        loaded = MigrationCursor.load(cursor_path)
        assert loaded.offset == "abc-123"
        assert loaded.migrated == 42
        assert loaded.source_count == 100
        assert loaded.timestamp == "2026-02-17T00:00:00Z"

    def test_load_returns_empty_when_no_file(self, tmp_path):
        path = str(tmp_path / "nonexistent")
        cursor = MigrationCursor.load(path)
        assert cursor.offset is None
        assert cursor.migrated == 0

    def test_clear_removes_file(self, cursor_path):
        cursor = MigrationCursor(migrated=5)
        cursor.save(cursor_path)
        assert Path(cursor_path).exists()

        cursor.clear(cursor_path)
        assert not Path(cursor_path).exists()

    def test_clear_noop_when_no_file(self, tmp_path):
        path = str(tmp_path / "nonexistent")
        cursor = MigrationCursor()
        cursor.clear(path)  # Should not raise


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransformPoints:
    def test_unnamed_to_named(self, sample_source_points, mock_sparse_model):
        points = transform_points(sample_source_points, mock_sparse_model)

        assert len(points) == 10
        for p in points:
            assert isinstance(p, models.PointStruct)
            assert "dense" in p.vector
            assert "sparse" in p.vector
            assert p.payload["sparse_indexed"] is True

    def test_already_named_vectors(self, mock_sparse_model):
        """Points with dict vectors (re-run scenario) should extract dense."""
        p = MagicMock()
        p.id = "point-rerun"
        p.vector = {"dense": [0.1] * 768, "sparse": None}
        p.payload = {"chunk_text": "rerun test"}

        result = transform_points([p], mock_sparse_model)
        assert result[0].vector["dense"] == [0.1] * 768

    def test_no_sparse_model(self, sample_source_points):
        """When sparse model is None, sparse_indexed should be False."""
        points = transform_points(sample_source_points, None)

        assert len(points) == 10
        for p in points:
            assert "dense" in p.vector
            assert "sparse" not in p.vector
            assert p.payload["sparse_indexed"] is False


# ---------------------------------------------------------------------------
# Collection creation test
# ---------------------------------------------------------------------------


class TestCreateTargetCollection:
    @pytest.mark.asyncio
    async def test_creates_collection_with_indexes(self, mock_qdrant_client):
        await create_target_collection(mock_qdrant_client, "test_hybrid", 768)

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
        assert "dense" in call_kwargs["vectors_config"]
        assert "sparse" in call_kwargs["sparse_vectors_config"]

        # 4 payload indexes: tags, document_id, tenant_id, sparse_indexed
        assert mock_qdrant_client.create_payload_index.call_count == 4


# ---------------------------------------------------------------------------
# Verification gate tests
# ---------------------------------------------------------------------------


class TestGatePointCount:
    @pytest.mark.asyncio
    async def test_pass_on_equal_counts(self, mock_qdrant_client):
        mock_qdrant_client.get_collection.return_value = MagicMock(points_count=50)
        result = await gate_point_count(mock_qdrant_client, "src", "tgt")
        assert result.passed is True
        assert "source=50, target=50" in result.detail

    @pytest.mark.asyncio
    async def test_fail_on_mismatched_counts(self, mock_qdrant_client):
        mock_qdrant_client.get_collection.side_effect = [
            MagicMock(points_count=50),
            MagicMock(points_count=45),
        ]
        result = await gate_point_count(mock_qdrant_client, "src", "tgt")
        assert result.passed is False


class TestGatePayloadIntegrity:
    @pytest.mark.asyncio
    async def test_pass_on_matching_payloads(self, mock_qdrant_client):
        points = []
        for i in range(3):
            p = MagicMock()
            p.id = f"p-{i}"
            p.payload = {"chunk_text": f"chunk {i}"}
            points.append(p)

        mock_qdrant_client.scroll.return_value = (points, None)
        mock_qdrant_client.retrieve.return_value = points

        result = await gate_payload_integrity(mock_qdrant_client, "src", "tgt")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fail_on_missing_target_point(self, mock_qdrant_client):
        src_point = MagicMock()
        src_point.id = "p-0"
        src_point.payload = {"chunk_text": "hello"}

        mock_qdrant_client.scroll.return_value = ([src_point], None)
        mock_qdrant_client.retrieve.return_value = []  # missing from target

        result = await gate_payload_integrity(mock_qdrant_client, "src", "tgt")
        assert result.passed is False
        assert "mismatches=1" in result.detail

    @pytest.mark.asyncio
    async def test_pass_on_empty_collection(self, mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = ([], None)
        result = await gate_payload_integrity(mock_qdrant_client, "src", "tgt")
        assert result.passed is True


class TestGateSearchParity:
    @pytest.mark.asyncio
    async def test_pass_on_high_overlap(self, mock_qdrant_client):
        src_point = MagicMock()
        src_point.id = "q-0"
        src_point.vector = [0.1] * 768

        mock_qdrant_client.scroll.return_value = ([src_point], None)

        # Both return same IDs -> 100% overlap
        result_points = [MagicMock(id=f"r-{i}") for i in range(5)]
        mock_qdrant_client.query_points.return_value = MagicMock(points=result_points)

        result = await gate_search_parity(mock_qdrant_client, "src", "tgt")
        assert result.passed is True
        assert "100.00%" in result.detail

    @pytest.mark.asyncio
    async def test_fail_on_low_overlap(self, mock_qdrant_client):
        src_point = MagicMock()
        src_point.id = "q-0"
        src_point.vector = [0.1] * 768

        mock_qdrant_client.scroll.return_value = ([src_point], None)

        # Return different IDs for source and target -> 0% overlap
        src_results = [MagicMock(id=f"src-{i}") for i in range(5)]
        tgt_results = [MagicMock(id=f"tgt-{i}") for i in range(5)]
        mock_qdrant_client.query_points.side_effect = [
            MagicMock(points=src_results),
            MagicMock(points=tgt_results),
        ]

        result = await gate_search_parity(mock_qdrant_client, "src", "tgt")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_pass_on_empty(self, mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = ([], None)
        result = await gate_search_parity(mock_qdrant_client, "src", "tgt")
        assert result.passed is True


class TestGateLatency:
    @pytest.mark.asyncio
    async def test_pass_on_similar_latency(self, mock_qdrant_client):
        src_point = MagicMock()
        src_point.id = "q-0"
        src_point.vector = [0.1] * 768

        mock_qdrant_client.scroll.return_value = ([src_point], None)
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        result = await gate_latency(mock_qdrant_client, "src", "tgt")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_pass_on_empty(self, mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = ([], None)
        result = await gate_latency(mock_qdrant_client, "src", "tgt")
        assert result.passed is True


class TestGateSparsePresence:
    @pytest.mark.asyncio
    async def test_pass_when_all_have_sparse(self, mock_qdrant_client):
        points = []
        for i in range(5):
            p = MagicMock()
            p.id = f"p-{i}"
            p.vector = {
                "dense": [0.1] * 768,
                "sparse": models.SparseVector(indices=[0, 1], values=[0.5, 0.3]),
            }
            points.append(p)

        mock_qdrant_client.scroll.return_value = (points, None)
        result = await gate_sparse_presence(mock_qdrant_client, "tgt")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fail_when_sparse_missing(self, mock_qdrant_client):
        p = MagicMock()
        p.id = "p-0"
        p.vector = {"dense": [0.1] * 768}  # No sparse key

        mock_qdrant_client.scroll.return_value = ([p], None)
        result = await gate_sparse_presence(mock_qdrant_client, "tgt")
        assert result.passed is False
        assert "missing_sparse=1" in result.detail

    @pytest.mark.asyncio
    async def test_pass_on_empty(self, mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = ([], None)
        result = await gate_sparse_presence(mock_qdrant_client, "tgt")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Full verification gates test
# ---------------------------------------------------------------------------


class TestRunVerificationGates:
    @pytest.mark.asyncio
    async def test_all_pass_on_valid_migration(self, mock_qdrant_client):
        """All 5 gates pass when source and target are identical."""
        # Point count: equal
        mock_qdrant_client.get_collection.return_value = MagicMock(points_count=5)

        # Payload integrity + search parity + latency: use scroll
        points = []
        for i in range(5):
            p = MagicMock()
            p.id = f"p-{i}"
            p.vector = {
                "dense": [0.1 * i] * 768,
                "sparse": models.SparseVector(indices=[0], values=[0.5]),
            }
            p.payload = {"chunk_text": f"chunk {i}"}
            points.append(p)

        mock_qdrant_client.scroll.return_value = (points, None)
        mock_qdrant_client.retrieve.return_value = points

        # Search parity: same results for both
        result_points = [MagicMock(id=f"p-{i}") for i in range(5)]
        mock_qdrant_client.query_points.return_value = MagicMock(points=result_points)

        passed = await run_verification_gates(mock_qdrant_client, "src", "tgt")
        assert passed is True

    @pytest.mark.asyncio
    async def test_fail_on_count_mismatch(self, mock_qdrant_client):
        """Gates fail when point counts differ."""
        mock_qdrant_client.get_collection.side_effect = [
            MagicMock(points_count=50),
            MagicMock(points_count=45),
        ]
        mock_qdrant_client.scroll.return_value = ([], None)
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])

        passed = await run_verification_gates(mock_qdrant_client, "src", "tgt")
        assert passed is False


# ---------------------------------------------------------------------------
# Full migration flow test
# ---------------------------------------------------------------------------


class TestDoMigrate:
    @pytest.mark.asyncio
    async def test_full_flow(self, sample_source_points, tmp_path, mock_sparse_model):
        """End-to-end: scroll -> transform -> upsert."""
        client = AsyncMock()

        # Collections include source but not target
        client.get_collections.return_value = _collections("documents")
        client.get_collection.return_value = MagicMock(points_count=10)
        client.create_collection.return_value = None
        client.create_payload_index.return_value = None

        # First scroll returns points, second returns empty
        client.scroll.side_effect = [
            (sample_source_points, None),
        ]
        client.upsert.return_value = None

        cursor_file = str(tmp_path / ".migrate_cursor")

        with (
            patch("ai_ready_rag.cli.migrate_hybrid.AsyncQdrantClient", return_value=client),
            patch(
                "ai_ready_rag.cli.migrate_hybrid.get_sparse_model",
                return_value=mock_sparse_model,
            ),
            patch("ai_ready_rag.cli.migrate_hybrid.CURSOR_FILE", cursor_file),
        ):
            result = await do_migrate(
                "http://localhost:6333", "documents", "documents_hybrid", 100, 768, False
            )

        assert result is True
        client.upsert.assert_called_once()

        # Cursor should be cleared after successful migration
        assert not Path(cursor_file).exists()

    @pytest.mark.asyncio
    async def test_source_not_found(self):
        """Migration fails when source collection does not exist."""
        client = AsyncMock()
        client.get_collections.return_value = _collections("other")

        with patch("ai_ready_rag.cli.migrate_hybrid.AsyncQdrantClient", return_value=client):
            result = await do_migrate(
                "http://localhost:6333", "documents", "documents_hybrid", 100, 768, False
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_resume_from_cursor(self, tmp_path, mock_sparse_model):
        """Migration resumes from saved cursor offset."""
        cursor_file = str(tmp_path / ".migrate_cursor")
        cursor = MigrationCursor(offset="resume-offset", migrated=5, source_count=10)
        cursor.save(cursor_file)

        client = AsyncMock()
        client.get_collections.return_value = _collections("documents", "documents_hybrid")
        client.get_collection.return_value = MagicMock(points_count=10)

        # Return remaining 5 points
        remaining = []
        for i in range(5):
            p = MagicMock()
            p.id = f"point-{i + 5}"
            p.vector = [0.1] * 768
            p.payload = {"chunk_text": f"chunk {i + 5}"}
            remaining.append(p)

        client.scroll.side_effect = [
            (remaining, None),
        ]
        client.upsert.return_value = None

        with (
            patch("ai_ready_rag.cli.migrate_hybrid.AsyncQdrantClient", return_value=client),
            patch(
                "ai_ready_rag.cli.migrate_hybrid.get_sparse_model",
                return_value=mock_sparse_model,
            ),
            patch("ai_ready_rag.cli.migrate_hybrid.CURSOR_FILE", cursor_file),
        ):
            result = await do_migrate(
                "http://localhost:6333", "documents", "documents_hybrid", 100, 768, False
            )

        assert result is True

        # Scroll should have been called with the resume offset
        scroll_call = client.scroll.call_args
        assert scroll_call.kwargs.get("offset") == "resume-offset"
