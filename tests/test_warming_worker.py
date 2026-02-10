"""Tests for WarmingWorker background service (DB-first architecture)."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.workers.warming_worker import WarmingWorker, recover_stale_batches


class TestWarmingWorkerInit:
    """Test worker initialization."""

    def test_creates_unique_worker_id(self):
        """Each worker gets unique ID."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_lease_renewal_seconds = 60

        worker1 = WarmingWorker(mock_rag, mock_settings)
        worker2 = WarmingWorker(mock_rag, mock_settings)

        assert worker1.worker_id != worker2.worker_id
        assert worker1.worker_id.startswith("worker-")
        assert worker2.worker_id.startswith("worker-")

    def test_starts_with_no_current_batch(self):
        """Initial state has no batch."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()

        worker = WarmingWorker(mock_rag, mock_settings)

        assert worker._current_batch_id is None
        assert worker._task is None
        assert worker._lease_task is None


class TestWarmingWorkerLifecycle:
    """Test start/stop behavior."""

    @pytest.mark.asyncio
    async def test_start_creates_tasks(self):
        """Start creates run and lease tasks."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_lease_renewal_seconds = 60
        mock_settings.warming_scan_interval_seconds = 5

        worker = WarmingWorker(mock_rag, mock_settings)

        with patch.object(worker, "_run_loop", new_callable=AsyncMock):
            with patch.object(worker, "_lease_renewal_loop", new_callable=AsyncMock):
                await worker.start()

                assert worker._task is not None
                assert worker._lease_task is not None
                assert not worker._shutdown.is_set()

                await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """Stop cancels and awaits tasks."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_lease_renewal_seconds = 60

        worker = WarmingWorker(mock_rag, mock_settings)

        async def slow_task():
            await asyncio.sleep(100)

        worker._task = asyncio.create_task(slow_task())
        worker._lease_task = asyncio.create_task(slow_task())

        await worker.stop()

        assert worker._shutdown.is_set()
        assert worker._task.cancelled() or worker._task.done()
        assert worker._lease_task.cancelled() or worker._lease_task.done()


class TestRecoverStaleBatches:
    """Test stale batch recovery."""

    @pytest.mark.asyncio
    async def test_recover_expired_leases(self, db):
        """Reset batches with expired leases."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="old-worker",
            worker_lease_expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        db.add(batch)
        db.commit()

        batch_id = batch.id

        with patch("ai_ready_rag.workers.warming_worker.SessionLocal", return_value=db):
            count = await recover_stale_batches()

        updated = db.query(WarmingBatch).filter(WarmingBatch.id == batch_id).first()

        assert count == 1
        assert updated.status == "pending"
        assert updated.worker_id is None
        assert updated.worker_lease_expires_at is None

    @pytest.mark.asyncio
    async def test_does_not_recover_active_leases(self, db):
        """Active leases are not recovered."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="active-worker",
            worker_lease_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(batch)
        db.commit()

        with patch("ai_ready_rag.workers.warming_worker.SessionLocal", return_value=db):
            count = await recover_stale_batches()

        assert count == 0

    @pytest.mark.asyncio
    async def test_does_not_recover_pending_batches(self, db):
        """Pending batches are not affected."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="pending",
        )
        db.add(batch)
        db.commit()

        with patch("ai_ready_rag.workers.warming_worker.SessionLocal", return_value=db):
            count = await recover_stale_batches()

        assert count == 0

    @pytest.mark.asyncio
    async def test_recover_resets_orphaned_processing_queries(self, db):
        """Orphaned processing queries are reset to pending."""
        from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery

        batch = WarmingBatch(
            source_type="manual",
            total_queries=3,
            status="running",
            worker_id="dead-worker",
            worker_lease_expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        db.add(batch)
        db.flush()

        # One completed, one stuck processing, one pending
        q1 = WarmingQuery(batch_id=batch.id, query_text="Q1", sort_order=0, status="completed")
        q2 = WarmingQuery(batch_id=batch.id, query_text="Q2", sort_order=1, status="processing")
        q3 = WarmingQuery(batch_id=batch.id, query_text="Q3", sort_order=2, status="pending")
        db.add_all([q1, q2, q3])
        db.commit()

        q2_id = q2.id
        q1_id = q1.id

        with patch("ai_ready_rag.workers.warming_worker.SessionLocal", return_value=db):
            count = await recover_stale_batches()

        assert count == 1  # 1 batch recovered

        # processing query should now be pending
        db.expire_all()
        q2_updated = db.query(WarmingQuery).filter(WarmingQuery.id == q2_id).first()
        assert q2_updated.status == "pending"

        # completed query should be unchanged
        q1_updated = db.query(WarmingQuery).filter(WarmingQuery.id == q1_id).first()
        assert q1_updated.status == "completed"


class TestFindPendingBatch:
    """Test batch discovery."""

    def test_finds_pending_batch(self, db):
        """Worker finds oldest pending batch."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        mock_rag = MagicMock()
        mock_settings = MagicMock()

        worker = WarmingWorker(mock_rag, mock_settings)

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="pending",
        )
        db.add(batch)
        db.commit()

        found_id = worker._find_pending_batch(db)

        assert found_id == batch.id

    def test_finds_stale_lease_batch(self, db):
        """Worker finds batches with expired leases."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        mock_rag = MagicMock()
        mock_settings = MagicMock()

        worker = WarmingWorker(mock_rag, mock_settings)

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="dead-worker",
            worker_lease_expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        db.add(batch)
        db.commit()

        found_id = worker._find_pending_batch(db)

        assert found_id == batch.id

    def test_skips_active_lease_batch(self, db):
        """Worker skips batches with active leases."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        mock_rag = MagicMock()
        mock_settings = MagicMock()

        worker = WarmingWorker(mock_rag, mock_settings)

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="active-worker",
            worker_lease_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(batch)
        db.commit()

        found_id = worker._find_pending_batch(db)

        assert found_id is None

    def test_returns_none_when_no_batches(self, db):
        """Returns None when no batches available."""
        mock_rag = MagicMock()
        mock_settings = MagicMock()

        worker = WarmingWorker(mock_rag, mock_settings)

        found_id = worker._find_pending_batch(db)

        assert found_id is None


class TestLeaseRenewal:
    """Test lease renewal."""

    def test_renews_owned_batch_lease(self, db):
        """Lease renewal extends expiry for owned batch."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        original_expiry = datetime.utcnow() + timedelta(minutes=1)
        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id=worker.worker_id,
            worker_lease_expires_at=original_expiry,
        )
        db.add(batch)
        db.commit()

        renewed = worker._renew_batch_lease(db, batch.id)
        db.refresh(batch)

        assert renewed is True
        assert batch.worker_lease_expires_at > original_expiry

    def test_does_not_renew_other_workers_lease(self, db):
        """Cannot renew lease owned by another worker."""
        from ai_ready_rag.db.models.warming import WarmingBatch

        mock_rag = MagicMock()
        mock_settings = MagicMock()
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        batch = WarmingBatch(
            source_type="manual",
            total_queries=10,
            status="running",
            worker_id="other-worker",
            worker_lease_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(batch)
        db.commit()

        renewed = worker._renew_batch_lease(db, batch.id)

        assert renewed is False


class TestProcessBatch:
    """Test batch processing."""

    @pytest.mark.asyncio
    async def test_processes_all_queries(self, db):
        """Worker processes all queries in a batch via shared helpers."""
        from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery

        mock_rag = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.warming_delay_seconds = 0
        mock_settings.warming_retry_delays = "1,2,3"
        mock_settings.warming_max_retries = 1
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        batch = WarmingBatch(
            source_type="manual",
            total_queries=2,
            status="running",
            worker_id=worker.worker_id,
            worker_lease_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(batch)
        db.flush()

        q1 = WarmingQuery(batch_id=batch.id, query_text="test query 1", sort_order=0)
        q2 = WarmingQuery(batch_id=batch.id, query_text="test query 2", sort_order=1)
        db.add_all([q1, q2])
        db.commit()

        # Create a mock SessionLocal that returns our db but prevents close()
        # from detaching objects (since worker calls close() in finally blocks)
        mock_session_local = MagicMock()
        mock_db = MagicMock(wraps=db)
        mock_db.close = MagicMock()  # Prevent actual close
        mock_db.query = db.query
        mock_db.execute = db.execute
        mock_db.commit = db.commit
        mock_db.expire_all = db.expire_all
        mock_db.refresh = db.refresh
        mock_session_local.return_value = mock_db

        with patch("ai_ready_rag.workers.warming_worker.SessionLocal", mock_session_local):
            with patch(
                "ai_ready_rag.services.rag_service.RAGRequest",
                MagicMock(),
            ):
                mock_rag.generate = AsyncMock(return_value=MagicMock())
                await worker._process_batch(batch.id)

        db.expire_all()
        updated_batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch.id).first()

        assert updated_batch.status in ("completed", "completed_with_errors")

    @pytest.mark.asyncio
    async def test_handles_cancel_during_processing(self, db):
        """Worker cancels batch when is_cancel_requested is set."""
        from ai_ready_rag.db.models.warming import WarmingBatch, WarmingQuery

        mock_rag = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.warming_delay_seconds = 0
        mock_settings.warming_retry_delays = "1"
        mock_settings.warming_max_retries = 0
        mock_settings.warming_lease_duration_minutes = 10

        worker = WarmingWorker(mock_rag, mock_settings)

        batch = WarmingBatch(
            source_type="manual",
            total_queries=2,
            status="running",
            is_cancel_requested=True,
            worker_id=worker.worker_id,
            worker_lease_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(batch)
        db.flush()

        q1 = WarmingQuery(batch_id=batch.id, query_text="test query 1", sort_order=0)
        db.add(q1)
        db.commit()

        # Create a mock SessionLocal that returns our db but prevents close()
        mock_session_local = MagicMock()
        mock_db = MagicMock(wraps=db)
        mock_db.close = MagicMock()  # Prevent actual close
        mock_db.query = db.query
        mock_db.execute = db.execute
        mock_db.commit = db.commit
        mock_db.expire_all = db.expire_all
        mock_session_local.return_value = mock_db

        with patch("ai_ready_rag.workers.warming_worker.SessionLocal", mock_session_local):
            await worker._process_batch(batch.id)

        db.expire_all()
        updated_batch = db.query(WarmingBatch).filter(WarmingBatch.id == batch.id).first()

        assert updated_batch.status == "cancelled"

        # Pending queries should be skipped
        skipped = (
            db.query(WarmingQuery)
            .filter(WarmingQuery.batch_id == batch.id, WarmingQuery.status == "skipped")
            .count()
        )
        assert skipped >= 1
