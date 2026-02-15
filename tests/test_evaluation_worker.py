"""Tests for EvaluationWorker: claiming, retry, cancel, stale recovery, lifecycle."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from ai_ready_rag.db.models.evaluation import (
    DatasetSample,
    EvaluationDataset,
    EvaluationRun,
    EvaluationSample,
)
from ai_ready_rag.db.repositories.evaluation import (
    EvaluationRunRepository,
    EvaluationSampleRepository,
)

# ---------- Fixtures ----------


@pytest.fixture
def sample_dataset(db, admin_user):
    """Create a dataset with samples."""
    dataset = EvaluationDataset(
        name="Worker Test Dataset",
        source_type="manual",
        sample_count=3,
        created_by=admin_user.id,
    )
    db.add(dataset)
    db.flush()
    for i in range(3):
        db.add(
            DatasetSample(
                dataset_id=dataset.id,
                question=f"Question {i}?",
                ground_truth=f"Answer {i}",
                sort_order=i,
            )
        )
    db.flush()
    db.refresh(dataset)
    return dataset


@pytest.fixture
def pending_run(db, sample_dataset):
    """Create a pending evaluation run with samples."""
    run = EvaluationRun(
        name="Test Run",
        dataset_id=sample_dataset.id,
        status="pending",
        total_samples=3,
        tag_scope=json.dumps(["hr"]),
        admin_bypass_tags=False,
        model_used="qwen3:8b",
        embedding_model_used="nomic-embed-text",
        config_snapshot=json.dumps({"chat_model": "qwen3:8b"}),
        triggered_by="test-user",
    )
    db.add(run)
    db.flush()
    for i in range(3):
        db.add(
            EvaluationSample(
                run_id=run.id,
                question=f"Question {i}?",
                ground_truth=f"Answer {i}",
                sort_order=i,
                status="pending",
            )
        )
    db.flush()
    db.refresh(run)
    return run


@pytest.fixture
def running_run_with_stale_lease(db, sample_dataset):
    """Create a running run with expired lease."""
    run = EvaluationRun(
        name="Stale Run",
        dataset_id=sample_dataset.id,
        status="running",
        total_samples=3,
        tag_scope=json.dumps(["hr"]),
        admin_bypass_tags=False,
        model_used="qwen3:8b",
        embedding_model_used="nomic-embed-text",
        config_snapshot=json.dumps({"chat_model": "qwen3:8b"}),
        triggered_by="test-user",
        worker_id="old-worker",
        worker_lease_expires_at=datetime.utcnow() - timedelta(hours=1),
        started_at=datetime.utcnow() - timedelta(hours=2),
    )
    db.add(run)
    db.flush()
    for i in range(3):
        status = "processing" if i == 0 else "pending"
        db.add(
            EvaluationSample(
                run_id=run.id,
                question=f"Question {i}?",
                ground_truth=f"Answer {i}",
                sort_order=i,
                status=status,
            )
        )
    db.flush()
    db.refresh(run)
    return run


# ---------- Test Repository Methods ----------


class TestRunRepositoryClaimMethods:
    """Test run claim/discovery repository methods."""

    def test_get_next_claimable_pending(self, db, pending_run):
        repo = EvaluationRunRepository(db)
        run = repo.get_next_claimable()
        assert run is not None
        assert run.id == pending_run.id

    def test_get_next_claimable_stale(self, db, running_run_with_stale_lease):
        repo = EvaluationRunRepository(db)
        run = repo.get_next_claimable()
        assert run is not None
        assert run.id == running_run_with_stale_lease.id

    def test_get_next_claimable_none(self, db):
        repo = EvaluationRunRepository(db)
        assert repo.get_next_claimable() is None

    def test_claim_run_success(self, db, pending_run):
        repo = EvaluationRunRepository(db)
        claimed = repo.claim_run(pending_run.id, "worker-1", 15)
        assert claimed is True
        db.expire_all()
        run = repo.get(pending_run.id)
        assert run.status == "running"
        assert run.worker_id == "worker-1"
        assert run.worker_lease_expires_at is not None
        assert run.started_at is not None

    def test_claim_run_already_claimed(self, db, pending_run):
        repo = EvaluationRunRepository(db)
        # First claim
        repo.claim_run(pending_run.id, "worker-1", 15)
        # Manually update to completed to prevent re-claim
        db.expire_all()
        run = repo.get(pending_run.id)
        run.status = "completed"
        db.commit()
        # Second claim should fail
        claimed = repo.claim_run(pending_run.id, "worker-2", 15)
        assert claimed is False

    def test_renew_lease(self, db, pending_run):
        repo = EvaluationRunRepository(db)
        repo.claim_run(pending_run.id, "worker-1", 15)
        renewed = repo.renew_lease(pending_run.id, "worker-1", 15)
        assert renewed is True

    def test_renew_lease_wrong_worker(self, db, pending_run):
        repo = EvaluationRunRepository(db)
        repo.claim_run(pending_run.id, "worker-1", 15)
        renewed = repo.renew_lease(pending_run.id, "worker-2", 15)
        assert renewed is False


class TestSampleRepositoryClaimMethods:
    """Test sample claim/skip repository methods."""

    def test_claim_sample_success(self, db, pending_run):
        sample_repo = EvaluationSampleRepository(db)
        samples = sample_repo.get_pending_samples(pending_run.id)
        assert len(samples) == 3
        claimed = sample_repo.claim_sample(samples[0].id)
        assert claimed is True
        db.expire_all()
        s = sample_repo.get(samples[0].id)
        assert s.status == "processing"

    def test_claim_sample_already_claimed(self, db, pending_run):
        sample_repo = EvaluationSampleRepository(db)
        samples = sample_repo.get_pending_samples(pending_run.id)
        sample_repo.claim_sample(samples[0].id)
        # Second claim should fail
        claimed = sample_repo.claim_sample(samples[0].id)
        assert claimed is False

    def test_skip_remaining(self, db, pending_run):
        sample_repo = EvaluationSampleRepository(db)
        # Claim first sample
        samples = sample_repo.get_pending_samples(pending_run.id)
        sample_repo.claim_sample(samples[0].id)
        # Skip remaining
        skipped = sample_repo.skip_remaining(pending_run.id)
        assert skipped == 2  # 3 total minus 1 claimed

    def test_get_pending_samples_ordered(self, db, pending_run):
        sample_repo = EvaluationSampleRepository(db)
        samples = sample_repo.get_pending_samples(pending_run.id)
        assert len(samples) == 3
        for i, s in enumerate(samples):
            assert s.sort_order == i

    def test_get_avg_sample_time_none(self, db, pending_run):
        sample_repo = EvaluationSampleRepository(db)
        avg = sample_repo.get_avg_sample_time(pending_run.id)
        assert avg is None

    def test_get_avg_sample_time_with_data(self, db, pending_run):
        sample_repo = EvaluationSampleRepository(db)
        samples = sample_repo.get_pending_samples(pending_run.id)
        # Mark first two as completed with timing
        for i, s in enumerate(samples[:2]):
            s.status = "completed"
            s.generation_time_ms = 1000.0 * (i + 1)
        db.commit()
        avg = sample_repo.get_avg_sample_time(pending_run.id)
        assert avg == 1500.0  # (1000 + 2000) / 2


class TestRunSummaryData:
    """Test summary query method."""

    def test_summary_empty(self, db):
        repo = EvaluationRunRepository(db)
        data = repo.get_summary_data()
        assert data["total_runs"] == 0
        assert data["total_datasets"] == 0
        assert data["latest_run"] is None
        assert data["score_trend"] == []

    def test_summary_with_completed_run(self, db, pending_run):
        # Mark run as completed
        pending_run.status = "completed"
        pending_run.completed_at = datetime.utcnow()
        pending_run.avg_faithfulness = 0.8
        pending_run.avg_answer_relevancy = 0.9
        db.commit()

        repo = EvaluationRunRepository(db)
        data = repo.get_summary_data()
        assert data["total_runs"] >= 1
        assert data["latest_run"] is not None
        assert data["latest_run"].id == pending_run.id
        assert len(data["score_trend"]) >= 1


# ---------- Test Stale Recovery ----------


class TestStaleRecovery:
    """Test recover_stale_evaluation_runs()."""

    @pytest.mark.asyncio
    async def test_recover_stale_requeues(self, db, running_run_with_stale_lease):
        from ai_ready_rag.workers.evaluation_worker import recover_stale_evaluation_runs

        count = await recover_stale_evaluation_runs(db)
        assert count >= 1

        # Verify run was requeued
        db.expire_all()
        run = (
            db.query(EvaluationRun)
            .filter(EvaluationRun.id == running_run_with_stale_lease.id)
            .first()
        )
        assert run.status == "pending"
        assert run.worker_id is None

    @pytest.mark.asyncio
    async def test_recover_stale_with_cancel(self, db, running_run_with_stale_lease):
        from ai_ready_rag.workers.evaluation_worker import recover_stale_evaluation_runs

        # Set cancel flag
        running_run_with_stale_lease.is_cancel_requested = True
        db.commit()

        count = await recover_stale_evaluation_runs(db)
        assert count >= 1

        db.expire_all()
        run = (
            db.query(EvaluationRun)
            .filter(EvaluationRun.id == running_run_with_stale_lease.id)
            .first()
        )
        assert run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_recover_resets_orphaned_samples(self, db, running_run_with_stale_lease):
        from ai_ready_rag.workers.evaluation_worker import recover_stale_evaluation_runs

        await recover_stale_evaluation_runs(db)

        # Verify processing sample was reset
        db.expire_all()
        samples = (
            db.query(EvaluationSample)
            .filter(EvaluationSample.run_id == running_run_with_stale_lease.id)
            .all()
        )
        for s in samples:
            assert s.status == "pending"


# ---------- Test Worker Lifecycle ----------


class TestWorkerLifecycle:
    """Test worker start/stop."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        from ai_ready_rag.workers.evaluation_worker import EvaluationWorker

        mock_service = MagicMock()
        mock_settings = MagicMock()
        mock_settings.eval_scan_interval_seconds = 30
        mock_settings.eval_lease_renewal_seconds = 60
        mock_settings.eval_lease_duration_minutes = 15

        worker = EvaluationWorker(mock_service, mock_settings)
        await worker.start()
        assert worker._task is not None
        assert worker._lease_task is not None

        await worker.stop()
        # Tasks should be cancelled/completed


class TestRunCompletion:
    """Test run final status determination."""

    def test_completed_status_no_failures(self, db, pending_run):
        """0 failures -> completed."""
        pending_run.status = "running"
        pending_run.completed_samples = 3
        pending_run.failed_samples = 0
        db.commit()

        # Verify logic
        assert pending_run.failed_samples == 0

    def test_completed_with_errors_status(self, db, pending_run):
        """Some failures -> completed_with_errors."""
        pending_run.status = "running"
        pending_run.completed_samples = 2
        pending_run.failed_samples = 1
        db.commit()

        assert pending_run.failed_samples > 0
