"""Evaluation repository for datasets, samples, and runs."""

import json
import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_ready_rag.db.models.evaluation import (
    DatasetSample,
    EvaluationDataset,
    EvaluationRun,
    EvaluationSample,
)
from ai_ready_rag.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class EvaluationDatasetRepository(BaseRepository[EvaluationDataset]):
    model = EvaluationDataset

    def get_by_name(self, name: str) -> EvaluationDataset | None:
        """Get dataset by unique name."""
        results = self.list_by(name=name)
        return results[0] if results else None

    def list_paginated(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[EvaluationDataset], int]:
        """List datasets with pagination."""
        total = self.count()
        stmt = (
            select(EvaluationDataset)
            .order_by(EvaluationDataset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        datasets = list(self.db.scalars(stmt).all())
        return datasets, total

    def has_active_runs(self, dataset_id: str) -> bool:
        """Check if any evaluation runs reference this dataset."""
        return (
            self.db.scalar(
                select(func.count())
                .select_from(EvaluationRun)
                .where(EvaluationRun.dataset_id == dataset_id)
            )
            or 0
        ) > 0


class DatasetSampleRepository(BaseRepository[DatasetSample]):
    model = DatasetSample

    def list_all_by_dataset(self, dataset_id: str) -> list[DatasetSample]:
        """Return all samples for a dataset (no pagination)."""
        stmt = (
            select(DatasetSample)
            .where(DatasetSample.dataset_id == dataset_id)
            .order_by(DatasetSample.sort_order)
        )
        return list(self.db.scalars(stmt).all())

    def list_by_dataset(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DatasetSample], int]:
        """List samples for a dataset with pagination."""
        total = self.count(dataset_id=dataset_id)
        stmt = (
            select(DatasetSample)
            .where(DatasetSample.dataset_id == dataset_id)
            .order_by(DatasetSample.sort_order)
            .limit(limit)
            .offset(offset)
        )
        samples = list(self.db.scalars(stmt).all())
        return samples, total

    def bulk_create(
        self,
        dataset_id: str,
        samples_data: list[dict],
    ) -> list[DatasetSample]:
        """Create multiple samples with sort_order assignment.

        Returns the created sample objects. Does NOT flush/commit --
        caller is responsible for transaction boundaries.
        """
        samples = []
        for i, data in enumerate(samples_data):
            # Ground-truth normalization: strip whitespace, empty -> None
            ground_truth = data.get("ground_truth")
            if ground_truth is not None:
                ground_truth = ground_truth.strip()
                if not ground_truth:
                    ground_truth = None

            reference_contexts = data.get("reference_contexts")
            metadata = data.get("metadata")

            sample = DatasetSample(
                dataset_id=dataset_id,
                question=data["question"],
                ground_truth=ground_truth,
                reference_contexts=json.dumps(reference_contexts) if reference_contexts else None,
                metadata_=json.dumps(metadata) if metadata else None,
                sort_order=i,
            )
            samples.append(sample)

        self.add_all(samples)
        return samples


class EvaluationRunRepository(BaseRepository[EvaluationRun]):
    model = EvaluationRun

    def list_paginated(
        self,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[EvaluationRun], int]:
        """List runs with optional status filter and pagination."""
        total = self.count() if status is None else self.count(status=status)
        stmt = select(EvaluationRun)
        if status:
            stmt = stmt.where(EvaluationRun.status == status)
        stmt = stmt.order_by(EvaluationRun.created_at.desc()).limit(limit).offset(offset)
        runs = list(self.db.scalars(stmt).all())
        return runs, total

    def get_next_claimable(self) -> EvaluationRun | None:
        """Find oldest pending or stale-lease run."""
        now = datetime.utcnow()
        stmt = (
            select(EvaluationRun)
            .where(
                (EvaluationRun.status == "pending")
                | (
                    (EvaluationRun.status == "running")
                    & (EvaluationRun.worker_lease_expires_at < now)
                )
            )
            .order_by(EvaluationRun.created_at.asc())
            .limit(1)
        )
        return self.db.scalars(stmt).first()

    def claim_run(
        self,
        run_id: str,
        worker_id: str,
        lease_duration_minutes: int,
    ) -> bool:
        """Atomically claim a run via UPDATE WHERE."""
        now = datetime.utcnow()
        lease_expiry = now + timedelta(minutes=lease_duration_minutes)
        updated = (
            self.db.query(EvaluationRun)
            .filter(
                EvaluationRun.id == run_id,
                EvaluationRun.status.in_(["pending", "running"]),
            )
            .update(
                {
                    "status": "running",
                    "worker_id": worker_id,
                    "worker_lease_expires_at": lease_expiry,
                    "started_at": now,
                    "updated_at": now,
                }
            )
        )
        self.db.commit()
        return updated > 0

    def renew_lease(
        self,
        run_id: str,
        worker_id: str,
        lease_duration_minutes: int,
    ) -> bool:
        """Extend lease expiry for active run."""
        now = datetime.utcnow()
        new_expiry = now + timedelta(minutes=lease_duration_minutes)
        updated = (
            self.db.query(EvaluationRun)
            .filter(
                EvaluationRun.id == run_id,
                EvaluationRun.worker_id == worker_id,
                EvaluationRun.status == "running",
            )
            .update({"worker_lease_expires_at": new_expiry})
        )
        self.db.commit()
        return updated > 0

    def get_summary_data(self) -> dict:
        """Get summary data for dashboard."""
        total_runs = self.db.scalar(select(func.count()).select_from(EvaluationRun)) or 0
        total_datasets = self.db.scalar(select(func.count()).select_from(EvaluationDataset)) or 0

        # Latest completed run
        latest_run = self.db.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.status.in_(["completed", "completed_with_errors"]))
            .order_by(EvaluationRun.completed_at.desc())
            .limit(1)
        ).first()

        # Average scores across all completed runs
        avg_scores = {}
        for metric in [
            "avg_faithfulness",
            "avg_answer_relevancy",
            "avg_llm_context_precision",
            "avg_llm_context_recall",
        ]:
            col = getattr(EvaluationRun, metric)
            avg = self.db.scalar(
                select(func.avg(col)).where(
                    EvaluationRun.status.in_(["completed", "completed_with_errors"])
                )
            )
            key = metric.replace("avg_", "")
            avg_scores[key] = float(avg) if avg is not None else None

        # Score trend (last 10 completed runs)
        trend_runs = list(
            self.db.scalars(
                select(EvaluationRun)
                .where(EvaluationRun.status.in_(["completed", "completed_with_errors"]))
                .order_by(EvaluationRun.completed_at.desc())
                .limit(10)
            ).all()
        )
        score_trend = [
            {
                "run_id": r.id,
                "name": r.name,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "avg_faithfulness": r.avg_faithfulness,
                "avg_answer_relevancy": r.avg_answer_relevancy,
            }
            for r in reversed(trend_runs)
        ]

        return {
            "latest_run": latest_run,
            "total_runs": total_runs,
            "total_datasets": total_datasets,
            "avg_scores": avg_scores,
            "score_trend": score_trend,
        }


class EvaluationSampleRepository(BaseRepository[EvaluationSample]):
    model = EvaluationSample

    def list_by_run(
        self,
        run_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[EvaluationSample], int]:
        """List samples for a run with optional status filter."""
        count_stmt = (
            select(func.count())
            .select_from(EvaluationSample)
            .where(EvaluationSample.run_id == run_id)
        )
        if status:
            count_stmt = count_stmt.where(EvaluationSample.status == status)
        total = self.db.scalar(count_stmt) or 0

        stmt = select(EvaluationSample).where(EvaluationSample.run_id == run_id)
        if status:
            stmt = stmt.where(EvaluationSample.status == status)
        stmt = stmt.order_by(EvaluationSample.sort_order).limit(limit).offset(offset)
        samples = list(self.db.scalars(stmt).all())
        return samples, total

    def bulk_create_from_dataset(
        self,
        run_id: str,
        dataset_samples: list[DatasetSample],
    ) -> list[EvaluationSample]:
        """Create evaluation samples from dataset samples.

        Copies question, ground_truth, reference_contexts, sort_order.
        Does NOT flush/commit -- caller owns transaction.
        """
        eval_samples = []
        for ds in dataset_samples:
            sample = EvaluationSample(
                run_id=run_id,
                question=ds.question,
                ground_truth=ds.ground_truth,
                reference_contexts=ds.reference_contexts,
                sort_order=ds.sort_order,
                status="pending",
            )
            eval_samples.append(sample)
        self.add_all(eval_samples)
        return eval_samples

    def get_aggregate_scores(self, run_id: str) -> dict[str, float | None]:
        """Compute SQL AVG() for each metric column on completed samples."""
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision",
            "llm_context_recall",
        ]
        result = {}
        for metric in metrics:
            col = getattr(EvaluationSample, metric)
            avg = self.db.scalar(
                select(func.avg(col))
                .where(EvaluationSample.run_id == run_id)
                .where(EvaluationSample.status == "completed")
            )
            result[metric] = float(avg) if avg is not None else None
        return result

    def count_null_metrics(self, run_id: str) -> int:
        """Count NULL metric cells across completed samples."""
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision",
            "llm_context_recall",
        ]
        total_nulls = 0
        for metric in metrics:
            col = getattr(EvaluationSample, metric)
            count = (
                self.db.scalar(
                    select(func.count())
                    .select_from(EvaluationSample)
                    .where(EvaluationSample.run_id == run_id)
                    .where(EvaluationSample.status == "completed")
                    .where(col.is_(None))
                )
                or 0
            )
            total_nulls += count
        return total_nulls

    def claim_sample(self, sample_id: str) -> bool:
        """Atomically claim a sample via UPDATE WHERE status='pending'."""
        updated = (
            self.db.query(EvaluationSample)
            .filter(
                EvaluationSample.id == sample_id,
                EvaluationSample.status == "pending",
            )
            .update({"status": "processing"})
        )
        self.db.commit()
        return updated > 0

    def skip_remaining(self, run_id: str) -> int:
        """Bulk update pending samples to skipped for a run."""
        now = datetime.utcnow()
        count = (
            self.db.query(EvaluationSample)
            .filter(
                EvaluationSample.run_id == run_id,
                EvaluationSample.status == "pending",
            )
            .update({"status": "skipped", "processed_at": now})
        )
        self.db.commit()
        return count

    def get_pending_samples(self, run_id: str) -> list[EvaluationSample]:
        """Get all pending samples for a run, ordered by sort_order."""
        stmt = (
            select(EvaluationSample)
            .where(
                EvaluationSample.run_id == run_id,
                EvaluationSample.status == "pending",
            )
            .order_by(EvaluationSample.sort_order)
        )
        return list(self.db.scalars(stmt).all())

    def get_avg_sample_time(self, run_id: str) -> float | None:
        """Get average generation_time_ms for completed samples in a run."""
        avg = self.db.scalar(
            select(func.avg(EvaluationSample.generation_time_ms))
            .where(EvaluationSample.run_id == run_id)
            .where(EvaluationSample.status == "completed")
        )
        return float(avg) if avg is not None else None


def recompute_dataset_sample_counts(db: Session) -> int:
    """Recompute all dataset sample_count values from actual row counts.

    Returns number of datasets updated.
    """
    datasets = db.scalars(select(EvaluationDataset)).all()
    updated = 0
    for dataset in datasets:
        actual = (
            db.scalar(
                select(func.count())
                .select_from(DatasetSample)
                .where(DatasetSample.dataset_id == dataset.id)
            )
            or 0
        )
        if dataset.sample_count != actual:
            dataset.sample_count = actual
            updated += 1
    return updated
