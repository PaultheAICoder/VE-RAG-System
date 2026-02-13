"""Evaluation repository for datasets, samples, and runs."""

import json
import logging

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
