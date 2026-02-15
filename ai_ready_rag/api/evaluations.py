"""Evaluation framework API endpoints."""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai_ready_rag.config import get_settings
from ai_ready_rag.core.dependencies import require_admin
from ai_ready_rag.db.database import get_db
from ai_ready_rag.db.models import User
from ai_ready_rag.db.models.evaluation import EvaluationDataset, LiveEvaluationScore
from ai_ready_rag.db.repositories.evaluation import (
    DatasetSampleRepository,
    EvaluationDatasetRepository,
    EvaluationRunRepository,
    EvaluationSampleRepository,
    recompute_dataset_sample_counts,
)
from ai_ready_rag.schemas.evaluation import (
    CancelRunResponse,
    DatasetCreate,
    DatasetListResponse,
    DatasetResponse,
    DatasetSampleListResponse,
    EvaluationSampleListResponse,
    EvaluationSummaryResponse,
    LiveScoreListResponse,
    LiveScoreResponse,
    LiveStatsHourly,
    LiveStatsResponse,
    RAGBenchImportRequest,
    RunCreate,
    RunListResponse,
    RunResponse,
    SyntheticGenerateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    dependencies=[Depends(require_admin)],  # All endpoints admin-only
)


# ========== Dataset CRUD ==========


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List evaluation datasets with pagination."""
    repo = EvaluationDatasetRepository(db)
    datasets, total = repo.list_paginated(limit=limit, offset=offset)
    return DatasetListResponse(
        datasets=[DatasetResponse.model_validate(d) for d in datasets],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/datasets", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    request: DatasetCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a manual dataset with inline samples."""
    dataset_repo = EvaluationDatasetRepository(db)
    sample_repo = DatasetSampleRepository(db)

    # Check name uniqueness
    if dataset_repo.get_by_name(request.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset with name '{request.name}' already exists",
        )

    # Create dataset
    dataset = EvaluationDataset(
        name=request.name,
        description=request.description,
        source_type="manual",
        sample_count=len(request.samples),
        created_by=current_user.id,
    )
    dataset_repo.add(dataset)
    dataset_repo.flush()  # Get dataset.id for FK

    # Create samples (with ground-truth normalization)
    samples_data = [s.model_dump() for s in request.samples]
    sample_repo.bulk_create(dataset.id, samples_data)

    # Count samples missing ground truth for warning
    null_gt_count = sum(
        1
        for s in request.samples
        if s.ground_truth is None
        or (isinstance(s.ground_truth, str) and not s.ground_truth.strip())
    )

    db.commit()
    db.refresh(dataset)

    response = DatasetResponse.model_validate(dataset)
    if null_gt_count > 0:
        response.warning = (
            f"{null_gt_count} samples have no ground_truth "
            "-- retrieval metrics will be skipped for these"
        )
    return response


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Get a single dataset by ID."""
    repo = EvaluationDatasetRepository(db)
    dataset = repo.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)


@router.get("/datasets/{dataset_id}/samples", response_model=DatasetSampleListResponse)
async def list_dataset_samples(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List samples in a dataset with pagination."""
    dataset_repo = EvaluationDatasetRepository(db)
    if not dataset_repo.get(dataset_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    sample_repo = DatasetSampleRepository(db)
    samples, total = sample_repo.list_by_dataset(dataset_id, limit=limit, offset=offset)
    return DatasetSampleListResponse(
        samples=samples,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Delete dataset and all its samples.

    Fails with 409 if any evaluation runs reference this dataset.
    """
    repo = EvaluationDatasetRepository(db)
    dataset = repo.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    if repo.has_active_runs(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete dataset with existing evaluation runs. Delete the runs first.",
        )

    repo.delete(dataset)
    db.commit()
    return None


@router.post("/datasets/recompute-counts", status_code=status.HTTP_200_OK)
async def recompute_counts(
    db: Session = Depends(get_db),
):
    """Recompute all dataset sample_count values (admin-only, idempotent)."""
    updated = recompute_dataset_sample_counts(db)
    db.commit()
    return {"updated": updated}


@router.post(
    "/datasets/import-ragbench",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def import_ragbench(
    request: RAGBenchImportRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Import a RAGBench subset from pre-downloaded parquet files."""
    service = _get_evaluation_service(db)
    dataset = await service.import_ragbench(db, request, current_user.id)
    return DatasetResponse.model_validate(dataset)


@router.post(
    "/datasets/generate-synthetic",
    response_model=DatasetResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_synthetic(
    request: SyntheticGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Generate a synthetic dataset from uploaded documents.

    Returns 202 Accepted immediately. Generation runs in background.
    """
    service = _get_evaluation_service(db)
    dataset = await service.generate_synthetic(db, request, current_user.id)
    background_tasks.add_task(
        service.run_synthetic_generation,
        dataset.id,
        request.document_ids,
        request.num_samples,
    )
    return DatasetResponse.model_validate(dataset)


# ========== Run Endpoints ==========


def _get_evaluation_service(db: Session):
    """Construct EvaluationService for route handlers."""
    from ai_ready_rag.services.evaluation_service import EvaluationService
    from ai_ready_rag.services.rag_service import RAGService

    settings = get_settings()
    rag_service = RAGService(settings)
    return EvaluationService(settings, rag_service)


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List evaluation runs with optional status filter."""
    repo = EvaluationRunRepository(db)
    runs, total = repo.list_paginated(status=status_filter, limit=limit, offset=offset)
    return RunListResponse(runs=runs, total=total, limit=limit, offset=offset)


@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    request: RunCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create an evaluation run from a dataset."""
    service = _get_evaluation_service(db)
    run = await service.create_run(db, request, triggered_by=current_user.id)
    return RunResponse.model_validate(run)


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Get evaluation run details with ETA for running runs."""
    repo = EvaluationRunRepository(db)
    run = repo.get(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    response = RunResponse.model_validate(run)

    # Compute ETA for running runs
    if run.status == "running" and run.completed_samples > 0:
        sample_repo = EvaluationSampleRepository(db)
        avg_ms = sample_repo.get_avg_sample_time(run_id)
        if avg_ms is not None:
            remaining = run.total_samples - run.completed_samples - run.failed_samples
            response.eta_seconds = (remaining * avg_ms) / 1000.0

    return response


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Delete an evaluation run. Cannot delete active runs."""
    repo = EvaluationRunRepository(db)
    run = repo.get(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    if run.status in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete an active run. Cancel it first.",
        )
    repo.delete(run)
    db.commit()
    return None


@router.post("/runs/{run_id}/cancel", response_model=CancelRunResponse)
async def cancel_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Request cancellation of a running evaluation."""
    repo = EvaluationRunRepository(db)
    run = repo.get(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    if run.status in ("completed", "completed_with_errors", "failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel a run with status '{run.status}'",
        )
    run.is_cancel_requested = True
    db.commit()
    return CancelRunResponse(
        id=run.id,
        status=run.status,
        is_cancel_requested=True,
    )


@router.get("/runs/{run_id}/samples", response_model=EvaluationSampleListResponse)
async def list_run_samples(
    run_id: str,
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List samples for an evaluation run."""
    run_repo = EvaluationRunRepository(db)
    if not run_repo.get(run_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    sample_repo = EvaluationSampleRepository(db)
    samples, total = sample_repo.list_by_run(
        run_id, status=status_filter, limit=limit, offset=offset
    )
    return EvaluationSampleListResponse(samples=samples, total=total, limit=limit, offset=offset)


@router.get("/summary", response_model=EvaluationSummaryResponse)
async def get_summary(
    db: Session = Depends(get_db),
):
    """Dashboard summary: latest run, totals, average scores, score trends."""
    run_repo = EvaluationRunRepository(db)
    data = run_repo.get_summary_data()

    latest_run = None
    if data["latest_run"]:
        latest_run = RunResponse.model_validate(data["latest_run"])

    return EvaluationSummaryResponse(
        latest_run=latest_run,
        total_runs=data["total_runs"],
        total_datasets=data["total_datasets"],
        avg_scores=data["avg_scores"],
        score_trend=data["score_trend"],
    )


# ========== Live Monitoring Endpoints ==========


@router.get("/live/stats", response_model=LiveStatsResponse)
async def get_live_stats(
    request: Request,
    db: Session = Depends(get_db),
):
    """Live monitoring stats: aggregates, hourly breakdown, queue metrics."""
    now = datetime.utcnow()
    cutoff_24h = now - timedelta(hours=24)

    # Total count
    total_scores = db.query(func.count(LiveEvaluationScore.id)).scalar() or 0

    # Last 24h count
    scores_last_24h = (
        db.query(func.count(LiveEvaluationScore.id))
        .filter(LiveEvaluationScore.created_at >= cutoff_24h)
        .scalar()
        or 0
    )

    # Overall averages
    avg_faithfulness = (
        db.query(func.avg(LiveEvaluationScore.faithfulness))
        .filter(LiveEvaluationScore.faithfulness.isnot(None))
        .scalar()
    )
    avg_answer_relevancy = (
        db.query(func.avg(LiveEvaluationScore.answer_relevancy))
        .filter(LiveEvaluationScore.answer_relevancy.isnot(None))
        .scalar()
    )

    # Hourly breakdown (last 24h) — SQLite strftime
    hourly_rows = (
        db.query(
            func.strftime("%Y-%m-%dT%H:00:00", LiveEvaluationScore.created_at).label("hour"),
            func.count(LiveEvaluationScore.id).label("count"),
            func.avg(LiveEvaluationScore.faithfulness).label("avg_faithfulness"),
            func.avg(LiveEvaluationScore.answer_relevancy).label("avg_answer_relevancy"),
        )
        .filter(LiveEvaluationScore.created_at >= cutoff_24h)
        .group_by("hour")
        .order_by("hour")
        .all()
    )

    hourly_breakdown = [
        LiveStatsHourly(
            hour=row.hour,
            count=row.count,
            avg_faithfulness=round(row.avg_faithfulness, 4)
            if row.avg_faithfulness is not None
            else None,
            avg_answer_relevancy=round(row.avg_answer_relevancy, 4)
            if row.avg_answer_relevancy is not None
            else None,
        )
        for row in hourly_rows
    ]

    # Queue stats from app.state
    live_queue = getattr(request.app.state, "live_eval_queue", None)
    queue_depth = live_queue.depth if live_queue else 0
    queue_capacity = live_queue.capacity if live_queue else 0
    drops = live_queue.drops_since_startup if live_queue else 0

    return LiveStatsResponse(
        total_scores=total_scores,
        scores_last_24h=scores_last_24h,
        avg_faithfulness=round(avg_faithfulness, 4) if avg_faithfulness is not None else None,
        avg_answer_relevancy=round(avg_answer_relevancy, 4)
        if avg_answer_relevancy is not None
        else None,
        hourly_breakdown=hourly_breakdown,
        queue_depth=queue_depth,
        queue_capacity=queue_capacity,
        drops_since_startup=drops,
    )


@router.get("/live/scores", response_model=LiveScoreListResponse)
async def list_live_scores(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Paginated live evaluation scores, newest first."""
    total = db.query(func.count(LiveEvaluationScore.id)).scalar() or 0

    scores = (
        db.query(LiveEvaluationScore)
        .order_by(LiveEvaluationScore.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return LiveScoreListResponse(
        scores=[LiveScoreResponse.model_validate(s) for s in scores],
        total=total,
        limit=limit,
        offset=offset,
    )
