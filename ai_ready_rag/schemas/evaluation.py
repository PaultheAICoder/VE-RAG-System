"""Pydantic schemas for evaluation framework."""

import json
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# ---------- Dataset Schemas ----------


class DatasetSampleCreate(BaseModel):
    """Single Q&A pair for dataset creation."""

    question: str
    ground_truth: str | None = None
    reference_contexts: list[str] | None = None
    metadata: dict | None = None


class DatasetCreate(BaseModel):
    """Request to create a manual dataset."""

    name: str
    description: str | None = None
    samples: list[DatasetSampleCreate]

    @field_validator("samples")
    @classmethod
    def at_least_one_sample(cls, v: list) -> list:
        if not v:
            raise ValueError("At least one sample is required")
        return v


class RAGBenchImportRequest(BaseModel):
    """Request to import a RAGBench subset from pre-downloaded parquet files."""

    subset: str
    max_samples: int = Field(50, ge=1, le=5000)
    name: str
    description: str | None = None


class SyntheticGenerateRequest(BaseModel):
    """Request to generate a synthetic dataset from uploaded documents."""

    name: str
    document_ids: list[str]
    num_samples: int = Field(30, ge=1, le=200)
    description: str | None = None

    @field_validator("document_ids")
    @classmethod
    def at_least_one_document(cls, v: list) -> list:
        if not v:
            raise ValueError("At least one document_id is required")
        return v


class DatasetSampleResponse(BaseModel):
    """Single dataset sample response."""

    id: str
    dataset_id: str
    question: str
    ground_truth: str | None
    reference_contexts: list[str] | None = None
    metadata: dict | None = Field(default=None, validation_alias="metadata_")
    sort_order: int
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True

    @field_validator("reference_contexts", mode="before")
    @classmethod
    def parse_reference_contexts(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class DatasetResponse(BaseModel):
    """Dataset response."""

    id: str
    name: str
    description: str | None
    source_type: str
    sample_count: int
    created_by: str | None
    created_at: datetime
    updated_at: datetime
    warning: str | None = None

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Paginated dataset list."""

    datasets: list[DatasetResponse]
    total: int
    limit: int
    offset: int


class DatasetSampleListResponse(BaseModel):
    """Paginated sample list."""

    samples: list[DatasetSampleResponse]
    total: int
    limit: int
    offset: int


# ---------- Run Schemas ----------


class RunCreate(BaseModel):
    """Request to trigger an evaluation run."""

    dataset_id: str
    name: str
    description: str | None = None
    tag_scope: list[str] | None = None
    admin_bypass_tags: bool = False


class RunResponse(BaseModel):
    """Evaluation run response."""

    id: str
    name: str
    description: str | None
    dataset_id: str
    status: str
    total_samples: int
    completed_samples: int
    failed_samples: int
    tag_scope: list[str] | None = None
    admin_bypass_tags: bool
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    avg_llm_context_precision: float | None
    avg_llm_context_recall: float | None
    invalid_score_count: int
    model_used: str
    embedding_model_used: str
    config_snapshot: dict
    is_cancel_requested: bool
    error_message: str | None
    triggered_by: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    eta_seconds: float | None = None

    class Config:
        from_attributes = True

    @field_validator("tag_scope", mode="before")
    @classmethod
    def parse_tag_scope(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("config_snapshot", mode="before")
    @classmethod
    def parse_config_snapshot(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class RunListResponse(BaseModel):
    """Paginated run list."""

    runs: list[RunResponse]
    total: int
    limit: int
    offset: int


class EvaluationSampleResponse(BaseModel):
    """Per-sample evaluation result."""

    id: str
    sort_order: int
    status: str
    question: str
    ground_truth: str | None
    generated_answer: str | None
    faithfulness: float | None
    answer_relevancy: float | None
    llm_context_precision: float | None
    llm_context_recall: float | None
    generation_time_ms: float | None
    retry_count: int
    error_message: str | None
    error_type: str | None
    processed_at: datetime | None

    class Config:
        from_attributes = True


class EvaluationSampleListResponse(BaseModel):
    """Paginated evaluation sample list."""

    samples: list[EvaluationSampleResponse]
    total: int
    limit: int
    offset: int


class CancelRunResponse(BaseModel):
    """Response from cancel request."""

    id: str
    status: str
    is_cancel_requested: bool


class EvaluationSummaryResponse(BaseModel):
    """Dashboard summary."""

    latest_run: RunResponse | None
    total_runs: int
    total_datasets: int
    avg_scores: dict
    score_trend: list[dict]


# ---------- Live Monitoring Schemas ----------


class LiveScoreResponse(BaseModel):
    """Single live evaluation score."""

    id: str
    query: str
    answer: str
    model_used: str
    faithfulness: float | None
    answer_relevancy: float | None
    generation_time_ms: float | None
    evaluation_time_ms: float | None
    error_message: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class LiveScoreListResponse(BaseModel):
    """Paginated live scores."""

    scores: list[LiveScoreResponse]
    total: int
    limit: int
    offset: int


class LiveStatsHourly(BaseModel):
    """Per-hour aggregate for live monitoring."""

    hour: str
    count: int
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None


class LiveStatsResponse(BaseModel):
    """Live monitoring dashboard stats."""

    total_scores: int
    scores_last_24h: int
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    hourly_breakdown: list[LiveStatsHourly]
    queue_depth: int
    queue_capacity: int
    drops_since_startup: int
