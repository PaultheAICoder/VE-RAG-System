"""Tests for EvaluationService: sanitization, create_run, compute_aggregates, process_sample."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_ready_rag.db.models.evaluation import (
    EvaluationDataset,
    EvaluationRun,
    EvaluationSample,
)
from ai_ready_rag.services.evaluation_service import EvaluationService

# ---------- Fixtures ----------


@pytest.fixture
def mock_settings():
    """Minimal Settings mock for EvaluationService."""
    settings = MagicMock()
    settings.chat_model = "qwen3:8b"
    settings.embedding_model = "nomic-embed-text"
    settings.ollama_base_url = "http://localhost:11434"
    settings.rag_temperature = 0.1
    settings.rag_timeout_seconds = 30
    settings.eval_timeout_seconds = 120
    settings.eval_max_samples_per_run = 500
    return settings


@pytest.fixture
def mock_rag_service():
    """Mock RAGService with generate_for_eval."""
    service = MagicMock()
    service.generate_for_eval = AsyncMock()
    return service


@pytest.fixture
def eval_service(mock_settings, mock_rag_service):
    return EvaluationService(mock_settings, mock_rag_service)


# ---------- TestSanitizeScore ----------


class TestSanitizeScore:
    def test_valid_scores(self):
        assert EvaluationService.sanitize_score(0.0) == 0.0
        assert EvaluationService.sanitize_score(0.5) == 0.5
        assert EvaluationService.sanitize_score(1.0) == 1.0

    def test_clamp_near_zero(self):
        assert EvaluationService.sanitize_score(-0.03) == 0.0
        assert EvaluationService.sanitize_score(-0.049) == 0.0

    def test_clamp_near_one(self):
        assert EvaluationService.sanitize_score(1.02) == 1.0
        assert EvaluationService.sanitize_score(1.049) == 1.0

    def test_none_returns_none(self):
        assert EvaluationService.sanitize_score(None) is None

    def test_nan_returns_none(self):
        assert EvaluationService.sanitize_score(float("nan")) is None

    def test_inf_returns_none(self):
        assert EvaluationService.sanitize_score(float("inf")) is None
        assert EvaluationService.sanitize_score(float("-inf")) is None

    def test_out_of_range(self):
        assert EvaluationService.sanitize_score(-0.1) is None
        assert EvaluationService.sanitize_score(1.1) is None
        assert EvaluationService.sanitize_score(-1.0) is None
        assert EvaluationService.sanitize_score(2.0) is None

    def test_non_numeric(self):
        assert EvaluationService.sanitize_score("abc") is None
        assert EvaluationService.sanitize_score([]) is None
        assert EvaluationService.sanitize_score({}) is None

    def test_integer_values(self):
        assert EvaluationService.sanitize_score(0) == 0.0
        assert EvaluationService.sanitize_score(1) == 1.0


# ---------- TestBuildConfigSnapshot ----------


class TestBuildConfigSnapshot:
    def test_snapshot_has_required_fields(self, eval_service, db):
        snapshot = eval_service.build_config_snapshot(db)
        required_keys = {
            "chat_model",
            "embedding_model",
            "temperature",
            "chunking_strategy",
            "chunk_max_tokens",
            "chunk_overlap_tokens",
            "retrieval_top_k",
            "reranker_enabled",
            "reranker_model",
            "prompt_template_hash",
            "corpus_doc_count",
            "corpus_last_ingested_at",
            "rag_timeout_seconds",
            "eval_timeout_seconds",
        }
        assert required_keys.issubset(snapshot.keys())

    def test_prompt_template_hash_deterministic(self, eval_service, db):
        s1 = eval_service.build_config_snapshot(db)
        s2 = eval_service.build_config_snapshot(db)
        assert s1["prompt_template_hash"] == s2["prompt_template_hash"]
        assert s1["prompt_template_hash"].startswith("sha256:")


# ---------- TestCreateRun ----------


class TestCreateRun:
    @pytest.mark.asyncio
    async def test_mutually_exclusive_tags(self, eval_service, db):
        from fastapi import HTTPException

        from ai_ready_rag.schemas.evaluation import RunCreate

        request = RunCreate(
            dataset_id="fake-id",
            name="test",
            tag_scope=["hr"],
            admin_bypass_tags=True,
        )
        with pytest.raises(HTTPException) as exc_info:
            await eval_service.create_run(db, request, triggered_by="user1")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_no_scope_specified(self, eval_service, db):
        from fastapi import HTTPException

        from ai_ready_rag.schemas.evaluation import RunCreate

        request = RunCreate(
            dataset_id="fake-id",
            name="test",
            tag_scope=None,
            admin_bypass_tags=False,
        )
        with pytest.raises(HTTPException) as exc_info:
            await eval_service.create_run(db, request, triggered_by="user1")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_dataset_not_found(self, eval_service, db):
        from fastapi import HTTPException

        from ai_ready_rag.schemas.evaluation import RunCreate

        request = RunCreate(
            dataset_id="nonexistent",
            name="test",
            tag_scope=["hr"],
        )
        with pytest.raises(HTTPException) as exc_info:
            await eval_service.create_run(db, request, triggered_by="user1")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_samples_copied(self, eval_service, db):
        from ai_ready_rag.db.repositories.evaluation import (
            DatasetSampleRepository,
            EvaluationDatasetRepository,
            EvaluationSampleRepository,
        )
        from ai_ready_rag.schemas.evaluation import RunCreate

        # Create dataset with samples
        dataset_repo = EvaluationDatasetRepository(db)
        dataset = EvaluationDataset(
            name="test-ds",
            source_type="manual",
            sample_count=2,
        )
        dataset_repo.add(dataset)
        db.flush()

        sample_repo = DatasetSampleRepository(db)
        sample_repo.bulk_create(
            dataset.id,
            [
                {"question": "Q1", "ground_truth": "A1"},
                {"question": "Q2", "ground_truth": None},
            ],
        )
        db.flush()

        request = RunCreate(
            dataset_id=dataset.id,
            name="test-run",
            tag_scope=["hr"],
        )
        run = await eval_service.create_run(db, request, triggered_by="user1")

        # Verify eval samples created
        eval_sample_repo = EvaluationSampleRepository(db)
        samples, total = eval_sample_repo.list_by_run(run.id)
        assert total == 2
        assert samples[0].question == "Q1"
        assert samples[0].ground_truth == "A1"
        assert samples[0].status == "pending"
        assert samples[1].question == "Q2"
        assert samples[1].ground_truth is None


# ---------- TestComputeAggregates ----------


class TestComputeAggregates:
    @pytest.mark.asyncio
    async def test_avg_scores_populated(self, eval_service, db):
        from ai_ready_rag.db.repositories.evaluation import EvaluationRunRepository

        # Create a run + completed samples
        run_repo = EvaluationRunRepository(db)
        run = EvaluationRun(
            name="agg-test",
            dataset_id="ds-fake",
            model_used="qwen3:8b",
            embedding_model_used="nomic-embed-text",
            config_snapshot="{}",
            total_samples=2,
        )
        run_repo.add(run)
        db.flush()

        s1 = EvaluationSample(
            run_id=run.id,
            question="Q1",
            status="completed",
            faithfulness=0.8,
            answer_relevancy=0.6,
            sort_order=0,
        )
        s2 = EvaluationSample(
            run_id=run.id,
            question="Q2",
            status="completed",
            faithfulness=0.4,
            answer_relevancy=0.8,
            sort_order=1,
        )
        db.add_all([s1, s2])
        db.flush()

        await eval_service.compute_aggregates(db, run)
        assert run.avg_faithfulness == pytest.approx(0.6)
        assert run.avg_answer_relevancy == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_null_metrics_counted(self, eval_service, db):
        from ai_ready_rag.db.repositories.evaluation import EvaluationRunRepository

        run_repo = EvaluationRunRepository(db)
        run = EvaluationRun(
            name="null-test",
            dataset_id="ds-fake",
            model_used="qwen3:8b",
            embedding_model_used="nomic-embed-text",
            config_snapshot="{}",
            total_samples=1,
        )
        run_repo.add(run)
        db.flush()

        # Sample with some NULL metrics
        s1 = EvaluationSample(
            run_id=run.id,
            question="Q1",
            status="completed",
            faithfulness=0.8,
            answer_relevancy=None,  # NULL
            llm_context_precision=None,  # NULL
            llm_context_recall=None,  # NULL
            sort_order=0,
        )
        db.add(s1)
        db.flush()

        await eval_service.compute_aggregates(db, run)
        assert run.invalid_score_count == 3  # 3 NULL metric cells


# ---------- TestProcessSample ----------


class TestProcessSample:
    @pytest.mark.asyncio
    async def test_success_updates_sample(self, eval_service, mock_rag_service):
        from ai_ready_rag.services.rag_service import (
            ConfidenceScore,
            EvalRAGResult,
            RAGResponse,
        )

        mock_response = RAGResponse(
            answer="Test answer",
            confidence=ConfidenceScore(
                overall=80, retrieval_score=0.9, coverage_score=0.8, llm_score=75
            ),
            citations=[],
            action="CITE",
            route_to=None,
            model_used="qwen3:8b",
            context_chunks_used=3,
            context_tokens_used=100,
            generation_time_ms=500.0,
            grounded=True,
        )
        mock_rag_service.generate_for_eval.return_value = EvalRAGResult(
            response=mock_response,
            retrieved_contexts=["ctx1", "ctx2"],
            retrieval_scores=[0.9, 0.8],
        )

        sample = EvaluationSample(
            run_id="run1",
            question="What is X?",
            status="processing",
            sort_order=0,
        )

        with patch.object(eval_service, "evaluate_single", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = {
                "faithfulness": 0.85,
                "answer_relevancy": 0.9,
            }
            result = await eval_service.process_sample(
                MagicMock(), sample, tag_scope=["hr"], admin_bypass_tags=False
            )

        assert result.status == "completed"
        assert result.generated_answer == "Test answer"
        assert result.faithfulness == 0.85
        assert result.answer_relevancy == 0.9
        assert result.processed_at is not None

    @pytest.mark.asyncio
    async def test_rag_error_marks_failed(self, eval_service, mock_rag_service):
        mock_rag_service.generate_for_eval.side_effect = RuntimeError("Ollama down")

        sample = EvaluationSample(
            run_id="run1",
            question="What is X?",
            status="processing",
            sort_order=0,
        )

        result = await eval_service.process_sample(
            MagicMock(), sample, tag_scope=["hr"], admin_bypass_tags=False
        )

        assert result.status == "failed"
        assert "Ollama down" in result.error_message
        assert result.error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_all_none_metrics_marks_failed(self, eval_service, mock_rag_service):
        from ai_ready_rag.services.rag_service import (
            ConfidenceScore,
            EvalRAGResult,
            RAGResponse,
        )

        mock_response = RAGResponse(
            answer="Test",
            confidence=ConfidenceScore(
                overall=50, retrieval_score=0.5, coverage_score=0.5, llm_score=50
            ),
            citations=[],
            action="CITE",
            route_to=None,
            model_used="qwen3:8b",
            context_chunks_used=1,
            context_tokens_used=50,
            generation_time_ms=200.0,
            grounded=False,
        )
        mock_rag_service.generate_for_eval.return_value = EvalRAGResult(
            response=mock_response,
            retrieved_contexts=["ctx1"],
            retrieval_scores=[0.5],
        )

        sample = EvaluationSample(
            run_id="run1",
            question="What is X?",
            status="processing",
            sort_order=0,
        )

        with patch.object(eval_service, "evaluate_single", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = {
                "faithfulness": None,
                "answer_relevancy": None,
            }
            result = await eval_service.process_sample(
                MagicMock(), sample, tag_scope=["hr"], admin_bypass_tags=False
            )

        assert result.status == "failed"
        assert result.error_type == "MetricValidationError"
