"""Evaluation service wrapping RAGAS v0.4 for RAG quality assessment."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ai_ready_rag.config import Settings
from ai_ready_rag.db.models.evaluation import (
    EvaluationRun,
    EvaluationSample,
)
from ai_ready_rag.db.repositories.evaluation import (
    DatasetSampleRepository,
    EvaluationDatasetRepository,
    EvaluationRunRepository,
    EvaluationSampleRepository,
)
from ai_ready_rag.schemas.evaluation import RunCreate

if TYPE_CHECKING:
    from ai_ready_rag.services.rag_service import RAGService

logger = logging.getLogger(__name__)

CLAMP_TOLERANCE = 0.05


class EvaluationService:
    """Wraps RAGAS evaluation with Ollama via OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings, rag_service: RAGService) -> None:
        self.settings = settings
        self.rag_service = rag_service

    def _get_ragas_llm(self):
        """Get RAGAS-compatible LLM via Ollama's OpenAI-compatible endpoint."""
        from ragas.llms import llm_factory

        return llm_factory(
            model=self.settings.chat_model,
            run_config=None,
            default_headers=None,
            base_url=f"{self.settings.ollama_base_url}/v1",
            api_key="ollama",
        )

    def _get_ragas_embeddings(self):
        """Get RAGAS-compatible embeddings via Ollama's OpenAI-compatible endpoint."""
        from ragas.embeddings import embedding_factory

        return embedding_factory(
            model=self.settings.embedding_model,
            run_config=None,
            default_headers=None,
            base_url=f"{self.settings.ollama_base_url}/v1",
            api_key="ollama",
        )

    @staticmethod
    def sanitize_score(value: Any) -> float | None:
        """Sanitize a single RAGAS metric score before persistence.

        Rules:
        - Accept only finite floats in [0.0, 1.0]
        - Clamp values slightly outside range (tolerance of 0.05)
        - Reject values outside tolerance as None
        - None/NaN/Inf -> None
        """
        if value is None:
            return None

        try:
            fval = float(value)
        except (TypeError, ValueError):
            logger.warning("Metric: non-numeric value %r, storing as NULL", value)
            return None

        if not math.isfinite(fval):
            logger.warning("Metric: non-finite value %s, storing as NULL", fval)
            return None

        if -CLAMP_TOLERANCE <= fval < 0.0:
            return 0.0
        elif 1.0 < fval <= 1.0 + CLAMP_TOLERANCE:
            return 1.0
        elif 0.0 <= fval <= 1.0:
            return fval
        else:
            logger.warning("Metric: out-of-range value %s, storing as NULL", fval)
            return None

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float | None]:
        """Run RAGAS metrics on a single Q&A pair.

        Returns dict of metric_name -> sanitized score.
        """
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import AnswerRelevancy, Faithfulness

        ragas_llm = self._get_ragas_llm()
        ragas_embeddings = self._get_ragas_embeddings()

        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ]

        # Normalize ground truth
        gt_normalized = ground_truth.strip() if ground_truth else None
        gt_normalized = gt_normalized or None  # "" -> None

        if gt_normalized:
            from ragas.metrics import LLMContextPrecision, LLMContextRecall

            metrics.extend(
                [
                    LLMContextPrecision(llm=ragas_llm),
                    LLMContextRecall(llm=ragas_llm),
                ]
            )

        eval_sample = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if gt_normalized:
            eval_sample["reference"] = gt_normalized

        dataset = EvaluationDataset.from_list([eval_sample])

        # ragas.evaluate() is synchronous — run in thread to avoid blocking event loop
        result = await asyncio.to_thread(evaluate, dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        score_columns = [
            c
            for c in df.columns
            if c not in ("user_input", "response", "retrieved_contexts", "reference")
        ]
        raw_scores = {col: df[col].iloc[0] for col in score_columns}

        return {k: self.sanitize_score(v) for k, v in raw_scores.items()}

    def build_config_snapshot(self, db: Session) -> dict:
        """Capture immutable config snapshot for evaluation run reproducibility."""
        from ai_ready_rag.db.models import Document
        from ai_ready_rag.services.rag_service import RAG_SYSTEM_PROMPT
        from ai_ready_rag.services.settings_service import SettingsService

        settings_svc = SettingsService(db)

        # Corpus stats
        corpus_doc_count = db.query(Document).filter(Document.status == "ready").count()
        latest_doc = (
            db.query(Document)
            .filter(Document.status == "ready")
            .order_by(Document.uploaded_at.desc())
            .first()
        )
        corpus_last_ingested_at = latest_doc.uploaded_at.isoformat() if latest_doc else None

        # Prompt template hash
        prompt_hash = hashlib.sha256(RAG_SYSTEM_PROMPT.encode()).hexdigest()

        return {
            "chat_model": self.settings.chat_model,
            "embedding_model": self.settings.embedding_model,
            "temperature": self.settings.rag_temperature,
            "chunking_strategy": "hybrid",
            "chunk_max_tokens": 512,
            "chunk_overlap_tokens": 50,
            "retrieval_top_k": settings_svc.get("retrieval_top_k") or 5,
            "reranker_enabled": False,
            "reranker_model": None,
            "prompt_template_hash": f"sha256:{prompt_hash}",
            "corpus_doc_count": corpus_doc_count,
            "corpus_last_ingested_at": corpus_last_ingested_at,
            "rag_timeout_seconds": self.settings.rag_timeout_seconds,
            "eval_timeout_seconds": self.settings.eval_timeout_seconds,
        }

    async def create_run(
        self,
        db: Session,
        request: RunCreate,
        triggered_by: str,
    ) -> EvaluationRun:
        """Create a new evaluation run from a dataset."""
        # Validate mutual exclusivity
        if request.tag_scope and request.admin_bypass_tags:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="tag_scope and admin_bypass_tags are mutually exclusive",
            )
        if not request.tag_scope and not request.admin_bypass_tags:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Must specify either tag_scope or admin_bypass_tags=true",
            )

        dataset_repo = EvaluationDatasetRepository(db)
        dataset = dataset_repo.get(request.dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )

        sample_repo = DatasetSampleRepository(db)
        dataset_samples = sample_repo.list_all_by_dataset(request.dataset_id)
        if not dataset_samples:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Dataset has no samples",
            )

        if len(dataset_samples) > self.settings.eval_max_samples_per_run:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Dataset has {len(dataset_samples)} samples, max is {self.settings.eval_max_samples_per_run}",
            )

        config_snapshot = self.build_config_snapshot(db)

        run_repo = EvaluationRunRepository(db)
        run = EvaluationRun(
            name=request.name,
            description=request.description,
            dataset_id=request.dataset_id,
            status="pending",
            total_samples=len(dataset_samples),
            tag_scope=json.dumps(request.tag_scope) if request.tag_scope else None,
            admin_bypass_tags=request.admin_bypass_tags,
            model_used=self.settings.chat_model,
            embedding_model_used=self.settings.embedding_model,
            config_snapshot=json.dumps(config_snapshot),
            triggered_by=triggered_by,
        )
        run_repo.add(run)
        run_repo.flush()

        # Copy dataset samples into evaluation samples
        eval_sample_repo = EvaluationSampleRepository(db)
        eval_sample_repo.bulk_create_from_dataset(run.id, dataset_samples)

        db.commit()
        db.refresh(run)
        return run

    async def process_sample(
        self,
        db: Session,
        sample: EvaluationSample,
        tag_scope: list[str] | None,
        admin_bypass_tags: bool,
    ) -> EvaluationSample:
        """Process a single evaluation sample: run RAG + RAGAS metrics."""
        from ai_ready_rag.services.rag_service import RAGRequest

        user_tags = None if admin_bypass_tags else tag_scope

        rag_request = RAGRequest(
            query=sample.question,
            user_tags=user_tags,
            is_warming=False,
        )

        try:
            eval_result = await self.rag_service.generate_for_eval(rag_request, db)

            sample.generated_answer = eval_result.response.answer
            sample.retrieved_contexts = json.dumps(eval_result.retrieved_contexts)
            sample.generation_time_ms = eval_result.response.generation_time_ms

            scores = await self.evaluate_single(
                question=sample.question,
                answer=eval_result.response.answer,
                contexts=eval_result.retrieved_contexts,
                ground_truth=sample.ground_truth,
            )

            sample.faithfulness = scores.get("faithfulness")
            sample.answer_relevancy = scores.get("answer_relevancy")
            sample.llm_context_precision = scores.get("llm_context_precision")
            sample.llm_context_recall = scores.get("llm_context_recall")

            # Check if all metrics are None -> failed
            gt_normalized = sample.ground_truth.strip() if sample.ground_truth else None
            gt_normalized = gt_normalized or None
            metric_values = [scores.get("faithfulness"), scores.get("answer_relevancy")]
            if gt_normalized:
                metric_values.extend(
                    [
                        scores.get("llm_context_precision"),
                        scores.get("llm_context_recall"),
                    ]
                )
            if all(v is None for v in metric_values):
                sample.status = "failed"
                sample.error_type = "MetricValidationError"
                sample.error_message = "All RAGAS metrics returned invalid values"
            else:
                sample.status = "completed"

            sample.processed_at = datetime.utcnow()

        except Exception as e:
            logger.error("Error processing eval sample %s: %s", sample.id, e)
            sample.status = "failed"
            sample.error_message = str(e)
            sample.error_type = type(e).__name__
            sample.processed_at = datetime.utcnow()

        return sample

    async def compute_aggregates(self, db: Session, run: EvaluationRun) -> EvaluationRun:
        """Compute aggregate scores for a completed run."""
        eval_sample_repo = EvaluationSampleRepository(db)
        avg_scores = eval_sample_repo.get_aggregate_scores(run.id)
        invalid_count = eval_sample_repo.count_null_metrics(run.id)

        run.avg_faithfulness = avg_scores.get("faithfulness")
        run.avg_answer_relevancy = avg_scores.get("answer_relevancy")
        run.avg_llm_context_precision = avg_scores.get("llm_context_precision")
        run.avg_llm_context_recall = avg_scores.get("llm_context_recall")
        run.invalid_score_count = invalid_count

        return run
