"""Tests for evaluation framework API endpoints."""

import pytest
from fastapi import status

from ai_ready_rag.db.models.evaluation import DatasetSample, EvaluationDataset


@pytest.fixture
def sample_dataset(db, admin_user):
    """Create a sample evaluation dataset."""
    dataset = EvaluationDataset(
        name="Test Dataset",
        description="Test description",
        source_type="manual",
        sample_count=2,
        created_by=admin_user.id,
    )
    db.add(dataset)
    db.flush()

    # Add samples
    for i in range(2):
        sample = DatasetSample(
            dataset_id=dataset.id,
            question=f"Question {i}?",
            ground_truth=f"Answer {i}",
            sort_order=i,
        )
        db.add(sample)
    db.flush()
    db.refresh(dataset)
    return dataset


class TestDatasetCRUD:
    """Test dataset CRUD operations."""

    def test_create_dataset_requires_admin(self, client, user_headers):
        """Non-admin users cannot create datasets."""
        response = client.post(
            "/api/evaluations/datasets",
            json={
                "name": "Test",
                "samples": [{"question": "Q?", "ground_truth": "A"}],
            },
            headers=user_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_create_dataset_success(self, client, admin_headers):
        """Admin can create a manual dataset."""
        response = client.post(
            "/api/evaluations/datasets",
            json={
                "name": "HR Policy QA",
                "description": "Test dataset",
                "samples": [
                    {"question": "What is PTO?", "ground_truth": "15 days"},
                    {"question": "What is sick leave?", "ground_truth": "10 days"},
                ],
            },
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "HR Policy QA"
        assert data["source_type"] == "manual"
        assert data["sample_count"] == 2
        assert data.get("warning") is None

    def test_create_dataset_ground_truth_warning(self, client, admin_headers):
        """Warning returned when samples lack ground truth."""
        response = client.post(
            "/api/evaluations/datasets",
            json={
                "name": "Mixed Dataset",
                "samples": [
                    {"question": "Q1?", "ground_truth": "A1"},
                    {"question": "Q2?"},  # No ground truth
                    {"question": "Q3?", "ground_truth": "  "},  # Whitespace only
                ],
            },
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["sample_count"] == 3
        assert data["warning"] is not None
        assert "2 samples" in data["warning"]

    def test_create_dataset_duplicate_name(self, client, admin_headers, sample_dataset):
        """Duplicate dataset name returns 409."""
        response = client.post(
            "/api/evaluations/datasets",
            json={
                "name": sample_dataset.name,
                "samples": [{"question": "Q?"}],
            },
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_create_dataset_empty_samples(self, client, admin_headers):
        """Empty samples list returns 422."""
        response = client.post(
            "/api/evaluations/datasets",
            json={"name": "Empty", "samples": []},
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_datasets(self, client, admin_headers, sample_dataset):
        """Admin can list datasets."""
        response = client.get("/api/evaluations/datasets", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 1
        assert len(data["datasets"]) >= 1

    def test_list_datasets_pagination(self, client, admin_headers, sample_dataset):
        """Pagination parameters work."""
        response = client.get(
            "/api/evaluations/datasets?limit=1&offset=0",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["limit"] == 1
        assert data["offset"] == 0

    def test_get_dataset(self, client, admin_headers, sample_dataset):
        """Admin can get a single dataset."""
        response = client.get(
            f"/api/evaluations/datasets/{sample_dataset.id}",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == sample_dataset.name
        assert data["sample_count"] == 2

    def test_get_dataset_not_found(self, client, admin_headers):
        """404 for non-existent dataset."""
        response = client.get(
            "/api/evaluations/datasets/nonexistent-id",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_dataset_samples(self, client, admin_headers, sample_dataset):
        """Admin can list samples in a dataset."""
        response = client.get(
            f"/api/evaluations/datasets/{sample_dataset.id}/samples",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["samples"]) == 2
        assert data["samples"][0]["sort_order"] == 0

    def test_list_dataset_samples_not_found(self, client, admin_headers):
        """404 for non-existent dataset."""
        response = client.get(
            "/api/evaluations/datasets/nonexistent-id/samples",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_dataset(self, client, admin_headers, sample_dataset):
        """Admin can delete a dataset."""
        response = client.delete(
            f"/api/evaluations/datasets/{sample_dataset.id}",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_dataset_not_found(self, client, admin_headers):
        """404 for non-existent dataset."""
        response = client.delete(
            "/api/evaluations/datasets/nonexistent-id",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_recompute_counts(self, client, admin_headers):
        """Recompute counts endpoint works."""
        response = client.post(
            "/api/evaluations/datasets/recompute-counts",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        assert "updated" in response.json()


class TestRunEndpoints:
    """Test run endpoints (replaced Phase 1 stubs)."""

    def test_create_run_requires_body(self, client, admin_headers):
        """POST without body returns 422."""
        response = client.post("/api/evaluations/runs", headers=admin_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_run_dataset_not_found(self, client, admin_headers):
        """POST with invalid dataset_id returns 404."""
        response = client.post(
            "/api/evaluations/runs",
            json={
                "dataset_id": "nonexistent",
                "name": "test",
                "tag_scope": ["hr"],
            },
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_run_mutually_exclusive(self, client, admin_headers):
        """POST with both tag_scope and admin_bypass_tags returns 422."""
        response = client.post(
            "/api/evaluations/runs",
            json={
                "dataset_id": "fake",
                "name": "test",
                "tag_scope": ["hr"],
                "admin_bypass_tags": True,
            },
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_run_not_found(self, client, admin_headers):
        """GET non-existent run returns 404."""
        response = client.get("/api/evaluations/runs/some-id", headers=admin_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_run_not_found(self, client, admin_headers):
        """DELETE non-existent run returns 404."""
        response = client.delete("/api/evaluations/runs/some-id", headers=admin_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_cancel_run_not_found(self, client, admin_headers):
        """Cancel non-existent run returns 404."""
        response = client.post("/api/evaluations/runs/some-id/cancel", headers=admin_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_summary_returns_data(self, client, admin_headers):
        """Summary endpoint returns dashboard data."""
        response = client.get("/api/evaluations/summary", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_runs" in data
        assert "total_datasets" in data
        assert "avg_scores" in data
        assert "score_trend" in data

    def test_run_samples_not_found(self, client, admin_headers):
        """List samples for non-existent run returns 404."""
        response = client.get("/api/evaluations/runs/some-id/samples", headers=admin_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestRunList:
    """Test that GET /runs works (returns empty list)."""

    def test_list_runs_empty(self, client, admin_headers):
        """List runs returns empty when no runs exist."""
        response = client.get("/api/evaluations/runs", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert data["runs"] == []


class TestAccessControl:
    """Test that all endpoints require admin."""

    def test_list_datasets_requires_admin(self, client, user_headers):
        response = client.get("/api/evaluations/datasets", headers=user_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_list_runs_requires_admin(self, client, user_headers):
        response = client.get("/api/evaluations/runs", headers=user_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_unauthenticated_rejected(self, client):
        response = client.get("/api/evaluations/datasets")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
