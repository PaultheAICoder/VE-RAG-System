"""Tests for the review queue API endpoints."""

import pytest
from fastapi import status

from ai_ready_rag.db.models.review import ReviewItem


@pytest.fixture
def pending_item(db, admin_user):
    """Create a pending review item."""
    item = ReviewItem(
        review_type="low_confidence_answer",
        query="What is the deductible?",
        tentative_answer="The deductible is $1,000.",
        confidence=0.42,
        review_status="pending",
    )
    db.add(item)
    db.flush()
    db.refresh(item)
    return item


@pytest.fixture
def accepted_item(db, admin_user):
    """Create an already-accepted review item."""
    from datetime import UTC, datetime

    item = ReviewItem(
        review_type="account_match_pending",
        query="Find account for Acme Corp",
        tentative_answer=None,
        confidence=0.65,
        review_status="accepted",
        reviewer_id=admin_user.id,
        resolved_at=datetime.now(UTC),
    )
    db.add(item)
    db.flush()
    db.refresh(item)
    return item


@pytest.fixture
def dismissed_item(db, admin_user):
    """Create a dismissed review item."""
    from datetime import UTC, datetime

    item = ReviewItem(
        review_type="unknown_document_type",
        query=None,
        tentative_answer=None,
        confidence=None,
        review_status="dismissed",
        reviewer_id=admin_user.id,
        resolved_at=datetime.now(UTC),
    )
    db.add(item)
    db.flush()
    db.refresh(item)
    return item


@pytest.fixture
def deleted_item(db):
    """Create a soft-deleted review item (should not appear in results)."""
    item = ReviewItem(
        review_type="low_confidence_answer",
        query="Deleted item query",
        review_status="pending",
        is_deleted=True,
    )
    db.add(item)
    db.flush()
    db.refresh(item)
    return item


class TestListReviewQueue:
    """Tests for GET /api/admin/review-queue."""

    def test_list_requires_auth(self, client):
        """Unauthenticated users cannot access the review queue."""
        response = client.get("/api/admin/review-queue")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_requires_admin(self, client, user_headers):
        """Regular users cannot access the review queue."""
        response = client.get("/api/admin/review-queue", headers=user_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_list_empty_queue(self, client, admin_headers):
        """Empty queue returns empty list."""
        response = client.get("/api/admin/review-queue", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    def test_list_returns_items(self, client, admin_headers, pending_item, accepted_item):
        """Returns all non-deleted items when no filter is applied."""
        response = client.get("/api/admin/review-queue", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_list_excludes_deleted(self, client, admin_headers, pending_item, deleted_item):
        """Soft-deleted items are excluded from results."""
        response = client.get("/api/admin/review-queue", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        ids = [item["id"] for item in data["items"]]
        assert pending_item.id in ids
        assert deleted_item.id not in ids

    def test_filter_by_pending_status(self, client, admin_headers, pending_item, accepted_item):
        """Filtering by 'pending' returns only pending items."""
        response = client.get("/api/admin/review-queue?status=pending", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == pending_item.id
        assert data["items"][0]["review_status"] == "pending"

    def test_filter_by_accepted_status(self, client, admin_headers, pending_item, accepted_item):
        """Filtering by 'accepted' returns only accepted items."""
        response = client.get("/api/admin/review-queue?status=accepted", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == accepted_item.id

    def test_filter_by_dismissed_status(self, client, admin_headers, pending_item, dismissed_item):
        """Filtering by 'dismissed' returns only dismissed items."""
        response = client.get("/api/admin/review-queue?status=dismissed", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == dismissed_item.id

    def test_response_shape(self, client, admin_headers, pending_item):
        """Response items contain expected fields."""
        response = client.get("/api/admin/review-queue", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        item = response.json()["items"][0]
        assert "id" in item
        assert "review_type" in item
        assert "review_status" in item
        assert "created_at" in item
        assert item["query"] == "What is the deductible?"
        assert item["confidence"] == pytest.approx(0.42)

    def test_pagination(self, client, admin_headers, db):
        """Pagination returns correct page and page_size."""
        # Create 5 items
        for i in range(5):
            item = ReviewItem(
                review_type="low_confidence_answer",
                query=f"Query {i}",
                review_status="pending",
            )
            db.add(item)
        db.flush()

        response = client.get("/api/admin/review-queue?page=1&page_size=3", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 3
        assert data["page"] == 1
        assert data["page_size"] == 3

    def test_pagination_second_page(self, client, admin_headers, db):
        """Second page returns remaining items."""
        for i in range(5):
            item = ReviewItem(
                review_type="low_confidence_answer",
                query=f"Query {i}",
                review_status="pending",
            )
            db.add(item)
        db.flush()

        response = client.get("/api/admin/review-queue?page=2&page_size=3", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["page"] == 2

    def test_admin_can_access(self, client, admin_headers, pending_item):
        """Admin user (role='admin') can access the review queue."""
        response = client.get("/api/admin/review-queue", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK

    def test_customer_admin_can_access(self, client, customer_admin_headers, pending_item):
        """Customer admin can access the review queue."""
        response = client.get("/api/admin/review-queue", headers=customer_admin_headers)
        assert response.status_code == status.HTTP_200_OK


class TestApproveReviewItem:
    """Tests for POST /api/admin/review-queue/{item_id}/approve."""

    def test_approve_requires_auth(self, client, pending_item):
        """Unauthenticated users cannot approve items."""
        response = client.post(f"/api/admin/review-queue/{pending_item.id}/approve")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_approve_requires_admin(self, client, user_headers, pending_item):
        """Regular users cannot approve items."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/approve",
            headers=user_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_approve_pending_item(self, client, admin_headers, pending_item):
        """Admin can approve a pending item."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/approve",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == pending_item.id
        assert data["review_status"] == "accepted"
        assert data["resolved_at"] is not None
        assert data["reviewer_id"] is not None

    def test_approve_sets_reviewer(self, client, admin_headers, admin_user, pending_item):
        """Approve records the reviewer's ID."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/approve",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["reviewer_id"] == admin_user.id

    def test_approve_not_found(self, client, admin_headers):
        """Returns 404 for non-existent item."""
        response = client.post(
            "/api/admin/review-queue/nonexistent-id/approve",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_approve_already_resolved_item(self, client, admin_headers, accepted_item):
        """Cannot approve an item that is already resolved."""
        response = client.post(
            f"/api/admin/review-queue/{accepted_item.id}/approve",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_approve_dismissed_item_returns_conflict(self, client, admin_headers, dismissed_item):
        """Cannot approve a dismissed item."""
        response = client.post(
            f"/api/admin/review-queue/{dismissed_item.id}/approve",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_approve_deleted_item_returns_404(self, client, admin_headers, deleted_item):
        """Soft-deleted items return 404."""
        response = client.post(
            f"/api/admin/review-queue/{deleted_item.id}/approve",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestRejectReviewItem:
    """Tests for POST /api/admin/review-queue/{item_id}/reject."""

    def test_reject_requires_auth(self, client, pending_item):
        """Unauthenticated users cannot reject items."""
        response = client.post(f"/api/admin/review-queue/{pending_item.id}/reject")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_reject_requires_admin(self, client, user_headers, pending_item):
        """Regular users cannot reject items."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/reject",
            headers=user_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_reject_pending_item(self, client, admin_headers, pending_item):
        """Admin can reject a pending item."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/reject",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == pending_item.id
        assert data["review_status"] == "dismissed"
        assert data["resolved_at"] is not None
        assert data["reviewer_id"] is not None

    def test_reject_sets_reviewer(self, client, admin_headers, admin_user, pending_item):
        """Reject records the reviewer's ID."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/reject",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["reviewer_id"] == admin_user.id

    def test_reject_not_found(self, client, admin_headers):
        """Returns 404 for non-existent item."""
        response = client.post(
            "/api/admin/review-queue/nonexistent-id/reject",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_reject_already_resolved_item(self, client, admin_headers, accepted_item):
        """Cannot reject an item that is already resolved."""
        response = client.post(
            f"/api/admin/review-queue/{accepted_item.id}/reject",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_409_CONFLICT

    def test_reject_deleted_item_returns_404(self, client, admin_headers, deleted_item):
        """Soft-deleted items return 404."""
        response = client.post(
            f"/api/admin/review-queue/{deleted_item.id}/reject",
            headers=admin_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_approve_then_reject_returns_conflict(self, client, admin_headers, pending_item):
        """Approving then trying to reject returns 409."""
        # First approve
        approve_response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/approve",
            headers=admin_headers,
        )
        assert approve_response.status_code == status.HTTP_200_OK

        # Now try to reject - should be conflict
        reject_response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/reject",
            headers=admin_headers,
        )
        assert reject_response.status_code == status.HTTP_409_CONFLICT

    def test_customer_admin_can_reject(self, client, customer_admin_headers, pending_item):
        """Customer admin can also reject items."""
        response = client.post(
            f"/api/admin/review-queue/{pending_item.id}/reject",
            headers=customer_admin_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["review_status"] == "dismissed"
