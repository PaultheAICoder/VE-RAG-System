"""Tests for curated Q&A CRUD endpoints."""

import json

from ai_ready_rag.db.models import CuratedQA, CuratedQAKeyword


class TestListCuratedQA:
    """Tests for GET /api/admin/qa endpoint."""

    def test_returns_empty_list_initially(self, client, system_admin_headers):
        """List returns empty when no Q&A pairs exist."""
        response = client.get("/api/admin/qa", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["qa_pairs"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50

    def test_returns_qa_pairs(self, client, system_admin_headers, db):
        """List returns Q&A pairs that exist."""
        # Create a Q&A pair directly
        qa = CuratedQA(
            keywords=json.dumps(["test keyword", "another"]),
            answer="Test answer",
            source_reference="Test source",
            confidence=90,
            priority=5,
            enabled=True,
        )
        db.add(qa)
        db.flush()

        response = client.get("/api/admin/qa", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["qa_pairs"]) == 1
        assert data["qa_pairs"][0]["keywords"] == ["test keyword", "another"]
        assert data["qa_pairs"][0]["answer"] == "Test answer"
        assert data["qa_pairs"][0]["source_reference"] == "Test source"

    def test_pagination_works(self, client, system_admin_headers, db):
        """List respects pagination parameters."""
        # Create multiple Q&A pairs
        for i in range(5):
            qa = CuratedQA(
                keywords=json.dumps([f"keyword{i}"]),
                answer=f"Answer {i}",
                source_reference=f"Source {i}",
            )
            db.add(qa)
        db.flush()

        response = client.get("/api/admin/qa?page=1&page_size=2", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["qa_pairs"]) == 2
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["page_size"] == 2

    def test_filters_by_enabled(self, client, system_admin_headers, db):
        """List filters by enabled status."""
        qa_enabled = CuratedQA(
            keywords=json.dumps(["enabled"]),
            answer="Enabled answer",
            source_reference="Source",
            enabled=True,
        )
        qa_disabled = CuratedQA(
            keywords=json.dumps(["disabled"]),
            answer="Disabled answer",
            source_reference="Source",
            enabled=False,
        )
        db.add(qa_enabled)
        db.add(qa_disabled)
        db.flush()

        response = client.get("/api/admin/qa?enabled=true", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["qa_pairs"]) == 1
        assert data["qa_pairs"][0]["enabled"] is True

    def test_search_filters_keywords(self, client, system_admin_headers, db):
        """List filters by keyword search."""
        qa1 = CuratedQA(
            keywords=json.dumps(["vacation policy"]),
            answer="Answer 1",
            source_reference="Source 1",
        )
        qa2 = CuratedQA(
            keywords=json.dumps(["health benefits"]),
            answer="Answer 2",
            source_reference="Source 2",
        )
        db.add(qa1)
        db.add(qa2)
        db.flush()

        response = client.get("/api/admin/qa?search=vacation", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["qa_pairs"]) == 1
        assert "vacation" in data["qa_pairs"][0]["keywords"][0]

    def test_requires_system_admin(self, client, customer_admin_headers):
        """List requires system_admin role (customer_admin is not sufficient)."""
        response = client.get("/api/admin/qa", headers=customer_admin_headers)
        assert response.status_code == 403

    def test_requires_auth(self, client):
        """List requires authentication."""
        response = client.get("/api/admin/qa")
        assert response.status_code == 401


class TestCreateCuratedQA:
    """Tests for POST /api/admin/qa endpoint."""

    def test_creates_qa_pair(self, client, system_admin_headers, db):
        """Creates a Q&A pair successfully."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["test keyword", "another keyword"],
                "answer": "This is the answer.",
                "source_reference": "HR Policy Manual Section 5.2",
                "confidence": 90,
                "priority": 10,
            },
            headers=system_admin_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["keywords"] == ["test keyword", "another keyword"]
        assert data["answer"] == "This is the answer."
        assert data["source_reference"] == "HR Policy Manual Section 5.2"
        assert data["confidence"] == 90
        assert data["priority"] == 10
        assert data["enabled"] is True
        assert data["access_count"] == 0
        assert "id" in data

    def test_syncs_keywords_to_lookup_table(self, client, system_admin_headers, db):
        """Creating Q&A syncs keywords to curated_qa_keywords table."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["paid time off", "pto"],
                "answer": "PTO policy answer.",
                "source_reference": "Policy 123",
            },
            headers=system_admin_headers,
        )
        assert response.status_code == 201
        qa_id = response.json()["id"]

        # Check keywords were synced
        keywords = db.query(CuratedQAKeyword).filter(CuratedQAKeyword.qa_id == qa_id).all()
        keyword_values = {kw.keyword for kw in keywords}
        # "paid time off" should tokenize to: paid, time, off
        # "pto" should tokenize to: pto
        assert "paid" in keyword_values
        assert "time" in keyword_values
        assert "off" in keyword_values
        assert "pto" in keyword_values

    def test_sanitizes_html(self, client, system_admin_headers):
        """Creating Q&A sanitizes HTML in answer field."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["test"],
                "answer": '<p>Safe content</p><script>alert("xss")</script>',
                "source_reference": "Source",
            },
            headers=system_admin_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "<script>" not in data["answer"]
        assert "<p>Safe content</p>" in data["answer"]

    def test_rejects_empty_source_reference(self, client, system_admin_headers):
        """Creating Q&A rejects empty source_reference."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["test"],
                "answer": "Answer",
                "source_reference": "",
            },
            headers=system_admin_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_rejects_whitespace_source_reference(self, client, system_admin_headers):
        """Creating Q&A rejects whitespace-only source_reference."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["test"],
                "answer": "Answer",
                "source_reference": "   ",
            },
            headers=system_admin_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_requires_system_admin(self, client, customer_admin_headers):
        """Creating Q&A requires system_admin role (customer_admin is not sufficient)."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["test"],
                "answer": "Answer",
                "source_reference": "Source",
            },
            headers=customer_admin_headers,
        )
        assert response.status_code == 403

    def test_requires_auth(self, client):
        """Creating Q&A requires authentication."""
        response = client.post(
            "/api/admin/qa",
            json={
                "keywords": ["test"],
                "answer": "Answer",
                "source_reference": "Source",
            },
        )
        assert response.status_code == 401


class TestUpdateCuratedQA:
    """Tests for PUT /api/admin/qa/{qa_id} endpoint."""

    def test_updates_qa_pair(self, client, system_admin_headers, db):
        """Updates a Q&A pair successfully."""
        qa = CuratedQA(
            keywords=json.dumps(["original"]),
            answer="Original answer",
            source_reference="Original source",
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        response = client.put(
            f"/api/admin/qa/{qa_id}",
            json={
                "keywords": ["updated keyword"],
                "answer": "Updated answer",
            },
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["keywords"] == ["updated keyword"]
        assert data["answer"] == "Updated answer"
        assert data["source_reference"] == "Original source"  # Unchanged

    def test_partial_update(self, client, system_admin_headers, db):
        """Updates only provided fields."""
        qa = CuratedQA(
            keywords=json.dumps(["original"]),
            answer="Original answer",
            source_reference="Original source",
            priority=5,
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        response = client.put(
            f"/api/admin/qa/{qa_id}",
            json={"priority": 100},
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["keywords"] == ["original"]  # Unchanged
        assert data["priority"] == 100

    def test_resyncs_keywords_on_update(self, client, system_admin_headers, db):
        """Updating keywords re-syncs the lookup table."""
        qa = CuratedQA(
            keywords=json.dumps(["old keyword"]),
            answer="Answer",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        # Add old keywords manually
        db.add(CuratedQAKeyword(qa_id=qa_id, keyword="old", original_keyword="old keyword"))
        db.flush()

        response = client.put(
            f"/api/admin/qa/{qa_id}",
            json={"keywords": ["new keyword phrase"]},
            headers=system_admin_headers,
        )
        assert response.status_code == 200

        # Check keywords were re-synced
        keywords = db.query(CuratedQAKeyword).filter(CuratedQAKeyword.qa_id == qa_id).all()
        keyword_values = {kw.keyword for kw in keywords}
        assert "old" not in keyword_values
        assert "new" in keyword_values
        assert "keyword" in keyword_values
        assert "phrase" in keyword_values

    def test_sanitizes_html_on_update(self, client, system_admin_headers, db):
        """Updating answer sanitizes HTML."""
        qa = CuratedQA(
            keywords=json.dumps(["test"]),
            answer="Original",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        response = client.put(
            f"/api/admin/qa/{qa_id}",
            json={"answer": '<strong>Bold</strong><img src="x" onerror="alert(1)">'},
            headers=system_admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "<strong>Bold</strong>" in data["answer"]
        assert "<img" not in data["answer"]
        assert "onerror" not in data["answer"]

    def test_rejects_empty_source_reference(self, client, system_admin_headers, db):
        """Updating with empty source_reference fails."""
        qa = CuratedQA(
            keywords=json.dumps(["test"]),
            answer="Answer",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        response = client.put(
            f"/api/admin/qa/{qa_id}",
            json={"source_reference": ""},
            headers=system_admin_headers,
        )
        assert response.status_code == 400

    def test_returns_404_for_nonexistent(self, client, system_admin_headers):
        """Updating nonexistent Q&A returns 404."""
        response = client.put(
            "/api/admin/qa/nonexistent-id",
            json={"answer": "Updated"},
            headers=system_admin_headers,
        )
        assert response.status_code == 404

    def test_requires_system_admin(self, client, customer_admin_headers, db):
        """Updating Q&A requires system_admin role (customer_admin is not sufficient)."""
        qa = CuratedQA(
            keywords=json.dumps(["test"]),
            answer="Answer",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()

        response = client.put(
            f"/api/admin/qa/{qa.id}",
            json={"answer": "Updated"},
            headers=customer_admin_headers,
        )
        assert response.status_code == 403


class TestDeleteCuratedQA:
    """Tests for DELETE /api/admin/qa/{qa_id} endpoint."""

    def test_deletes_qa_pair(self, client, system_admin_headers, db):
        """Deletes a Q&A pair successfully."""
        qa = CuratedQA(
            keywords=json.dumps(["test"]),
            answer="Answer",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        response = client.delete(f"/api/admin/qa/{qa_id}", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify deleted
        deleted_qa = db.query(CuratedQA).filter(CuratedQA.id == qa_id).first()
        assert deleted_qa is None

    def test_cascades_keyword_deletion(self, client, system_admin_headers, db):
        """Deleting Q&A also deletes associated keywords."""
        qa = CuratedQA(
            keywords=json.dumps(["test keyword"]),
            answer="Answer",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()
        qa_id = qa.id

        # Add keywords
        db.add(CuratedQAKeyword(qa_id=qa_id, keyword="test", original_keyword="test keyword"))
        db.add(CuratedQAKeyword(qa_id=qa_id, keyword="keyword", original_keyword="test keyword"))
        db.flush()

        response = client.delete(f"/api/admin/qa/{qa_id}", headers=system_admin_headers)
        assert response.status_code == 200

        # Verify keywords deleted
        remaining_keywords = (
            db.query(CuratedQAKeyword).filter(CuratedQAKeyword.qa_id == qa_id).all()
        )
        assert len(remaining_keywords) == 0

    def test_returns_404_for_nonexistent(self, client, system_admin_headers):
        """Deleting nonexistent Q&A returns 404."""
        response = client.delete("/api/admin/qa/nonexistent-id", headers=system_admin_headers)
        assert response.status_code == 404

    def test_requires_system_admin(self, client, customer_admin_headers, db):
        """Deleting Q&A requires system_admin role (customer_admin is not sufficient)."""
        qa = CuratedQA(
            keywords=json.dumps(["test"]),
            answer="Answer",
            source_reference="Source",
        )
        db.add(qa)
        db.flush()

        response = client.delete(f"/api/admin/qa/{qa.id}", headers=customer_admin_headers)
        assert response.status_code == 403


class TestInvalidateCuratedQACache:
    """Tests for POST /api/admin/qa/invalidate-cache endpoint."""

    def test_invalidates_cache(self, client, system_admin_headers):
        """Cache invalidation returns success."""
        response = client.post("/api/admin/qa/invalidate-cache", headers=system_admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cache" in data["message"].lower()

    def test_requires_system_admin(self, client, customer_admin_headers):
        """Cache invalidation requires system_admin role (customer_admin is not sufficient)."""
        response = client.post("/api/admin/qa/invalidate-cache", headers=customer_admin_headers)
        assert response.status_code == 403

    def test_requires_auth(self, client):
        """Cache invalidation requires authentication."""
        response = client.post("/api/admin/qa/invalidate-cache")
        assert response.status_code == 401
