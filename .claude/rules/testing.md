# Testing Patterns - AI Ready RAG

## Test Framework

- **Framework**: pytest 9.0 with pytest-asyncio
- **Coverage**: pytest-cov
- **Async**: All tests support async via `pytest.mark.asyncio`

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Quick run
pytest tests/ -q

# With coverage
pytest tests/ --cov=ai_ready_rag --cov-report=term-missing

# Specific test file
pytest tests/test_documents.py -v

# Specific test class
pytest tests/test_auth.py::TestLogin -v

# Skip slow tests (if tagged)
pytest tests/ -m "not slow"
```

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures (db, client, auth)
├── test_auth.py         # Authentication tests
├── test_chat_api.py     # Chat session/message tests
├── test_documents.py    # Document CRUD tests
├── test_health.py       # Health endpoint tests
├── test_profile_pipeline.py  # Profile/backend tests
├── test_rag_service.py  # RAG service tests
├── test_users.py        # User management tests
├── test_vector_service.py    # Vector service tests
└── test_vector_utils.py      # Utility function tests
```

## Fixture Patterns

### Database Fixture
```python
@pytest.fixture(scope="function")
def db():
    """Fresh database per test with rollback."""
    # Uses in-memory SQLite for speed
    # Each test gets isolated transaction
```

### Client Fixture
```python
@pytest.fixture
def client(db):
    """TestClient with db override."""
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)
```

### Auth Fixtures
```python
@pytest.fixture
def admin_headers(client, db):
    """Get auth headers for admin user."""
    # Creates admin, logs in, returns Bearer token

@pytest.fixture
def user_headers(client, db):
    """Get auth headers for regular user."""
```

## Test Conventions

### Naming
- Test files: `test_<module>.py`
- Test classes: `TestFeatureName`
- Test methods: `test_<scenario>_<expected_behavior>`

### Examples
```python
class TestDocumentUpload:
    def test_upload_requires_admin(self, client, user_headers):
        """Non-admin users cannot upload."""

    def test_upload_valid_document(self, client, admin_headers):
        """Admin can upload valid document."""

    def test_upload_validates_file_type(self, client, admin_headers):
        """Invalid file types are rejected."""
```

### Assertions
```python
# Status codes
assert response.status_code == 200
assert response.status_code == status.HTTP_201_CREATED

# Response structure
data = response.json()
assert "id" in data
assert data["status"] == "ready"

# Lists
assert len(data["documents"]) == 3
assert any(d["filename"] == "test.pdf" for d in data["documents"])
```

## Async Test Patterns

```python
@pytest.mark.asyncio
async def test_vector_search():
    """Test async vector operations."""
    service = VectorService(...)
    await service.initialize()
    results = await service.search(query="test", user_tags=["hr"])
    assert len(results) > 0
```

## Mocking Patterns

### Mock External Services
```python
@pytest.fixture
def mock_ollama(mocker):
    """Mock Ollama API responses."""
    return mocker.patch(
        "ai_ready_rag.services.rag_service.httpx.AsyncClient.post",
        return_value=MockResponse({"response": "test"})
    )
```

### Mock Settings
```python
@pytest.fixture
def mock_settings(mocker):
    """Override settings for test."""
    return mocker.patch(
        "ai_ready_rag.config.get_settings",
        return_value=Settings(enable_ocr=False)
    )
```

## Pre-Commit Testing

Before committing:
```bash
# Quick verification
ruff check ai_ready_rag tests && pytest tests/ -q

# Full check
ruff check ai_ready_rag tests && ruff format --check ai_ready_rag tests && pytest tests/ -v
```

## Test Data

- Use minimal test data
- Clean up after tests (fixtures handle this)
- Don't depend on external services (mock them)
- Use in-memory SQLite for speed
