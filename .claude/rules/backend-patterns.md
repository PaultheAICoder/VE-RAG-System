# Backend Patterns - AI Ready RAG

## FastAPI Patterns

### Route Structure
```python
# Routes go in api/*.py
# Each route file has its own router
router = APIRouter()

# Use Depends for auth
@router.get("/items")
async def list_items(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ...
```

### Response Models
```python
# Always use Pydantic models for responses
class ItemResponse(BaseModel):
    id: str
    name: str

    class Config:
        from_attributes = True  # For SQLAlchemy models
```

### Error Handling
```python
# Use HTTPException with appropriate status codes
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Item not found"
)
```

## SQLAlchemy Patterns

### Models
```python
# Models in db/models.py
# Use UUID strings for IDs
def generate_uuid() -> str:
    return str(uuid.uuid4())

class MyModel(Base):
    __tablename__ = "my_models"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Session Management
```python
# Use dependency injection for sessions
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Service Layer

### Pattern
```python
# Business logic in services/*.py
# Services are stateless, receive db session

class MyService:
    def __init__(self, db: Session):
        self.db = db

    def get_item(self, item_id: str) -> Item | None:
        return self.db.query(Item).filter(Item.id == item_id).first()
```

## Authentication

### JWT Pattern
```python
# Create tokens with user info in payload
token = create_access_token(data={
    "sub": user.id,
    "email": user.email,
    "role": user.role
})

# Decode and validate in dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    payload = decode_token(credentials.credentials)
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    return user
```

## Async Patterns

### Async Services
```python
# Use async for I/O operations (Ollama, Qdrant)
async def embed(self, text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
    return response.json()["embedding"]
```

## Testing Patterns

### Fixtures
```python
@pytest.fixture(scope="function")
def db():
    # Fresh database per test
    # Use transaction rollback for isolation

@pytest.fixture
def client(db):
    # TestClient with db override
    app.dependency_overrides[get_db] = lambda: db
```

### Test Structure
```python
class TestFeature:
    def test_happy_path(self, client, admin_headers):
        response = client.get("/api/items", headers=admin_headers)
        assert response.status_code == 200

    def test_unauthorized(self, client):
        response = client.get("/api/items")
        assert response.status_code == 401
```

## Code Style

- Use type hints everywhere
- Use `list[str]` not `List[str]` (Python 3.12+)
- Use `str | None` not `Optional[str]`
- Run `ruff check` and `ruff format` before committing
- Docstrings for public functions
