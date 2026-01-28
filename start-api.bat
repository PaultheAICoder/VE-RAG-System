@echo off
echo Starting AI Ready RAG API Server...
echo.

cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install API dependencies if needed
pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings bcrypt pyjwt --quiet

:: Start the server
echo.
echo Server starting at http://localhost:8000
echo API docs at http://localhost:8000/api/docs
echo.
python -m uvicorn ai_ready_rag.main:app --reload --port 8000
