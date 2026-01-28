@echo off
echo Running AI Ready RAG Tests...
echo.

cd /d "%~dp0"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install test dependencies if needed
pip install pytest pytest-cov httpx --quiet

:: Run tests
echo.
pytest %*
