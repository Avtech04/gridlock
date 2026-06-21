@echo off
set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

if not exist ".venv" (
  python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install -r backend\requirements.txt

cd backend
echo Open http://localhost:8000/ui/
python -m uvicorn main:app --host 0.0.0.0 --port 8000
