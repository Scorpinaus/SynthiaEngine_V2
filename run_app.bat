@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

set VENV_PY=%ROOT%\.venv\Scripts\python.exe
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
start "Synthia Backend" cmd /k ""%VENV_PY%" -m uvicorn backend.main:app --workers 1 --host 0.0.0.0 --port 8000"

rem Wait 20 seconds
timeout /t 20 /nobreak >nul

start "Synthia Frontend" cmd /k ""%VENV_PY%" -m http.server 4173 --directory frontend"
start "" "http://127.0.0.1:4173/index.html"
