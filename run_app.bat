@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

set VENV_PY=%ROOT%\.venv\Scripts\python.exe
set PYTORCH_ALLOC_CONF=expandable_segments:True
REM --- Hugging Face cache + disable symlinks (fixes WinError 1314) ---
set HF_HUB_DISABLE_SYMLINKS=1

start "Synthia Backend" cmd /k ""%VENV_PY%" -m uvicorn backend.main:app --workers 1 --host 0.0.0.0 --port 8000"

rem Wait 20 seconds
timeout /t 20 /nobreak >nul

start "Synthia Frontend" cmd /k ""%VENV_PY%" -m http.server 4173 --directory frontend"
start "" "http://127.0.0.1:4173/sd15.html"

