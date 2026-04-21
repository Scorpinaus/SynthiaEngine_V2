@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

set VENV_PY=%ROOT%\.venv\Scripts\python.exe
set PYTORCH_ALLOC_CONF=expandable_segments:True
REM --- Hugging Face cache + disable symlinks (fixes WinError 1314) ---
set HF_HUB_DISABLE_SYMLINKS=1

set SYNTHA_LOG_ROLE=api
set SYNTHA_API_START_WORKER=0
start "Synthia API" cmd /k ""%VENV_PY%" -m uvicorn backend.main:app --workers 1 --host 0.0.0.0 --port 8000"

set SYNTHA_LOG_ROLE=render
set SYNTHA_API_START_WORKER=
start "Synthia Renderer" cmd /k ""%VENV_PY%" -m backend.jobs.render_worker"

set SYNTHA_LOG_ROLE=

rem Wait 20 seconds
timeout /t 20 /nobreak >nul

start "Synthia Frontend" cmd /k ""%VENV_PY%" -m http.server 4173 --directory frontend"
start "" "http://127.0.0.1:4173/sd15/text2img.html"

