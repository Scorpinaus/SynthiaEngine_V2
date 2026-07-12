@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

set VENV_PY=%ROOT%\.venv\Scripts\python.exe
set PYTORCH_ALLOC_CONF=expandable_segments:True
REM --- Hugging Face cache + disable symlinks (fixes WinError 1314) ---
set HF_HUB_DISABLE_SYMLINKS=1

if /i "%~1"=="api" (
    set SYNTHA_LOG_ROLE=api
    set SYNTHA_API_START_WORKER=0
    "%VENV_PY%" -m uvicorn backend.main:app --workers 1 --host 127.0.0.1 --port 8000
    exit /b
)

if /i "%~1"=="render" (
    set SYNTHA_LOG_ROLE=render
    set SYNTHA_API_START_WORKER=
    "%VENV_PY%" -m backend.jobs.render_worker
    exit /b
)

if /i "%~1"=="frontend" (
    set SYNTHA_LOG_ROLE=
    set SYNTHA_API_START_WORKER=
    "%VENV_PY%" -m http.server 4173 --directory frontend
    exit /b
)

if /i "%~1"=="frontend_wait" (
    timeout /t 20 /nobreak >nul
    "%~f0" frontend
    exit /b
)

where wt.exe >nul 2>nul
if errorlevel 1 (
    start "Synthia API" cmd.exe /k ""%~f0" api"
    start "Synthia Renderer" cmd.exe /k ""%~f0" render"
) else (
    start "" wt.exe new-tab --title "Synthia API" cmd.exe /k ""%~f0" api"
    timeout /t 2 /nobreak >nul
    wt.exe -w 0 new-tab --title "Synthia Renderer" cmd.exe /k ""%~f0" render"
)

where wt.exe >nul 2>nul
if errorlevel 1 (
    start "Synthia Frontend" cmd.exe /k ""%~f0" frontend_wait"
) else (
    wt.exe -w 0 new-tab --title "Synthia Frontend" cmd.exe /k ""%~f0" frontend_wait"
)

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 22; Start-Process 'http://127.0.0.1:4173/sd15/text2img.html'"

