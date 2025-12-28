@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

start "Synthia Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

rem Wait 20 seconds
timeout /t 20 /nobreak >nul

start "Synthia Frontend" cmd /k "python -m http.server 4173 --directory frontend"
start "" "http://127.0.0.1:4173/index.html"
