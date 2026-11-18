@echo off
setlocal
title Drone Oversight
cd /d "%~dp0"

where python >nul 2>nul || ( echo [!] Python 3.11+ not found. & pause & exit /b 1 )
where node   >nul 2>nul || ( echo [!] Node.js 22.12+ not found. & pause & exit /b 1 )

if not exist ".venv\Scripts\python.exe" (
    echo [*] First run: creating Python environment...
    python -m venv .venv || ( pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

fc /b requirements.txt ".venv\.requirements.installed" >nul 2>nul
if errorlevel 1 (
    echo [*] Installing Python packages ^(first run takes a few minutes^)...
    python -m pip install -r requirements.txt || ( pause & exit /b 1 )
    copy /y requirements.txt ".venv\.requirements.installed" >nul
)

rem Dev mode: API (auto-reload) + Vite UI, opens the browser on http://localhost:5173
python run.py --dev

pause
