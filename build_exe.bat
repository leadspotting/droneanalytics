@echo off
setlocal
cd /d "%~dp0"
echo [*] Building frontend...
call npm run build
if errorlevel 1 exit /b 1

echo [*] Installing PyInstaller...
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1

echo [*] Building Executable...
python -m PyInstaller -y --distpath "./release" DroneOversight.spec
if errorlevel 1 exit /b 1

echo [*] Done! Executable is located in the "release" folder.
pause
