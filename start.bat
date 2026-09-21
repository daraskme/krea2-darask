@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Krea 2 Studio is not installed yet. Run setup.bat first.
  pause
  exit /b 1
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8189/health' -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  start "" "http://127.0.0.1:8189"
  exit /b 0
)
".venv\Scripts\python.exe" -X utf8 -m krea2_studio.server
if errorlevel 1 pause
