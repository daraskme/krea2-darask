@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. Install it from https://docs.astral.sh/uv/ and run this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" uv venv --python 3.12 .venv
if errorlevel 1 goto :failed
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto :failed
uv pip install --python ".venv\Scripts\python.exe" -r requirements-windows-optimization.txt
if errorlevel 1 goto :failed
uv pip install --python ".venv\Scripts\python.exe" -e . --no-deps
if errorlevel 1 goto :failed
echo Installation completed.
exit /b 0

:failed
echo Installation failed. Review the error above.
pause
exit /b 1
