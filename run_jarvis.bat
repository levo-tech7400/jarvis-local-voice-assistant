@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "%~dp0jarvis.py" %*
) else (
  python "%~dp0jarvis.py" %*
)

if errorlevel 1 (
  echo.
  echo Jarvis hata kodu: %errorlevel%
  pause
)
