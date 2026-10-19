@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Jarvis - hizli test

set "LOG=%~dp0kurulum_sonuc.log"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ========================================== > "%LOG%"
echo  Jarvis hizli test   %DATE% %TIME% >> "%LOG%"
echo ========================================== >> "%LOG%"

"%PY%" -V >> "%LOG%" 2>&1
echo. >> "%LOG%"

echo --- 1/5 py_compile --- >> "%LOG%"
"%PY%" -m py_compile jarvis.py test_jarvis.py >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo. >> "%LOG%"

echo --- 2/5 unittest --- >> "%LOG%"
"%PY%" -m unittest -v test_jarvis.py >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo. >> "%LOG%"

echo --- 3/5 ortam kontrolu --- >> "%LOG%"
"%PY%" jarvis.py --check >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo. >> "%LOG%"

echo --- 4/5 TTS testi --- >> "%LOG%"
"%PY%" jarvis.py --say "Merhaba efendim, ses testi bir iki uc." >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo. >> "%LOG%"

echo --- 5/5 LM Studio tek soru --- >> "%LOG%"
"%PY%" jarvis.py --once "Tek cumleyle kendini tanit." >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo. >> "%LOG%"

echo ========================================== >> "%LOG%"
echo  BITTI >> "%LOG%"
echo ========================================== >> "%LOG%"

start notepad "%LOG%"
