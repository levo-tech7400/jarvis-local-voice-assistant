@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Jarvis - kurulum ve test

rem Not: Bu dosyada bilerek goto/etiket kullanilmadi. LF satir sonlu batch
rem dosyalarinda cmd etiket aramasi sapittigi icin akis tamamen dogrusal.

set "LOG=%~dp0kurulum_sonuc.log"
echo. > "%LOG%"

echo ============================================================== >> "%LOG%"
echo  Jarvis kurulum + test                                        >> "%LOG%"
echo  %DATE% %TIME%                                               >> "%LOG%"
echo  Klasor: %CD%                                                >> "%LOG%"
echo ============================================================== >> "%LOG%"

echo [1/4] Python ve sanal ortam
set "BASEPY=py -3"
where py >nul 2>nul
if errorlevel 1 set "BASEPY=python"
if not exist ".venv\Scripts\python.exe" %BASEPY% -m venv .venv >> "%LOG%" 2>&1
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -V >> "%LOG%" 2>&1

echo [2/4] Bagimliliklar
"%PY%" -m pip install -r requirements.txt --quiet >> "%LOG%" 2>&1
if errorlevel 1 echo UYARI: bazi paketler kurulamadi, log dosyasina bakin. >> "%LOG%"

echo [3/4] Kod testleri
echo. >> "%LOG%"
echo --- py_compile --- >> "%LOG%"
"%PY%" -m py_compile jarvis.py test_jarvis.py >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo --- unittest --- >> "%LOG%"
"%PY%" -m unittest -v test_jarvis.py >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"

echo [4/4] Ortam kontrolu + ses testi
echo. >> "%LOG%"
echo --- jarvis --check --- >> "%LOG%"
"%PY%" jarvis.py --check >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo --- jarvis --say --- >> "%LOG%"
"%PY%" jarvis.py --say "Merhaba efendim, sistem hazir." >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"
echo --- jarvis --once --- >> "%LOG%"
"%PY%" jarvis.py --once "Tek cumleyle kendini tanit." >> "%LOG%" 2>&1
echo cikis kodu: %errorlevel% >> "%LOG%"

echo. >> "%LOG%"
echo ============================================================== >> "%LOG%"
echo  BITTI - ayrintilar yukarida                                >> "%LOG%"
echo ============================================================== >> "%LOG%"

echo.
echo Bitti. Sonuclar: "%LOG%"
start notepad "%LOG%"
pause
