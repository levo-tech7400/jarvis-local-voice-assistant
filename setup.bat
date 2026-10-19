@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   Jarvis - kurulum
echo ============================================

set "PY=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"

%PY% --version >nul 2>nul
if errorlevel 1 (
  echo [HATA] Python bulunamadi. https://www.python.org/downloads/ adresinden kurun.
  echo        Kurulumda "Add python.exe to PATH" secenegini isaretleyin.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Sanal ortam olusturuluyor...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [HATA] Sanal ortam olusturulamadi.
    pause
    exit /b 1
  )
) else (
  echo [1/3] Sanal ortam zaten var: .venv
)

echo [2/3] pip guncelleniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo [3/3] Bagimliliklar kuruluyor (birkac dakika surebilir)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [HATA] Bagimliliklar kurulamadi. Internet baglantisini kontrol edin.
  pause
  exit /b 1
)

echo.
echo Kurulum tamamlandi.
echo.
echo GPU'lu konusma tanima icin (NVIDIA): pip install -r requirements-gpu.txt
echo Test icin:  run_jarvis.bat --check
pause
