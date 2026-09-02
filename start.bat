@echo off
chcp 65001 >nul
title NeuroScan AI
echo.
echo  [AI] Brain Tumor MRI Classifier - Flask API Server
echo  ====================================================
echo.
cd /d "%~dp0"
echo [*] Kiem tra Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python!
    pause
    exit /b 1
)
echo [OK] Python da san sang.

if exist "venv\Scripts\activate.bat" (
    echo [*] Kich hoat venv...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [*] Kich hoat .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [!] Dung Python he thong.
)

set CHECKPOINT=checkpoints\efficientnet_v2_s\20260320_091400\efficientnet_v2_s_best.pth
if not exist "%CHECKPOINT%" (
    echo [LOI] Khong tim thay model: %CHECKPOINT%
    pause
    exit /b 1
)
echo [OK] Tim thay model: %CHECKPOINT%

python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [!] Cai dat thu vien...
    pip install -r requirements.txt
)

echo.
echo ============================================================
echo  Flask server: http://localhost:5000
echo  Nhan Ctrl+C de dung
echo ============================================================
echo.

python app.py --effnet_v2 "%CHECKPOINT%"
echo.
pause
