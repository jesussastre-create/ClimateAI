@echo off
chcp 65001 >nul
title ClimateAI - Servidor Local de Alto Rendimiento (http://127.0.0.1:8080)
cd /d "%~dp0"

echo ========================================================
echo  ClimateAI: Resiliencia Energetica y Demanda HVAC
echo ========================================================

:: 1. Verificar y activar entorno virtual si existe
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activando entorno virtual .venv...
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activando entorno virtual venv...
    call "venv\Scripts\activate.bat"
)

:: 2. Verificar que Python este disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no se encontro en el PATH del sistema.
    pause
    exit /b 1
)

:: 3. Verificar dependencias criticas e instalarlas si faltan
python -c "import fastapi, uvicorn, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Instalando dependencias desde requirements.txt...
    python -m pip install -r requirements.txt
) else (
    echo [OK] Entorno y dependencias verificadas.
)

:: 4. Ejecutar servidor en primer plano (abre automaticamente http://127.0.0.1:8080)
echo [START] Levantando main.py en primer plano...
python main.py
pause
