@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PY=%~dp0python\python.exe"

if not exist "%PY%" (
    echo [ERROR] python\python.exe not found. The zip was not extracted completely.
    echo Please extract the whole archive to a normal folder and run again.
    echo.
    pause
    exit /b 1
)

if exist ".livetranslate-ready" goto launch

echo First run: downloading PyTorch for your GPU. This can take several minutes...
"%PY%" "%~dp0first_run.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. See messages above, then run LiveTranslate.bat again.
    pause
    exit /b 1
)

:launch
echo Starting LiveTranslate...
"%PY%" main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
