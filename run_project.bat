@echo off
title BioSpan AI - Biomedical NER & Clinical Ontology Engine
echo =====================================================================
echo                     🧬 BioSpan AI Launcher
echo     Biomedical Named Entity Recognition & Clinical Ontology Engine
echo =====================================================================
echo.
echo [1/3] Checking Python 3.10 Environment...
py -3.10 --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python 3.10 is required. Please install Python 3.10.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Starting BioSpan AI FastAPI Engine...
echo Server running at: http://127.0.0.1:8000
echo Documentation at:  http://127.0.0.1:8000/docs
echo.
echo [3/3] Opening BioSpan AI Dashboard in your default browser...
start http://127.0.0.1:8000

py -3.10 server.py
pause
