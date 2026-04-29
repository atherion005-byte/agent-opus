@echo off
title Agent Opus Setup
echo ============================================
echo   Agent Opus — Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause & exit /b 1
)

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo.
echo   To start the Clip Generator:
echo     python clipping_tool\app.py
echo.
echo   (Optional) Install Ollama for AI analysis:
echo     https://ollama.com  then:  ollama pull llama3
echo ============================================
pause
