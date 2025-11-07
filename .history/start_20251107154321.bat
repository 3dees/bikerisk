@echo off
echo E-Bike Standards Requirement Extractor
echo ==========================================
echo.

REM Check if venv exists, create if not
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in virtual environment
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing dependencies...
pip install -r requirements.txt >nul 2>&1

REM Start FastAPI in background (with venv activated)
echo Starting FastAPI backend on http://localhost:8000
start "FastAPI Backend" cmd /k "cd /d %CD% && call venv\Scripts\activate.bat && python main.py"

REM Wait for FastAPI to start
timeout /t 3 /nobreak >nul

REM Start Streamlit
echo Starting Streamlit UI on http://localhost:8501
echo.
echo Press Ctrl+C in each window to stop the servers
echo.
streamlit run app.py

pause
