@echo off
REM Auto-activate venv and run Python commands
REM Usage: run.bat <python-command>
REM Example: run.bat main.py
REM Example: run.bat -c "import anthropic; print('OK')"

setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0
set VENV_PYTHON=%PROJECT_ROOT%venv\Scripts\python.exe

REM Check if venv exists
if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found at %PROJECT_ROOT%venv
    echo Run: python -m venv venv
    exit /b 1
)

REM Load .env if present
if exist "%PROJECT_ROOT%.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_ROOT%.env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            set "%%a=%%b"
        )
    )
)

REM Run Python with venv
"%VENV_PYTHON%" %*
exit /b %ERRORLEVEL%
