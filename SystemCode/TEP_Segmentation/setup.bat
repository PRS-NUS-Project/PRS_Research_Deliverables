@echo off
REM ===============================
REM TEP Segmentation - Development Environment Setup (Windows)
REM ===============================

REM Check for Python 3.12
python --version | findstr "3.12" >nul
IF %ERRORLEVEL% NEQ 0 (
    echo Please install Python 3.12 before running this script.
    exit /b 1
)

REM Create virtual environment if it doesn't exist
IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    py -3.12 -m venv .venv
) ELSE (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements-dev.txt

echo.
echo ===============================
echo Setup complete!
echo ===============================