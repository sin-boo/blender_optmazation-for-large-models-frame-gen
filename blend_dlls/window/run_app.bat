@echo off
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║      🖥️  SCREEN CAPTURE APP - SELF-SUFFICIENT EDITION      ║
echo ║                                                              ║
echo ║         🚀 Starting automatic setup and launch...           ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Change to script directory
cd /d "%~dp0"

REM Try to find Python
set PYTHON_CMD=
set PYTHON_FOUND=0

REM Check if Python is in PATH
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    set PYTHON_FOUND=1
    echo ✅ Found Python in PATH
    goto :run_app
)

REM Try py launcher (Windows)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    set PYTHON_FOUND=1
    echo ✅ Found Python via py launcher
    goto :run_app
)

REM Try common installation paths
echo 🔍 Searching for Python in common locations...

if exist "C:\Python3\python.exe" (
    set PYTHON_CMD="C:\Python3\python.exe"
    set PYTHON_FOUND=1
    echo ✅ Found Python at C:\Python3
    goto :run_app
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*\python.exe" (
    for /f "delims=" %%i in ('dir /b "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*\python.exe" 2^>nul') do (
        set PYTHON_CMD="C:\Users\%USERNAME%\AppData\Local\Programs\Python\%%i"
        set PYTHON_FOUND=1
        echo ✅ Found Python in user directory
        goto :run_app
    )
)

if exist "C:\Program Files\Python*\python.exe" (
    for /f "delims=" %%i in ('dir /b "C:\Program Files\Python*\python.exe" 2^>nul') do (
        set PYTHON_CMD="C:\Program Files\%%i"
        set PYTHON_FOUND=1
        echo ✅ Found Python in Program Files
        goto :run_app
    )
)

REM Check portable Python
if exist "tools\python\python.exe" (
    set PYTHON_CMD="tools\python\python.exe"
    set PYTHON_FOUND=1
    echo ✅ Found portable Python
    goto :run_app
)

REM If no Python found, try to run with basic Python anyway
if %PYTHON_FOUND% equ 0 (
    echo ❌ Python not found anywhere!
    echo.
    echo 📥 This application can install Python automatically.
    echo    Just run it anyway and it will guide you through setup.
    echo.
    set PYTHON_CMD=python
)

:run_app
echo.
echo 🚀 Starting application...
echo ─────────────────────────────────────────────────────────
echo.

REM Run the ultimate launcher
%PYTHON_CMD% run.py

REM Handle exit
if %errorlevel% neq 0 (
    echo.
    echo ❌ Application exited with error code %errorlevel%
    echo 💡 Try running again - the first run might install dependencies
    echo.
    pause
)
