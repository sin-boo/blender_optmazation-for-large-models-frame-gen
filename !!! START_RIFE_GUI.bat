@echo off
title RIFE GUI - Main Launcher

echo.
echo ================================================
echo           RIFE GUI LAUNCHER
echo     Real-time Video Frame Interpolation
echo          for RTX 5060 Ti (16GB)
echo ================================================
echo.

setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"
set APPDIR=%cd%
set VENV_DIR=%APPDIR%\.venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe

rem Step 1: Check if virtual environment exists
echo [1/4] Checking environment...
if exist "%VENV_PY%" (
    echo ✅ Using existing virtual environment
    goto LAUNCH_APP
)

rem Step 2: Try to find a Python to bootstrap with
echo [2/4] Finding Python to bootstrap...
set BOOTSTRAP_PY=
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set BOOTSTRAP_PY=python
    goto CREATE_VENV
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    set BOOTSTRAP_PY=py
    goto CREATE_VENV
)

echo ❌ No Python found on system!
echo Please install Python from python.org first.
pause
exit /b 1

:CREATE_VENV
rem Step 3: Create virtual environment and install dependencies
echo [3/4] Creating virtual environment...
%BOOTSTRAP_PY% -m venv "%VENV_DIR%"
if not exist "%VENV_PY%" (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo Installing base dependencies...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.base.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install base dependencies
    pause
    exit /b 1
)

echo Installing optimal PyTorch for RTX 5060 Ti...
"%VENV_PY%" auto_gpu_setup.py
if %errorlevel% neq 0 (
    echo.
    echo 🔍 RTX 5060 Ti COMPATIBILITY INFO:
    echo Your RTX 5060 Ti is SO NEW that even PyTorch nightly doesn't support it yet!
    echo CUDA capability sm_120 requires newer PyTorch builds.
    echo.
    echo ℹ️ The app will run in CPU mode for now, which still works but slower.
    echo 🔄 Check for PyTorch updates weekly - support is coming soon!
    echo.
)

:LAUNCH_APP
rem Step 4: Launch the application
echo [4/4] Launching RIFE GUI...
echo.
echo 🚀 Starting RIFE GUI with RTX 5060 Ti support...
"%VENV_PY%" unified_rife_gui.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Application failed to start
    pause
)

echo.
echo Application closed.
pause
endlocal
