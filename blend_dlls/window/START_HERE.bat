@echo off
title Screen Capture Application
color 0B

echo.
echo     ╔══════════════════════════════════════════════════════════════════╗
echo     ║                                                                  ║
echo     ║  🖥️   SCREEN CAPTURE APPLICATION - READY TO RUN!               ║
echo     ║                                                                  ║
echo     ║  📱 Modern Dark Theme • 🪟 OBS-Style Window Capture             ║
echo     ║  🌐 HTTP Server • 💾 Local Storage • ⚙️ Zero Config            ║
echo     ║                                                                  ║
echo     ║  👆 Just double-click this file to start!                       ║
echo     ║                                                                  ║
echo     ╚══════════════════════════════════════════════════════════════════╝
echo.
echo    🚀 Initializing application...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if this is the first run
if not exist "tools" (
    echo    📦 First run detected - setting up environment...
    echo.
)

REM Try different Python commands in order of preference
set PYTHON_FOUND=0

REM 1. Try py launcher (most reliable on Windows)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✅ Using Python launcher
    py run.py
    set PYTHON_FOUND=1
    goto :end
)

REM 2. Try python command
python --version >nul 2>&1  
if %errorlevel% equ 0 (
    echo    ✅ Using system Python
    python run.py
    set PYTHON_FOUND=1
    goto :end
)

REM 3. Try portable Python if exists
if exist "tools\python\python.exe" (
    echo    ✅ Using portable Python
    "tools\python\python.exe" run.py
    set PYTHON_FOUND=1
    goto :end
)

REM 4. Last resort - try to run anyway (run.py will handle installation)
echo    ⚠️  Python not found in standard locations
echo    📥 Attempting to start anyway - will try to install Python...
echo.

python run.py
if %errorlevel% neq 0 (
    py run.py
    if %errorlevel% neq 0 (
        echo.
        echo    ❌ Could not start application
        echo    📖 Manual Python installation may be required
        echo    🔗 Download from: https://python.org
        echo.
        echo    💡 Make sure to check "Add Python to PATH" during installation
        echo.
    )
)

:end
if %errorlevel% neq 0 (
    echo.
    echo    ⚠️  Application ended with issues
    echo    💡 This is normal on the first run while setting up
    echo    🔄 Try running again - it should work the second time
    echo.
    pause
) else (
    echo.
    echo    👋 Application closed normally
    echo.
    timeout /t 3 /nobreak >nul
)
