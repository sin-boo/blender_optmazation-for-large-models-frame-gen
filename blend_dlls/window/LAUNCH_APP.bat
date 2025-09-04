@echo off
title Screen Capture App - Ultimate Launcher
color 0B
setlocal enabledelayedexpansion

REM Change to script directory
cd /d "%~dp0"

echo.
echo     ╔══════════════════════════════════════════════════════════════════╗
echo     ║                                                                  ║
echo     ║  🖥️   SCREEN CAPTURE APP - ULTIMATE SELF-SUFFICIENT EDITION     ║
echo     ║                                                                  ║
echo     ║  ✨ Zero Setup Required • 🚀 One-Click Launch • 🔧 Auto-Install ║
echo     ║  📱 Modern Dark GUI • 🪟 OBS-Style Capture • 🌐 HTTP Streaming  ║
echo     ║                                                                  ║
echo     ║  👆 This will automatically handle everything for you!          ║
echo     ║                                                                  ║
echo     ╚══════════════════════════════════════════════════════════════════╝
echo.

REM Try the ultimate launcher
set "LAUNCHER_SCRIPT=launch_app.py"
set "PYTHON_FOUND=0"

REM Method 1: Try py launcher (most reliable)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo    🚀 Starting with Python launcher...
    py "%LAUNCHER_SCRIPT%"
    set PYTHON_FOUND=1
    goto :end
)

REM Method 2: Try system python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo    🚀 Starting with system Python...
    python "%LAUNCHER_SCRIPT%"
    set PYTHON_FOUND=1
    goto :end
)

REM Method 3: Try portable Python
if exist "tools\python\python.exe" (
    echo    🚀 Starting with portable Python...
    "tools\python\python.exe" "%LAUNCHER_SCRIPT%"
    set PYTHON_FOUND=1
    goto :end
)

REM Method 4: Ultimate fallback - try to run launcher anyway
echo    ⚙️  No Python found - the launcher will handle installation...
echo.

REM Try any available Python to run the launcher
python "%LAUNCHER_SCRIPT%" 2>nul || py "%LAUNCHER_SCRIPT%" 2>nul || (
    echo    ❌ Cannot start launcher - manual Python installation required
    echo.
    echo    📖 Quick Manual Setup:
    echo       1. Go to https://python.org/downloads/
    echo       2. Download Python 3.7+ 
    echo       3. During install, CHECK "Add Python to PATH"
    echo       4. Run this script again
    echo.
    echo    💡 Alternative: Use INSTALL_AND_RUN.bat for automatic setup
    echo.
    pause
    exit /b 1
)

:end
echo.

if %errorlevel% neq 0 (
    echo    ⚠️  Application ended with issues
    echo    💡 This can happen on first run while setting up
    echo    🔄 Try running again - it usually works the second time
    echo.
    echo    🛠️  If problems persist, check:
    echo       - Internet connection (for downloading dependencies)
    echo       - Antivirus isn't blocking downloads
    echo       - Try running as Administrator
    echo.
    pause
) else (
    echo    ✅ Application completed successfully
    echo.
    timeout /t 2 /nobreak >nul 2>&1
)

endlocal
