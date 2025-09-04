@echo off
title Screen Capture App - Ultimate Working Installer
color 0B
setlocal enabledelayedexpansion

echo.
echo     ╔══════════════════════════════════════════════════════════════════╗
echo     ║                                                                  ║
echo     ║  🖥️   SCREEN CAPTURE APP - ULTIMATE WORKING INSTALLER          ║
echo     ║                                                                  ║
echo     ║  🚀 Downloads FULL Python + Dependencies + Modern GUI           ║
echo     ║  ✨ Includes tkinter for GUI support                            ║
echo     ║                                                                  ║
echo     ║  ⏱️ Takes 3-7 minutes depending on internet speed               ║
echo     ║                                                                  ║
echo     ╚══════════════════════════════════════════════════════════════════╝
echo.

REM Change to script directory
cd /d "%~dp0"

echo 🔍 Checking if Python is already available...

REM Try to find existing Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python found! Checking if tkinter is available...
    python -c "import tkinter" >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ Python with GUI support found! Skipping installation.
        set PYTHON_CMD=python
        goto :install_deps
    ) else (
        echo ⚠️ Python found but missing GUI support (tkinter)
    )
)

py --version >nul 2>&1  
if %errorlevel% equ 0 (
    echo ✅ Python found via py launcher! Checking GUI support...
    py -c "import tkinter" >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ Python with GUI support found via py launcher!
        set PYTHON_CMD=py
        goto :install_deps
    ) else (
        echo ⚠️ Python found but missing GUI support (tkinter)
    )
)

if exist "tools\python\python.exe" (
    echo ✅ Checking existing portable Python...
    "tools\python\python.exe" -c "import tkinter" >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ Portable Python with GUI support found!
        set PYTHON_CMD="tools\python\python.exe"
        goto :install_deps
    ) else (
        echo ⚠️ Portable Python found but missing GUI support
        echo 🧹 Cleaning old installation...
        rmdir /s /q "tools\python" >nul 2>&1
    )
)

echo 📥 Installing full Python with GUI support...
echo.

REM Create tools directory
if not exist "tools" mkdir "tools"
if not exist "tools\downloads" mkdir "tools\downloads"

echo 🌐 Downloading FULL Python installer from python.org...
echo    💡 Using full installer to ensure tkinter/GUI support

REM Determine architecture and download URL for FULL Python
set ARCH=amd64
if "%PROCESSOR_ARCHITECTURE%"=="x86" set ARCH=win32

set PYTHON_VERSION=3.11.9
REM Use the full installer instead of embedded
set DOWNLOAD_URL=https://www.python.org/ftp/python/!PYTHON_VERSION!/python-!PYTHON_VERSION!-!ARCH!.exe
set DOWNLOAD_FILE=tools\downloads\python_installer.exe

echo    📍 URL: %DOWNLOAD_URL%
echo    💾 Downloading to: %DOWNLOAD_FILE%
echo.

REM Create PowerShell download script
echo $ErrorActionPreference = "Stop" > temp_download.ps1
echo [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 >> temp_download.ps1
echo Write-Host "Starting Python download..." >> temp_download.ps1
echo try { >> temp_download.ps1
echo     $webClient = New-Object System.Net.WebClient >> temp_download.ps1
echo     $webClient.DownloadFile('%DOWNLOAD_URL%', '%DOWNLOAD_FILE%') >> temp_download.ps1
echo     Write-Host "Download completed successfully!" >> temp_download.ps1
echo     exit 0 >> temp_download.ps1
echo } catch { >> temp_download.ps1
echo     Write-Host "Download failed:" $_.Exception.Message >> temp_download.ps1
echo     exit 1 >> temp_download.ps1
echo } >> temp_download.ps1

REM Execute download
powershell -ExecutionPolicy Bypass -File temp_download.ps1
del temp_download.ps1 >nul 2>&1

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to download Python automatically.
    echo 🔧 Please install Python manually:
    echo    1. Go to https://python.org
    echo    2. Download Python 3.7 or newer  
    echo    3. Make sure to check "Add Python to PATH"
    echo    4. Run this script again
    echo.
    pause
    exit /b 1
)

echo 🔧 Installing Python to portable directory...
echo    💡 This will install Python locally without affecting your system

REM Install Python silently to our tools directory
"%DOWNLOAD_FILE%" /quiet InstallAllUsers=0 TargetDir="%cd%\tools\python" Include_pip=1 Include_tcltk=1 Include_test=0 Include_doc=0

if %errorlevel% equ 0 (
    echo ✅ Python installed successfully!
    set PYTHON_CMD="tools\python\python.exe"
) else (
    echo ❌ Python installation failed
    echo 💡 Trying alternative installation method...
    
    REM Try installing with default options but to custom directory
    "%DOWNLOAD_FILE%" /quiet TargetDir="%cd%\tools\python" Include_pip=1 Include_tcltk=1
    
    if %errorlevel% equ 0 (
        echo ✅ Python installed with alternative method!
        set PYTHON_CMD="tools\python\python.exe"
    ) else (
        echo ❌ Both installation methods failed
        echo 🔧 Please run the installer manually:
        echo    Double-click: %DOWNLOAD_FILE%
        echo    Choose custom installation and select tkinter support
        pause
        exit /b 1
    )
)

REM Clean up installer
del "%DOWNLOAD_FILE%" >nul 2>&1

REM Verify Python works with GUI
echo 🔬 Verifying Python installation...
%PYTHON_CMD% -c "import tkinter; print('GUI support confirmed!')" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python with GUI support verified!
) else (
    echo ❌ GUI support verification failed
    echo 💡 Continuing anyway - might still work
)

:install_deps
echo.
echo 📦 Installing application dependencies...

REM Install required packages with better error handling
set PACKAGES=Pillow mss requests flask

REM Install packages one by one
for %%p in (%PACKAGES%) do (
    echo    📦 Installing %%p...
    %PYTHON_CMD% -m pip install %%p --quiet --disable-pip-version-check --no-warn-script-location
    if !errorlevel! equ 0 (
        echo       ✅ %%p installed
    ) else (
        echo       ⚠️ %%p installation had issues
    )
)

REM Install pywin32 separately (Windows-specific)
if "%ARCH%"=="amd64" (
    echo    📦 Installing pywin32 for Windows...
    %PYTHON_CMD% -m pip install pywin32 --quiet --disable-pip-version-check --no-warn-script-location
    if !errorlevel! equ 0 (
        echo       ✅ pywin32 installed
    ) else (
        echo       ⚠️ pywin32 installation had issues
    )
)

echo.
echo ✅ All dependencies installed!
echo.

:launch_app
echo 🚀 Launching Screen Capture Application...
echo ─────────────────────────────────────────────────────────────
echo.

REM Launch the application with proper environment
set PYTHONPATH=%cd%\src
%PYTHON_CMD% src\main_app.py

echo.
if %errorlevel% equ 0 (
    echo ✅ Application closed normally.
    echo ✨ Installation complete! Use LAUNCH_APP.bat for future launches.
    timeout /t 3 /nobreak >nul
) else (
    echo ⚠️ Application ended with issues.
    echo.
    echo 🛠️ Troubleshooting:
    echo    - GUI might need a moment to load
    echo    - Try running LAUNCH_APP.bat directly
    echo    - Check if antivirus is blocking the app
    echo.
    pause
)

REM Clean up downloads
if exist "tools\downloads" rmdir /s /q "tools\downloads" >nul 2>&1

endlocal
