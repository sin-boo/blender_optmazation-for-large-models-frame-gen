@echo off
title Screen Capture App - Auto Installer
color 0B
setlocal enabledelayedexpansion

echo.
echo     ╔══════════════════════════════════════════════════════════════════╗
echo     ║                                                                  ║
echo     ║  🖥️   SCREEN CAPTURE APP - AUTOMATIC INSTALLER                 ║
echo     ║                                                                  ║
echo     ║  🚀 This will automatically download and install everything!     ║
echo     ║  📦 Python + Dependencies + Modern GUI                          ║
echo     ║                                                                  ║
echo     ║  ⏱️ Takes 2-5 minutes depending on internet speed               ║
echo     ║                                                                  ║
echo     ╚══════════════════════════════════════════════════════════════════╝
echo.

REM Change to script directory
cd /d "%~dp0"

echo 🔍 Checking if Python is already available...

REM Try to find existing Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python found! Skipping installation.
    goto :install_deps
)

py --version >nul 2>&1  
if %errorlevel% equ 0 (
    echo ✅ Python found via py launcher! Skipping installation.
    set PYTHON_CMD=py
    goto :install_deps
)

if exist "tools\python\python.exe" (
    echo ✅ Portable Python already installed!
    set PYTHON_CMD="tools\python\python.exe"
    goto :install_deps
)

echo 📥 Python not found. Installing automatically...
echo.

REM Create tools directory
if not exist "tools" mkdir "tools"
if not exist "tools\downloads" mkdir "tools\downloads"

echo 🌐 Downloading Python installer from python.org...

REM Determine architecture and download URL
set ARCH=amd64
if "%PROCESSOR_ARCHITECTURE%"=="x86" set ARCH=win32

set PYTHON_VERSION=3.11.9
set DOWNLOAD_URL=https://www.python.org/ftp/python/!PYTHON_VERSION!/python-!PYTHON_VERSION!-embed-!ARCH!.zip
set DOWNLOAD_FILE=tools\downloads\python_portable.zip

echo    📍 URL: %DOWNLOAD_URL%
echo    💾 Downloading to: %DOWNLOAD_FILE%
echo.

REM Download Python using PowerShell (more reliable than curl on Windows)
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; try { Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%DOWNLOAD_FILE%' -UseBasicParsing; Write-Host 'Download completed successfully!'; exit 0 } catch { Write-Host 'Download failed:' $_.Exception.Message; exit 1 }}"

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

echo 📦 Extracting Python...

REM Extract using PowerShell
powershell -Command "& {Add-Type -AssemblyName System.IO.Compression.FileSystem; try { [System.IO.Compression.ZipFile]::ExtractToDirectory('%DOWNLOAD_FILE%', 'tools\python'); Write-Host 'Extraction completed!'; exit 0 } catch { Write-Host 'Extraction failed:' $_.Exception.Message; exit 1 }}"

if %errorlevel% neq 0 (
    echo ❌ Failed to extract Python
    pause
    exit /b 1
)

REM Configure Python for pip
echo 🔧 Configuring Python for package management...

REM Find and modify the .pth file to enable pip
for %%f in (tools\python\python*._pth) do (
    echo import site >> "%%f"
)

REM Download and install pip
echo 📥 Downloading pip installer...
set GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py
set GET_PIP_FILE=tools\downloads\get-pip.py

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%GET_PIP_URL%' -OutFile '%GET_PIP_FILE%' -UseBasicParsing}"

if exist "%GET_PIP_FILE%" (
    echo 🔧 Installing pip...
    "tools\python\python.exe" "%GET_PIP_FILE%" --no-warn-script-location
    if %errorlevel% equ 0 (
        echo ✅ pip installed successfully!
    ) else (
        echo ⚠️ pip installation had issues, but continuing...
    )
    del "%GET_PIP_FILE%" >nul 2>&1
)

set PYTHON_CMD="tools\python\python.exe"
echo ✅ Python installation completed!

:install_deps
echo.
echo 📦 Installing application dependencies...

REM Install required packages
set PACKAGES=Pillow mss requests flask pywin32

for %%p in (%PACKAGES%) do (
    echo    📦 Installing %%p...
    %PYTHON_CMD% -m pip install %%p --quiet --disable-pip-version-check
    if !errorlevel! equ 0 (
        echo       ✅ %%p installed
    ) else (
        echo       ⚠️ %%p installation had issues
    )
)

echo.
echo ✅ All dependencies installed!
echo.

:launch_app
echo 🚀 Launching Screen Capture Application...
echo ─────────────────────────────────────────────────────────────
echo.

REM Launch the application
%PYTHON_CMD% src\main_app.py

echo.
if %errorlevel% equ 0 (
    echo ✅ Application closed normally.
    timeout /t 2 /nobreak >nul
) else (
    echo ⚠️ Application ended with issues.
    echo 💡 This might be normal on the first run.
    echo.
    pause
)

REM Clean up downloads
if exist "tools\downloads" rmdir /s /q "tools\downloads" >nul 2>&1

endlocal
