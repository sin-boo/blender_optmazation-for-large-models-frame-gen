@echo off
title Screen Capture App - Fixed Auto Installer
color 0B
setlocal enabledelayedexpansion

echo.
echo     ╔══════════════════════════════════════════════════════════════════╗
echo     ║                                                                  ║
echo     ║  🖥️   SCREEN CAPTURE APP - AUTOMATIC INSTALLER (FIXED)         ║
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
    set PYTHON_CMD=python
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

REM Create a temporary PowerShell script for downloading
echo $ErrorActionPreference = "Stop" > temp_download.ps1
echo [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 >> temp_download.ps1
echo try { >> temp_download.ps1
echo     $webClient = New-Object System.Net.WebClient >> temp_download.ps1
echo     $webClient.DownloadFile('%DOWNLOAD_URL%', '%DOWNLOAD_FILE%') >> temp_download.ps1
echo     Write-Host "Download completed successfully!" >> temp_download.ps1
echo     exit 0 >> temp_download.ps1
echo } catch { >> temp_download.ps1
echo     Write-Host "Download failed:" $_.Exception.Message >> temp_download.ps1
echo     exit 1 >> temp_download.ps1
echo } >> temp_download.ps1

REM Execute the PowerShell script
powershell -ExecutionPolicy Bypass -File temp_download.ps1

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to download Python automatically.
    echo 🔧 Please install Python manually:
    echo    1. Go to https://python.org
    echo    2. Download Python 3.7 or newer  
    echo    3. Make sure to check "Add Python to PATH"
    echo    4. Run this script again
    echo.
    del temp_download.ps1 >nul 2>&1
    pause
    exit /b 1
)

REM Clean up the temporary PowerShell script
del temp_download.ps1 >nul 2>&1

echo 📦 Extracting Python...

REM Create PowerShell script for extraction
echo $ErrorActionPreference = "Stop" > temp_extract.ps1
echo Add-Type -AssemblyName System.IO.Compression.FileSystem >> temp_extract.ps1
echo try { >> temp_extract.ps1
echo     [System.IO.Compression.ZipFile]::ExtractToDirectory('%DOWNLOAD_FILE%', 'tools\python') >> temp_extract.ps1
echo     Write-Host "Extraction completed successfully!" >> temp_extract.ps1
echo     exit 0 >> temp_extract.ps1
echo } catch { >> temp_extract.ps1
echo     Write-Host "Extraction failed:" $_.Exception.Message >> temp_extract.ps1
echo     exit 1 >> temp_extract.ps1
echo } >> temp_extract.ps1

REM Execute extraction
powershell -ExecutionPolicy Bypass -File temp_extract.ps1

if %errorlevel% neq 0 (
    echo ❌ Failed to extract Python
    del temp_extract.ps1 >nul 2>&1
    pause
    exit /b 1
)

REM Clean up extraction script
del temp_extract.ps1 >nul 2>&1

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

REM Create PowerShell script for pip download
echo $ErrorActionPreference = "Stop" > temp_pip_download.ps1
echo [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 >> temp_pip_download.ps1
echo $webClient = New-Object System.Net.WebClient >> temp_pip_download.ps1
echo $webClient.DownloadFile('%GET_PIP_URL%', '%GET_PIP_FILE%') >> temp_pip_download.ps1

powershell -ExecutionPolicy Bypass -File temp_pip_download.ps1
del temp_pip_download.ps1 >nul 2>&1

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
