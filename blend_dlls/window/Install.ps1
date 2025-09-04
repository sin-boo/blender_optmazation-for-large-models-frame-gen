# Screen Capture Application - PowerShell Installer
# More reliable than batch for downloading and installing

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan  
Write-Host "║  🖥️   SCREEN CAPTURE APP - POWERSHELL INSTALLER                ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "║  🔧 More reliable installation using PowerShell                 ║" -ForegroundColor Cyan
Write-Host "║  📦 Downloads Python + All Dependencies                         ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Set TLS version for downloads
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Get script location
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptPath

# Create directories
New-Item -ItemType Directory -Path "tools", "tools\downloads" -Force | Out-Null

function Test-PythonInstallation {
    Write-Host "🔍 Checking for Python..." -ForegroundColor Yellow
    
    # Test current Python
    try {
        $pythonVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $pythonVersion -match "Python 3\.([7-9]|\d{2})") {
            Write-Host "✅ Found system Python: $pythonVersion" -ForegroundColor Green
            return "python"
        }
    } catch {}
    
    # Test py launcher
    try {
        $pythonVersion = & py --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $pythonVersion -match "Python 3\.([7-9]|\d{2})") {
            Write-Host "✅ Found Python via py launcher: $pythonVersion" -ForegroundColor Green
            return "py"
        }
    } catch {}
    
    # Test portable Python
    $portablePython = "tools\python\python.exe"
    if (Test-Path $portablePython) {
        try {
            $pythonVersion = & $portablePython --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Found portable Python: $pythonVersion" -ForegroundColor Green
                return $portablePython
            }
        } catch {}
    }
    
    Write-Host "❌ No suitable Python found" -ForegroundColor Red
    return $null
}

function Install-PortablePython {
    Write-Host "🐍 Installing portable Python..." -ForegroundColor Yellow
    
    # Determine architecture
    $arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "win32" }
    $pythonVersion = "3.11.9"
    $downloadUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-$arch.zip"
    $downloadFile = "tools\downloads\python_portable.zip"
    
    Write-Host "📥 Downloading Python from: $downloadUrl" -ForegroundColor Cyan
    
    try {
        # Download with progress
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($downloadUrl, $downloadFile)
        Write-Host "✅ Download completed!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    # Extract Python
    Write-Host "📦 Extracting Python..." -ForegroundColor Yellow
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($downloadFile, "tools\python")
        Write-Host "✅ Extraction completed!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Extraction failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    # Configure for pip
    Write-Host "🔧 Configuring Python..." -ForegroundColor Yellow
    $pthFiles = Get-ChildItem "tools\python\python*._pth"
    foreach ($pthFile in $pthFiles) {
        Add-Content $pthFile "`nimport site"
    }
    
    # Download and install pip
    Write-Host "📥 Setting up pip..." -ForegroundColor Yellow
    $getPipUrl = "https://bootstrap.pypa.io/get-pip.py" 
    $getPipFile = "tools\downloads\get-pip.py"
    
    try {
        Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipFile -UseBasicParsing
        & "tools\python\python.exe" $getPipFile --no-warn-script-location
        Remove-Item $getPipFile -Force
        Write-Host "✅ pip installed!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ pip setup had issues, but continuing..." -ForegroundColor Yellow
    }
    
    return "tools\python\python.exe"
}

function Install-Dependencies {
    param($PythonCommand)
    
    Write-Host "📦 Installing application dependencies..." -ForegroundColor Yellow
    
    $packages = @("Pillow>=10.0.0", "mss>=6.0.0", "requests>=2.25.0", "flask>=2.0.0", "pywin32>=300")
    
    foreach ($package in $packages) {
        $packageName = $package.Split(">=")[0]
        Write-Host "   📦 Installing $packageName..." -ForegroundColor Cyan
        
        try {
            & $PythonCommand -m pip install $package --quiet --disable-pip-version-check
            if ($LASTEXITCODE -eq 0) {
                Write-Host "      ✅ $packageName installed" -ForegroundColor Green
            } else {
                Write-Host "      ⚠️ $packageName had issues" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "      ❌ $packageName failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

function Start-Application {
    param($PythonCommand)
    
    Write-Host ""
    Write-Host "🚀 Launching Screen Capture Application..." -ForegroundColor Green
    Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host ""
    
    try {
        & $PythonCommand "src\main_app.py"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Application closed normally." -ForegroundColor Green
        } else {
            Write-Host "⚠️ Application ended with exit code $LASTEXITCODE" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to launch application: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Main execution
try {
    $pythonCmd = Test-PythonInstallation
    
    if (-not $pythonCmd) {
        $pythonCmd = Install-PortablePython
        if (-not $pythonCmd) {
            Write-Host "❌ Failed to install Python" -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    
    Install-Dependencies $pythonCmd
    
    Write-Host ""
    Write-Host "🎉 Installation completed successfully!" -ForegroundColor Green
    Write-Host "✨ Ready to launch the application!" -ForegroundColor Green
    Write-Host ""
    
    $response = Read-Host "🚀 Launch Screen Capture App now? (Y/n)"
    if ($response -eq "" -or $response -eq "y" -or $response -eq "Y") {
        Start-Application $pythonCmd
    }
    
    Write-Host ""
    Write-Host "💡 Next time you can just double-click 'START_HERE.bat'" -ForegroundColor Cyan
    Write-Host ""
    
} catch {
    Write-Host "❌ Unexpected error: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
} finally {
    # Cleanup downloads
    if (Test-Path "tools\downloads") {
        Remove-Item "tools\downloads" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Read-Host "Press Enter to exit"
