#!/usr/bin/env python3
"""
Screen Capture Application Bootstrap
Self-sufficient launcher that can install Python and all dependencies
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import tempfile
import shutil
from pathlib import Path
import json

class BootstrapManager:
    def __init__(self):
        self.app_dir = Path(__file__).parent.absolute()
        self.tools_dir = self.app_dir / "tools"
        self.python_dir = self.tools_dir / "python"
        self.src_dir = self.app_dir / "src"
        
        self.tools_dir.mkdir(exist_ok=True)
        
        self.system_info = {
            "os": platform.system(),
            "arch": platform.machine().lower(),
            "is_64bit": platform.architecture()[0] == "64bit"
        }
        
        # Python download URLs (Windows portable)
        self.python_urls = {
            "windows": {
                "64": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
                "32": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-win32.zip"
            }
        }
    
    def print_styled_header(self):
        """Print a modern styled header"""
        width = 70
        print("┌" + "─" * (width - 2) + "┐")
        print("│" + " " * (width - 2) + "│")
        title = "🖥️  SCREEN CAPTURE APPLICATION"
        padding = (width - len(title) - 2) // 2
        print(f"│{' ' * padding}{title}{' ' * (width - len(title) - padding - 2)}│")
        
        subtitle = "Self-Sufficient Window & Screen Recorder"
        padding = (width - len(subtitle) - 2) // 2
        print(f"│{' ' * padding}{subtitle}{' ' * (width - len(subtitle) - padding - 2)}│")
        
        print("│" + " " * (width - 2) + "│")
        print("├" + "─" * (width - 2) + "┤")
        
        # System info
        os_info = f"💻 {self.system_info['os']} ({self.system_info['arch']})"
        print(f"│ {os_info:<{width-3}}│")
        
        try:
            python_info = f"🐍 Python {platform.python_version()}"
            print(f"│ {python_info:<{width-3}}│")
        except:
            python_info = "🐍 Python: Not Available"
            print(f"│ {python_info:<{width-3}}│")
        
        print("└" + "─" * (width - 2) + "┘")
        print()
    
    def check_python_available(self):
        """Check if Python is available and working"""
        try:
            # Check if current Python process is working
            if sys.version_info >= (3, 7):
                return True, sys.executable
            else:
                return False, "Python version too old (need 3.7+)"
        except:
            pass
        
        # Check system Python
        for cmd in ["python", "python3", "py"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0 and "Python 3." in result.stdout:
                    version_str = result.stdout.strip().split()[1]
                    major, minor = map(int, version_str.split('.')[:2])
                    if major >= 3 and minor >= 7:
                        return True, cmd
            except:
                continue
        
        # Check portable Python
        portable_python = self.python_dir / "python.exe"
        if portable_python.exists():
            try:
                result = subprocess.run(
                    [str(portable_python), "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    return True, str(portable_python)
            except:
                pass
        
        return False, None
    
    def download_file(self, url, destination, description="file"):
        """Download a file with progress indication"""
        print(f"📥 Downloading {description}...")
        print(f"   URL: {url}")
        
        try:
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = (block_num * block_size * 100) // total_size
                    percent = min(percent, 100)
                    bar_length = 30
                    filled_length = (percent * bar_length) // 100
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    print(f"\r   Progress: |{bar}| {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, destination, progress_hook)
            print()  # New line after progress bar
            return True
        except Exception as e:
            print(f"\n❌ Failed to download {description}: {e}")
            return False
    
    def install_portable_python(self):
        """Install portable Python for Windows"""
        if self.system_info["os"] != "Windows":
            print("❌ Portable Python installation only supported on Windows")
            return False
        
        print("🐍 Installing portable Python...")
        
        # Determine architecture
        arch = "64" if self.system_info["is_64bit"] else "32"
        python_url = self.python_urls["windows"][arch]
        
        # Download Python
        temp_zip = self.tools_dir / "python_portable.zip"
        if not self.download_file(python_url, temp_zip, "Python portable"):
            return False
        
        try:
            # Extract Python
            print("📦 Extracting Python...")
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(self.python_dir)
            
            # Download get-pip.py for package management
            get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
            get_pip_path = self.python_dir / "get-pip.py"
            
            if self.download_file(get_pip_url, get_pip_path, "pip installer"):
                # Install pip
                python_exe = self.python_dir / "python.exe"
                subprocess.run([str(python_exe), str(get_pip_path)], 
                             check=True, capture_output=True)
            
            # Cleanup
            temp_zip.unlink(missing_ok=True)
            get_pip_path.unlink(missing_ok=True)
            
            print("✅ Portable Python installed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install portable Python: {e}")
            return False
    
    def install_python_dependencies(self, python_cmd):
        """Install required Python packages"""
        requirements_file = self.app_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print("⚠️  No requirements.txt found, installing basic packages...")
            packages = ["Pillow", "mss", "requests", "flask"]
            if self.system_info["os"] == "Windows":
                packages.append("pywin32")
        else:
            print("📦 Installing packages from requirements.txt...")
            packages = None
        
        try:
            if packages:
                # Install individual packages
                for package in packages:
                    print(f"   Installing {package}...")
                    result = subprocess.run(
                        [python_cmd, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode != 0:
                        print(f"   ⚠️  Warning: {package} installation failed")
            else:
                # Install from requirements.txt
                result = subprocess.run(
                    [python_cmd, "-m", "pip", "install", "-r", str(requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode != 0:
                    print("⚠️  Some packages may have failed to install")
            
            print("✅ Package installation completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
            return False
    
    def ensure_python_environment(self):
        """Ensure Python is available and ready"""
        print("🔍 Checking Python environment...")
        
        python_available, python_cmd = self.check_python_available()
        
        if python_available:
            print(f"✅ Python found: {python_cmd}")
            return python_cmd
        
        print("❌ Suitable Python not found!")
        
        if self.system_info["os"] == "Windows":
            print("🔧 Installing portable Python...")
            if self.install_portable_python():
                python_cmd = str(self.python_dir / "python.exe")
                print("✅ Portable Python installed!")
                return python_cmd
            else:
                print("❌ Failed to install portable Python")
                return None
        else:
            print("❌ Please install Python 3.7+ manually")
            print("Visit: https://www.python.org/downloads/")
            return None
    
    def setup_environment(self):
        """Set up the complete environment"""
        self.print_styled_header()
        
        # Ensure Python is available
        python_cmd = self.ensure_python_environment()
        if not python_cmd:
            input("Press Enter to exit...")
            return False
        
        # Install dependencies
        print("\n🔧 Setting up dependencies...")
        if not self.install_python_dependencies(python_cmd):
            print("⚠️  Some dependencies may be missing, but continuing...")
        
        return python_cmd
    
    def launch_application(self, python_cmd):
        """Launch the main application"""
        app_file = self.src_dir / "main_app.py"
        
        if not app_file.exists():
            # Fallback to current directory
            app_file = self.app_dir / "screen_capture_app.py"
        
        if not app_file.exists():
            print("❌ Main application file not found!")
            return False
        
        print(f"🚀 Launching Screen Capture Application...")
        print("-" * 50)
        
        try:
            subprocess.run([python_cmd, str(app_file)], check=True)
            return True
        except KeyboardInterrupt:
            print("\n👋 Application closed by user")
            return True
        except Exception as e:
            print(f"❌ Error launching application: {e}")
            return False
    
    def run(self):
        """Main bootstrap routine"""
        python_cmd = self.setup_environment()
        
        if python_cmd:
            return self.launch_application(python_cmd)
        
        return False

if __name__ == "__main__":
    bootstrap = BootstrapManager()
    success = bootstrap.run()
    if not success:
        input("Press Enter to exit...")
        sys.exit(1)
