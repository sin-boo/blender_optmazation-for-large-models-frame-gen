#!/usr/bin/env python3
"""
Screen Capture Application - Ultimate Self-Sufficient Launcher
This script provides everything needed to run the app with zero pre-installation.

Features:
- Automatic Python detection and installation
- Dependency management  
- Modern GUI launch
- Interactive troubleshooting
- Cross-platform support
- Beautiful CLI interface
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import json
import shutil
from pathlib import Path
import time

class UltimateAppLauncher:
    def __init__(self):
        self.app_dir = Path(__file__).parent.absolute()
        self.tools_dir = self.app_dir / "tools"
        self.python_dir = self.tools_dir / "python"
        self.src_dir = self.app_dir / "src"
        self.downloads_dir = self.tools_dir / "downloads"
        
        # Create necessary directories
        self.tools_dir.mkdir(exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        
        # System information
        self.system = platform.system()
        self.is_64bit = platform.architecture()[0] == "64bit"
        self.arch = platform.machine().lower()
        
        # Python configuration
        self.python_version = "3.11.9"
        self.python_urls = {
            "windows": {
                "64": f"https://www.python.org/ftp/python/{self.python_version}/python-{self.python_version}-embed-amd64.zip",
                "32": f"https://www.python.org/ftp/python/{self.python_version}/python-{self.python_version}-embed-win32.zip"
            }
        }
        
        # Essential packages for the application
        self.required_packages = [
            "Pillow>=10.0.0",
            "requests>=2.25.0", 
            "flask>=2.0.0",
            "mss>=6.0.0"
        ]
        
        # Add Windows-specific packages
        if self.system == "Windows":
            self.required_packages.append("pywin32>=300")
    
    def print_header(self):
        """Print modern styled application header"""
        os.system('cls' if self.system == 'Windows' else 'clear')
        
        width = 74
        print("┌" + "─" * (width - 2) + "┐")
        print("│" + " " * (width - 2) + "│")
        
        title = "🖥️  SCREEN CAPTURE APPLICATION"
        padding = (width - len(title) - 2) // 2
        print(f"│{' ' * padding}{title}{' ' * (width - len(title) - padding - 2)}│")
        
        subtitle = "✨ Ultimate Self-Sufficient Edition ✨"
        padding = (width - len(subtitle) - 2) // 2  
        print(f"│{' ' * padding}{subtitle}{' ' * (width - len(subtitle) - padding - 2)}│")
        
        print("│" + " " * (width - 2) + "│")
        print("├" + "─" * (width - 2) + "┤")
        
        # System info
        sys_info = f"💻 {self.system} ({self.arch}) | 🐍 Python {self.get_python_status()}"
        padding = width - len(sys_info) - 3
        print(f"│ {sys_info}{' ' * max(0, padding)}│")
        
        print("└" + "─" * (width - 2) + "┘")
        print()
    
    def get_python_status(self):
        """Get current Python status string"""
        try:
            return platform.python_version()
        except:
            return "Not Available"
    
    def print_status(self, icon, message, status=""):
        """Print a status line with consistent formatting"""
        if status:
            print(f"   {icon} {message:<45} {status}")
        else:
            print(f"   {icon} {message}")
    
    def check_python_installation(self):
        """Comprehensively check for Python installation"""
        self.print_status("🔍", "Checking Python installation...")
        
        # 1. Check current Python process
        try:
            if sys.version_info >= (3, 7):
                version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                self.print_status("✅", f"Current Python {version} is suitable", "READY")
                return sys.executable
        except:
            pass
        
        # 2. Check system Python commands
        python_commands = ["py", "python", "python3"]
        for cmd in python_commands:
            try:
                result = subprocess.run(
                    [cmd, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0 and "Python 3." in result.stdout:
                    version_str = result.stdout.strip().split()[1]
                    major, minor = map(int, version_str.split('.')[:2])
                    if major >= 3 and minor >= 7:
                        self.print_status("✅", f"System Python {version_str} found ({cmd})", "READY")
                        return cmd
            except:
                continue
        
        # 3. Check portable Python
        portable_python = self.python_dir / "python.exe"
        if portable_python.exists():
            try:
                result = subprocess.run(
                    [str(portable_python), "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split()[1]
                    self.print_status("✅", f"Portable Python {version} found", "READY")
                    return str(portable_python)
            except:
                pass
        
        self.print_status("❌", "No suitable Python installation found", "NEEDS INSTALL")
        return None
    
    def download_with_progress(self, url, destination, description):
        """Download with beautiful progress bar"""
        self.print_status("📥", f"Downloading {description}...")
        print(f"      🔗 {url}")
        
        class ProgressHook:
            def __init__(self):
                self.last_percent = -1
            
            def __call__(self, block_num, block_size, total_size):
                if total_size <= 0:
                    return
                
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                
                if percent != self.last_percent:
                    bar_length = 40
                    filled_length = (percent * bar_length) // 100
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    
                    size_mb = total_size / (1024 * 1024)
                    downloaded_mb = downloaded / (1024 * 1024)
                    
                    print(f"\r      Progress: |{bar}| {percent}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)", 
                          end="", flush=True)
                    self.last_percent = percent
        
        try:
            urllib.request.urlretrieve(url, destination, ProgressHook())
            print()  # New line after progress
            self.print_status("✅", f"{description} downloaded successfully")
            return True
        except Exception as e:
            print()  # New line after progress
            self.print_status("❌", f"Download failed: {e}")
            return False
    
    def install_portable_python_windows(self):
        """Install portable Python on Windows"""
        if self.system != "Windows":
            self.print_status("❌", "Portable Python only supported on Windows")
            return None
        
        print()
        self.print_status("🐍", "Installing portable Python for Windows...")
        
        # Determine architecture and URL
        arch = "64" if self.is_64bit else "32"
        python_url = self.python_urls["windows"][arch]
        
        # Download Python
        zip_file = self.downloads_dir / "python_portable.zip"
        if not self.download_with_progress(python_url, zip_file, f"Python {self.python_version}"):
            return None
        
        try:
            # Extract Python
            self.print_status("📦", "Extracting Python...")
            if self.python_dir.exists():
                shutil.rmtree(self.python_dir)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.python_dir)
            
            # Configure for pip support
            self.print_status("🔧", "Configuring Python for package management...")
            pth_files = list(self.python_dir.glob("python*._pth"))
            if pth_files:
                pth_file = pth_files[0]
                with open(pth_file, 'a') as f:
                    f.write("\n# Enable site-packages for pip\n")
                    f.write("import site\n")
            
            # Download and install pip
            get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
            get_pip_path = self.python_dir / "get-pip.py"
            
            if self.download_with_progress(get_pip_url, get_pip_path, "pip installer"):
                self.print_status("🔧", "Installing pip...")
                python_exe = self.python_dir / "python.exe"
                subprocess.run(
                    [str(python_exe), str(get_pip_path), "--no-warn-script-location"],
                    check=True, capture_output=True, text=True
                )
                get_pip_path.unlink()  # Clean up
            
            # Clean up download
            zip_file.unlink()
            
            self.print_status("✅", "Portable Python installation completed!", "SUCCESS")
            return str(python_exe)
            
        except Exception as e:
            self.print_status("❌", f"Python installation failed: {e}")
            return None
    
    def install_dependencies(self, python_cmd):
        """Install all required dependencies"""
        print()
        self.print_status("📦", "Installing application dependencies...")
        
        success_count = 0
        total_packages = len(self.required_packages)
        
        for package in self.required_packages:
            package_name = package.split(">=")[0]
            self.print_status("📦", f"Installing {package_name}...", "")
            
            try:
                result = subprocess.run(
                    [python_cmd, "-m", "pip", "install", package, "--quiet", "--disable-pip-version-check"],
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                
                if result.returncode == 0:
                    self.print_status("✅", f"{package_name} installed successfully")
                    success_count += 1
                else:
                    self.print_status("⚠️", f"{package_name} installation had issues")
                    
            except subprocess.TimeoutExpired:
                self.print_status("⏰", f"{package_name} installation timed out")
            except Exception as e:
                self.print_status("❌", f"{package_name} error: {str(e)[:50]}")
        
        if success_count == total_packages:
            self.print_status("🎉", f"All {total_packages} dependencies installed!", "SUCCESS")
        else:
            self.print_status("⚠️", f"{success_count}/{total_packages} dependencies installed", "PARTIAL")
        
        return success_count > total_packages // 2  # At least half must succeed
    
    def verify_installation(self, python_cmd):
        """Verify the complete installation"""
        self.print_status("🔬", "Verifying installation...")
        
        test_script = '''
import sys
packages_status = []

test_packages = {
    "tkinter": "tkinter",
    "Pillow": "PIL.Image", 
    "requests": "requests",
    "flask": "flask",
    "mss": "mss"
}

if sys.platform == "win32":
    test_packages["pywin32"] = "win32gui"

for name, import_name in test_packages.items():
    try:
        __import__(import_name)
        packages_status.append(f"✅ {name}")
    except ImportError:
        packages_status.append(f"❌ {name}")

for status in packages_status:
    print(status)
'''
        
        try:
            result = subprocess.run(
                [python_cmd, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            verification_results = result.stdout.strip().split('\n')
            failed_count = sum(1 for line in verification_results if line.startswith("❌"))
            
            print("      Package verification:")
            for line in verification_results:
                if line.strip():
                    print(f"        {line}")
            
            if failed_count == 0:
                self.print_status("✅", "All packages verified successfully!", "READY")
                return True
            else:
                self.print_status("⚠️", f"{failed_count} packages have issues", "CHECK NEEDED")
                return False
                
        except Exception as e:
            self.print_status("❌", f"Verification failed: {e}")
            return False
    
    def ensure_python_ready(self):
        """Ensure Python is ready for the application"""
        python_cmd = self.check_python_installation()
        
        if python_cmd:
            return python_cmd
        
        # Need to install Python
        if self.system == "Windows":
            print()
            self.print_status("🔧", "Installing Python automatically...")
            python_cmd = self.install_portable_python_windows()
            
            if python_cmd:
                return python_cmd
            else:
                print()
                self.print_status("❌", "Automatic Python installation failed")
                self.print_status("💡", "Manual installation required:")
                print("      1. Visit: https://www.python.org/downloads/")
                print("      2. Download Python 3.7 or newer")
                print("      3. Check 'Add Python to PATH' during installation")
                print("      4. Run this script again")
                return None
        else:
            print()
            self.print_status("❌", "Python installation required")
            self.print_status("💡", "Install Python with your package manager:")
            if self.system == "Darwin":  # macOS
                print("      brew install python3")
            else:  # Linux
                print("      sudo apt install python3 python3-pip")
                print("      # or")
                print("      sudo yum install python3 python3-pip")
            return None
    
    def launch_application(self, python_cmd):
        """Launch the main application"""
        app_file = self.src_dir / "main_app.py"
        
        if not app_file.exists():
            self.print_status("❌", f"Application file not found: {app_file}")
            return False
        
        print()
        print("🚀 " + "─" * 60)
        print("   LAUNCHING SCREEN CAPTURE APPLICATION")
        print("─" * 68)
        print()
        
        try:
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.src_dir)
            
            # Launch with proper working directory
            os.chdir(self.app_dir)
            subprocess.run([python_cmd, str(app_file)], env=env, check=True)
            
            print()
            self.print_status("✅", "Application closed normally")
            return True
            
        except subprocess.CalledProcessError as e:
            print()
            self.print_status("⚠️", f"Application exited with code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print()
            self.print_status("👋", "Application closed by user")
            return True
        except Exception as e:
            print()
            self.print_status("❌", f"Launch error: {e}")
            return False
    
    def quick_setup_and_launch(self):
        """Streamlined setup and launch for most users"""
        self.print_header()
        
        # Step 1: Ensure Python
        print("🔧 STEP 1: Python Environment")
        print("─" * 40)
        python_cmd = self.ensure_python_ready()
        if not python_cmd:
            input("\nPress Enter to exit...")
            return False
        
        # Step 2: Install dependencies  
        print("\n📦 STEP 2: Dependencies")
        print("─" * 40)
        if not self.install_dependencies(python_cmd):
            self.print_status("⚠️", "Some dependencies failed, but continuing...")
        
        # Step 3: Verify installation
        print("\n🔬 STEP 3: Verification")
        print("─" * 40)
        verification_ok = self.verify_installation(python_cmd)
        
        if not verification_ok:
            self.print_status("⚠️", "Some issues detected, but trying to launch anyway...")
        
        # Step 4: Launch
        print("\n🚀 STEP 4: Launch")
        print("─" * 40)
        return self.launch_application(python_cmd)
    
    def show_advanced_menu(self):
        """Show advanced troubleshooting menu"""
        while True:
            self.print_header()
            
            print("╔" + "═" * 50 + "╗")
            print("║" + "  🛠️  ADVANCED OPTIONS & TROUBLESHOOTING  ".center(50) + "║")
            print("╠" + "═" * 50 + "╣")
            print("║  1️⃣  Quick Setup & Launch (Recommended)      ║")
            print("║  2️⃣  Force Reinstall Python                 ║") 
            print("║  3️⃣  Reinstall Dependencies Only            ║")
            print("║  4️⃣  Launch App (Skip Setup)                ║")
            print("║  5️⃣  System Diagnostics                     ║")
            print("║  6️⃣  Clean & Reset Environment              ║")
            print("║  0️⃣  Exit                                    ║")
            print("╚" + "═" * 50 + "╝")
            
            try:
                choice = input("\n🎯 Select option (0-6): ").strip()
                
                if choice == "1":
                    return self.quick_setup_and_launch()
                
                elif choice == "2":
                    self.force_reinstall_python()
                    input("\nPress Enter to continue...")
                
                elif choice == "3":
                    python_cmd = self.check_python_installation()
                    if python_cmd:
                        self.install_dependencies(python_cmd)
                    else:
                        self.print_status("❌", "Python not available")
                    input("\nPress Enter to continue...")
                
                elif choice == "4":
                    python_cmd = self.check_python_installation()
                    if python_cmd:
                        return self.launch_application(python_cmd)
                    else:
                        self.print_status("❌", "Python not available - setup required")
                        input("\nPress Enter to continue...")
                
                elif choice == "5":
                    self.show_system_diagnostics()
                    input("\nPress Enter to continue...")
                
                elif choice == "6":
                    self.clean_environment()
                    input("\nPress Enter to continue...")
                
                elif choice == "0":
                    self.print_status("👋", "Goodbye!")
                    return True
                
                else:
                    self.print_status("❌", "Invalid choice. Please enter 0-6.")
                    time.sleep(1)
                    
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                return True
    
    def force_reinstall_python(self):
        """Force reinstallation of portable Python"""
        print()
        self.print_status("🧹", "Cleaning existing Python installation...")
        
        if self.python_dir.exists():
            shutil.rmtree(self.python_dir)
            self.print_status("✅", "Previous installation cleaned")
        
        python_cmd = self.install_portable_python_windows()
        if python_cmd:
            self.install_dependencies(python_cmd)
            self.verify_installation(python_cmd)
    
    def clean_environment(self):
        """Clean the entire environment"""
        print()
        self.print_status("🧹", "Cleaning environment...")
        
        items_to_clean = [
            (self.python_dir, "Portable Python"),
            (self.downloads_dir, "Downloads"),
        ]
        
        for path, name in items_to_clean:
            if path.exists():
                shutil.rmtree(path)
                self.print_status("✅", f"{name} cleaned")
        
        # Recreate necessary directories
        self.tools_dir.mkdir(exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        
        self.print_status("✅", "Environment cleaned successfully")
    
    def show_system_diagnostics(self):
        """Show comprehensive system diagnostics"""
        print()
        print("🔍 " + "─" * 60)
        print("   SYSTEM DIAGNOSTICS")
        print("─" * 68)
        
        # System information
        print(f"\n💻 System Information:")
        print(f"   OS: {platform.platform()}")
        print(f"   Architecture: {platform.architecture()[0]} ({self.arch})")
        print(f"   Processor: {platform.processor()}")
        
        # Python information  
        print(f"\n🐍 Python Information:")
        print(f"   Current Python: {self.get_python_status()}")
        print(f"   Executable: {sys.executable}")
        
        # Directory information
        print(f"\n📁 Application Structure:")
        print(f"   App Directory: {self.app_dir}")
        print(f"   Tools Directory: {self.tools_dir}")
        print(f"   Python Directory: {self.python_dir}")
        print(f"   Source Directory: {self.src_dir}")
        
        # File status
        print(f"\n📋 Key Files Status:")
        key_files = [
            ("main_app.py", self.src_dir / "main_app.py"),
            ("modern_gui.py", self.src_dir / "modern_gui.py"),
            ("capture_server.py", self.src_dir / "capture_server.py"),
            ("requirements.txt", self.app_dir / "requirements.txt"),
            ("bootstrap.py", self.app_dir / "bootstrap.py"),
            ("Portable Python", self.python_dir / "python.exe")
        ]
        
        for name, path in key_files:
            status = "✅ EXISTS" if path.exists() else "❌ MISSING"
            print(f"   {status:<12} {name}")
    
    def run(self):
        """Main run method"""
        # Check if this looks like a first run
        is_first_run = not (self.tools_dir / "setup_completed").exists()
        
        if is_first_run:
            # First run - do automatic setup
            self.print_header()
            self.print_status("🎯", "First run detected - setting up environment...")
            result = self.quick_setup_and_launch()
            
            # Mark setup as completed
            if result:
                (self.tools_dir / "setup_completed").touch()
            
            return result
        else:
            # Subsequent runs - try quick launch first
            python_cmd = self.check_python_installation()
            app_file = self.src_dir / "main_app.py"
            
            if python_cmd and app_file.exists():
                self.print_header()
                self.print_status("⚡", "Environment ready!")
                
                response = input("\n🚀 Launch application? (Y/n/advanced): ").strip().lower()
                
                if response in ('', 'y', 'yes'):
                    return self.launch_application(python_cmd)
                elif response in ('a', 'advanced'):
                    return self.show_advanced_menu()
                else:
                    self.print_status("👋", "Cancelled by user")
                    return True
            else:
                # Something's wrong - show advanced menu
                return self.show_advanced_menu()

def main():
    """Main entry point"""
    try:
        launcher = UltimateAppLauncher()
        success = launcher.run()
        
        if not success:
            input("\nPress Enter to exit...")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
