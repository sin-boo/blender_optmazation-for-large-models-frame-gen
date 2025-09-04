#!/usr/bin/env python3
"""
Enhanced Screen Capture Application Launcher
Automatically installs dependencies, checks system requirements, and launches the application
"""

import os
import sys
import subprocess
import importlib
import platform
from pathlib import Path

class AppLauncher:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.requirements_file = self.script_dir / "requirements.txt"
        self.missing_packages = []
        self.system_info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }
    
    def print_header(self):
        """Print application header"""
        print("=" * 60)
        print("         Screen Capture Application Launcher")
        print("=" * 60)
        print(f"Operating System: {self.system_info['os']}")
        print(f"Python Version: {self.system_info['python_version']}")
        print(f"Architecture: {self.system_info['architecture']}")
        print("-" * 60)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 7):
            print("❌ ERROR: Python 3.7 or higher is required!")
            print(f"   Current version: {platform.python_version()}")
            print("   Please upgrade Python and try again.")
            return False
        
        print(f"✅ Python version {platform.python_version()} is compatible")
        return True
    
    def check_package(self, package_name, import_name=None):
        """Check if a package is installed"""
        if import_name is None:
            import_name = package_name
        
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    def parse_requirements(self):
        """Parse requirements.txt file"""
        requirements = []
        
        if not self.requirements_file.exists():
            print(f"❌ Requirements file not found: {self.requirements_file}")
            return requirements
        
        try:
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove version specifiers for checking
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                        requirements.append((package, line))  # (package_name, full_requirement)
        except Exception as e:
            print(f"❌ Error reading requirements file: {e}")
        
        return requirements
    
    def check_requirements(self):
        """Check all requirements"""
        print("🔍 Checking requirements...")
        
        requirements = self.parse_requirements()
        if not requirements:
            print("⚠️  No requirements file found, proceeding with basic checks...")
            # Basic required packages for the application to work
            basic_requirements = [
                ("tkinter", "tkinter"),
                ("PIL", "Pillow"),
                ("requests", "requests")
            ]
            
            for import_name, package_name in basic_requirements:
                if not self.check_package(import_name):
                    self.missing_packages.append(package_name)
            return
        
        # Special package name mappings
        package_mappings = {
            "PIL": "Pillow",
            "win32gui": "pywin32",
            "win32ui": "pywin32",
            "win32con": "pywin32",
            "win32api": "pywin32"
        }
        
        for package, requirement in requirements:
            # Handle special cases
            if package in package_mappings:
                import_name = package
                package_name = package_mappings[package]
                full_requirement = package_mappings[package]  # Use base package name
            else:
                import_name = package.replace('-', '_')  # Python module naming
                package_name = package
                full_requirement = requirement
            
            if not self.check_package(import_name):
                print(f"❌ Missing: {package_name}")
                self.missing_packages.append(full_requirement)
            else:
                print(f"✅ Found: {package_name}")
    
    def install_packages(self):
        """Install missing packages"""
        if not self.missing_packages:
            print("✅ All requirements are satisfied!")
            return True
        
        print(f"📦 Installing {len(self.missing_packages)} missing packages...")
        
        for package in self.missing_packages:
            print(f"   Installing {package}...")
            try:
                # Use sys.executable to ensure we're using the correct Python
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"   ✅ {package} installed successfully")
                else:
                    print(f"   ❌ Failed to install {package}")
                    print(f"      Error: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"   ❌ Timeout installing {package}")
                return False
            except Exception as e:
                print(f"   ❌ Error installing {package}: {e}")
                return False
        
        print("✅ All packages installed successfully!")
        return True
    
    def check_system_compatibility(self):
        """Check system-specific compatibility"""
        print("🔍 Checking system compatibility...")
        
        # Check for Windows-specific features
        if self.system_info["os"] == "Windows":
            print("✅ Windows detected - full window capture support available")
        else:
            print("⚠️  Non-Windows system detected - limited window capture support")
            print("   Screen capture will still work, but individual window capture may be limited")
        
        # Check if running in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Virtual environment detected")
        else:
            print("⚠️  Not running in virtual environment (recommended but not required)")
        
        return True
    
    def find_main_app(self):
        """Find the main application file"""
        possible_apps = [
            "screen_capture_app.py",
            "main.py",
            "app.py"
        ]
        
        for app_file in possible_apps:
            app_path = self.script_dir / app_file
            if app_path.exists():
                print(f"✅ Found application: {app_file}")
                return app_path
        
        print("❌ Could not find main application file")
        print("   Expected one of:", ", ".join(possible_apps))
        return None
    
    def launch_application(self):
        """Launch the main application"""
        app_path = self.find_main_app()
        if not app_path:
            return False
        
        print("🚀 Launching Screen Capture Application...")
        print("-" * 60)
        
        try:
            # Launch the application
            subprocess.run([sys.executable, str(app_path)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Application exited with error code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\\n⚠️  Application interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Error launching application: {e}")
            return False
    
    def show_menu(self):
        """Show interactive menu"""
        while True:
            print("\\n" + "=" * 40)
            print("          MAIN MENU")
            print("=" * 40)
            print("1. Launch Screen Capture App")
            print("2. Launch Server Only")
            print("3. Install/Update Dependencies")
            print("4. System Information")
            print("5. Exit")
            print("-" * 40)
            
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    if self.launch_application():
                        break
                elif choice == "2":
                    self.launch_server()
                elif choice == "3":
                    self.install_dependencies()
                elif choice == "4":
                    self.show_system_info()
                elif choice == "5":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except EOFError:
                print("\\n👋 Goodbye!")
                break
    
    def launch_server(self):
        """Launch just the server"""
        server_path = self.script_dir / "capture_server.py"
        if server_path.exists():
            print("🚀 Launching Server...")
            try:
                subprocess.run([sys.executable, str(server_path)])
            except KeyboardInterrupt:
                print("\\n⚠️  Server stopped by user")
        else:
            print("❌ Server file not found: capture_server.py")
    
    def install_dependencies(self):
        """Force install/update dependencies"""
        print("🔄 Checking and installing dependencies...")
        self.missing_packages = []
        self.check_requirements()
        self.install_packages()
    
    def show_system_info(self):
        """Show detailed system information"""
        print("\\n" + "=" * 40)
        print("      SYSTEM INFORMATION")
        print("=" * 40)
        print(f"Operating System: {platform.platform()}")
        print(f"Python Version: {platform.python_version()}")
        print(f"Python Path: {sys.executable}")
        print(f"Architecture: {platform.architecture()}")
        print(f"Processor: {platform.processor()}")
        print(f"Script Directory: {self.script_dir}")
        
        # Check available GUI frameworks
        gui_frameworks = [
            ("tkinter", "tkinter"),
            ("PyQt5", "PyQt5.QtWidgets"),
            ("PyQt6", "PyQt6.QtWidgets"),
        ]
        
        print("\\nAvailable GUI Frameworks:")
        for name, module in gui_frameworks:
            if self.check_package(module):
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}")
        
        input("\\nPress Enter to continue...")
    
    def run(self):
        """Main launcher routine"""
        self.print_header()
        
        # Check Python version first
        if not self.check_python_version():
            input("Press Enter to exit...")
            return False
        
        # Check system compatibility
        if not self.check_system_compatibility():
            input("Press Enter to exit...")
            return False
        
        # Check and install requirements
        self.check_requirements()
        
        if self.missing_packages:
            print(f"\\n📦 Found {len(self.missing_packages)} missing packages.")
            response = input("Install missing packages? (Y/n): ").strip().lower()
            
            if response in ('', 'y', 'yes'):
                if not self.install_packages():
                    print("❌ Failed to install some packages. The application may not work correctly.")
                    response = input("Continue anyway? (y/N): ").strip().lower()
                    if response not in ('y', 'yes'):
                        return False
            else:
                print("⚠️  Proceeding without installing packages. Some features may not work.")
        
        # Show menu or launch directly
        if len(sys.argv) > 1 and sys.argv[1] == "--direct":
            return self.launch_application()
        else:
            self.show_menu()
            return True

def main():
    """Main entry point"""
    launcher = AppLauncher()
    return launcher.run()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
