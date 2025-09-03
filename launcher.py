#!/usr/bin/env python3
"""
RIFE GUI Launcher - Complete installer and launcher
Installs all requirements and launches the GUI
"""

import subprocess
import sys
import importlib
import os

def install_package(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_package(package_name):
    """Check if a package is already installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_requirements():
    """Install all required packages if not already installed"""
    print("RIFE GUI Complete Installer & Launcher")
    print("=" * 40)
    print("Checking requirements...")
    
    required_packages = {
        'PyQt5': 'PyQt5',
        # Add more packages here as needed
        # 'torch': 'torch',
        # 'opencv-python': 'cv2',
    }
    
    all_installed = True
    
    for display_name, import_name in required_packages.items():
        print(f"Checking {display_name}...")
        
        if check_package(import_name):
            print(f"✓ {display_name} is already installed")
        else:
            print(f"✗ {display_name} not found. Installing...")
            try:
                install_package(display_name)
                print(f"✓ {display_name} installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {display_name}: {e}")
                all_installed = False
    
    if all_installed:
        print("✓ All requirements satisfied!")
    
    return all_installed

def main():
    print("RIFE GUI Launcher")
    print("=" * 30)
    
    # Install requirements
    if not install_requirements():
        input("Press Enter to exit...")
        return
    
    print("\nLaunching RIFE GUI...")
    
    # Try to run the GUI
    gui_files = ['rife_gui_app.py', 'gui.py']
    
    for gui_file in gui_files:
        if os.path.exists(gui_file):
            try:
                subprocess.run([sys.executable, gui_file])
                return
            except Exception as e:
                print(f"Failed to run {gui_file}: {e}")
    
    print("No GUI file found! Make sure rife_gui_app.py or gui.py exists in this folder.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()