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

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_requirements():
    """Install all required packages if not already installed"""
    print("RIFE GUI Complete Installer & Launcher")
    print("=" * 40)
    print("Checking requirements...")
    
    # Check PyQt5
    if not check_package('PyQt5'):
        print("✗ PyQt5 not found. Installing...")
        try:
            install_package('PyQt5')
            print("✓ PyQt5 installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install PyQt5: {e}")
            return False
    else:
        print("✓ PyQt5 is already installed")
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("✗ FFmpeg not found!")
        print("  Please install FFmpeg from: https://ffmpeg.org/download.html")
        print("  Or on Windows: winget install ffmpeg")
        print("  The GUI will still open, but video processing won't work without FFmpeg.")
    else:
        print("✓ FFmpeg is available")
    
    return True

def launch_gui():
    """Launch the RIFE GUI"""
    print("\n" + "=" * 40)
    print("Launching RIFE GUI...")
    
    # Look for the GUI file
    gui_file = 'rife_gui_app.py'
    
    if os.path.exists(gui_file):
        try:
            subprocess.run([sys.executable, gui_file])
            print("GUI closed successfully.")
        except Exception as e:
            print(f"Failed to run {gui_file}: {e}")
            input("Press Enter to exit...")
    else:
        print(f"Error: {gui_file} not found in current directory!")
        print("Make sure rife_gui_app.py exists in this folder.")
        input("Press Enter to exit...")

def main():
    # Install requirements first
    if not install_requirements():
        input("Press Enter to exit...")
        return
    
    # Launch the GUI
    launch_gui()

if __name__ == "__main__":
    main()