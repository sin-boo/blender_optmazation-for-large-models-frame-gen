#!/usr/bin/env python3
"""
Screen Capture Application - Main Entry Point
Self-sufficient application with modern GUI
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point for the screen capture application"""
    
    # Add current directory to Python path  
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    print("🖥️ Starting Screen Capture Application...")
    
    # Try to import and run the modern GUI
    try:
        from modern_gui import ModernScreenCaptureApp
        print("✅ Using modern GUI interface")
        
        app = ModernScreenCaptureApp()
        app.run()
        
    except ImportError as e:
        print(f"❌ Could not load modern GUI: {e}")
        print("📦 Some dependencies might be missing.")
        print("💡 Try running the main 'run.py' script to install dependencies.")
        
        # Try basic fallback
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            
            messagebox.showerror(
                "Missing Dependencies",
                "Some required packages are missing.\\n\\n"
                "Please run 'run.py' first to install dependencies,\\n"
                "or install manually with:\\n\\n"
                "pip install Pillow mss requests flask pywin32"
            )
            
        except ImportError:
            print("❌ Even tkinter GUI not available!")
            print("🔧 Please run 'run.py' to set up the environment.")
            input("Press Enter to exit...")
        
        return False
    
    except Exception as e:
        print(f"❌ Application error: {e}")
        input("Press Enter to exit...")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
