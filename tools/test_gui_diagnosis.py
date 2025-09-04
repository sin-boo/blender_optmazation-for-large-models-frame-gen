#!/usr/bin/env python3
"""
Simple GUI Test to diagnose Windows display issues
"""

import tkinter as tk
from tkinter import ttk
import sys
import threading
import time

def test_basic_gui():
    """Test basic tkinter GUI functionality"""
    print("🧪 Testing Basic GUI Functionality")
    print("=" * 40)
    
    try:
        # Create main window
        root = tk.Tk()
        root.title("RIFE GUI - Basic Test")
        root.geometry("400x300")
        root.configure(bg='#2b2b2b')
        
        # Add test elements
        label = tk.Label(root, text="✅ GUI Test Window", 
                        font=("Arial", 16, "bold"), 
                        fg='white', bg='#2b2b2b')
        label.pack(pady=20)
        
        info_text = tk.Text(root, height=10, width=50)
        info_text.pack(pady=10)
        info_text.insert(tk.END, "GUI Test Results:\n")
        info_text.insert(tk.END, "✅ Window created successfully\n")
        info_text.insert(tk.END, "✅ Tkinter working properly\n")
        info_text.insert(tk.END, "✅ Text widgets functional\n")
        info_text.insert(tk.END, "\nThis window should be visible!\n")
        info_text.insert(tk.END, "If you can see this, the GUI system works.\n")
        
        # Test button
        def on_test_click():
            info_text.insert(tk.END, "✅ Button click works!\n")
            info_text.see(tk.END)
        
        test_button = tk.Button(root, text="Test Button", command=on_test_click)
        test_button.pack(pady=10)
        
        # Force window to front
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(lambda: root.attributes('-topmost', False))
        
        print("✅ GUI window created - should be visible now!")
        print("   If you don't see it, there may be a Windows display issue.")
        
        # Run for 5 seconds then close
        def auto_close():
            time.sleep(5)
            root.quit()
            
        close_thread = threading.Thread(target=auto_close)
        close_thread.daemon = True
        close_thread.start()
        
        root.mainloop()
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def test_window_threading():
    """Test if threading with GUI works"""
    print("\n🧪 Testing GUI Threading")
    print("=" * 40)
    
    try:
        import queue
        
        root = tk.Tk()
        root.title("Threading Test")
        root.geometry("300x200")
        
        # Queue for thread communication
        msg_queue = queue.Queue()
        
        text_widget = tk.Text(root, height=8, width=40)
        text_widget.pack(pady=10)
        
        def worker_thread():
            """Background thread that sends messages"""
            for i in range(5):
                time.sleep(1)
                msg_queue.put(f"Message {i+1} from background thread\n")
            msg_queue.put("DONE")
        
        def check_queue():
            """Check for messages from background thread"""
            try:
                while True:
                    message = msg_queue.get_nowait()
                    if message == "DONE":
                        text_widget.insert(tk.END, "✅ Threading test complete!\n")
                        root.after(2000, root.quit)  # Close after 2 seconds
                        break
                    else:
                        text_widget.insert(tk.END, message)
                        text_widget.see(tk.END)
            except queue.Empty:
                pass
            
            # Check again in 100ms
            root.after(100, check_queue)
        
        # Start background thread
        thread = threading.Thread(target=worker_thread)
        thread.daemon = True
        thread.start()
        
        # Start queue checking
        root.after(100, check_queue)
        
        print("✅ Threading test window should appear...")
        root.mainloop()
        return True
        
    except Exception as e:
        print(f"❌ Threading test failed: {e}")
        return False

def main():
    print("🔧 DIAGNOSING RIFE GUI ISSUES")
    print("Testing core GUI functionality on Windows")
    print()
    
    # Test 1: Basic GUI
    basic_success = test_basic_gui()
    
    # Test 2: Threading with GUI
    thread_success = test_window_threading()
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS RESULTS:")
    
    if basic_success and thread_success:
        print("✅ GUI SYSTEM: WORKING!")
        print("✅ Threading integration: WORKING!")
        print("✅ The main RIFE GUI should work properly")
        
        print("\n💡 ISSUE LIKELY CAUSES:")
        print("   • Window may be opening off-screen")
        print("   • Main window might be minimized")
        print("   • App could be running in background")
        print("   • Windows focus/topmost issues")
        
        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Add window positioning and focus code")
        print("   2. Force window to center of screen")
        print("   3. Add taskbar icon and window management")
        print("   4. Improve error handling for GUI updates")
        
    elif basic_success and not thread_success:
        print("✅ Basic GUI: WORKING")
        print("❌ Threading: ISSUES")
        print("🔧 Fix: Threading problems in RIFE processing")
        
    elif not basic_success:
        print("❌ Basic GUI: BROKEN")
        print("🔧 Fix: Fundamental tkinter/Python issues")
        
    print(f"\n🚀 Next: Run the improved RIFE GUI with fixes")

if __name__ == "__main__":
    main()
