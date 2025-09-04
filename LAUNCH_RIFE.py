#!/usr/bin/env python3
"""
🚀 GPU-Only RIFE Launcher
Simplified entry point for all RIFE implementations

This launcher helps you choose the right RIFE implementation for your needs:
- Performance-optimized versions for RTX GPUs
- Automated benchmarking and testing
- Easy access to all tools and configurations
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

class RIFELauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 GPU-Only RIFE Launcher")
        self.root.geometry("600x500")
        self.root.configure(bg='#2b2b2b')
        
        # Get project paths
        self.project_dir = Path(__file__).parent
        self.core_dir = self.project_dir / "core"
        self.tools_dir = self.project_dir / "tools"
        self.docs_dir = self.project_dir / "docs"
        
        self._create_gui()
    
    def _create_gui(self):
        """Create the launcher GUI"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title = tk.Label(title_frame, 
                        text="🚀 GPU-Only RIFE Launcher", 
                        font=('Arial', 18, 'bold'),
                        fg='#ffffff', bg='#2b2b2b')
        title.pack()
        
        subtitle = tk.Label(title_frame,
                           text="Choose your RIFE implementation",
                           font=('Arial', 10),
                           fg='#cccccc', bg='#2b2b2b')
        subtitle.pack()
        
        # Main options
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Performance implementations
        perf_frame = tk.LabelFrame(main_frame, text="🔥 Performance Implementations", 
                                  fg='#ffffff', bg='#2b2b2b', font=('Arial', 12, 'bold'))
        perf_frame.pack(fill=tk.X, pady=5)
        
        self._create_button(perf_frame, "🚀 Optimized GPU RIFE", 
                           "High-performance GPU-only implementation with advanced optimizations",
                           lambda: self._launch_script("core/gpu_rife_optimized.py"))
        
        self._create_button(perf_frame, "💎 RTX 5060 Ti Edition", 
                           "Specifically optimized for NVIDIA RTX 5060 Ti (16GB VRAM)",
                           lambda: self._launch_script("core/gpu_only_rtx_5060_ti.py"))
        
        self._create_button(perf_frame, "⚡ Final GPU-Only RIFE", 
                           "Complete GPU-only implementation with real-time processing",
                           lambda: self._launch_script("core/final_gpu_only_rife.py"))
        
        # Tools and utilities
        tools_frame = tk.LabelFrame(main_frame, text="🛠️ Tools & Utilities", 
                                   fg='#ffffff', bg='#2b2b2b', font=('Arial', 12, 'bold'))
        tools_frame.pack(fill=tk.X, pady=5)
        
        self._create_button(tools_frame, "📊 Performance Benchmark", 
                           "Comprehensive GPU performance testing and analysis",
                           lambda: self._launch_script("gpu_benchmark.py"))
        
        self._create_button(tools_frame, "🔍 GPU Status Check", 
                           "Check CUDA installation and GPU compatibility",
                           lambda: self._launch_script("tools/check_gpu_cuda_status.py"))
        
        self._create_button(tools_frame, "⚙️ Auto GPU Setup", 
                           "Automatic GPU environment setup and optimization",
                           lambda: self._launch_script("tools/auto_gpu_setup.py"))
        
        # Legacy and examples
        legacy_frame = tk.LabelFrame(main_frame, text="📁 Legacy & Examples", 
                                    fg='#ffffff', bg='#2b2b2b', font=('Arial', 12, 'bold'))
        legacy_frame.pack(fill=tk.X, pady=5)
        
        self._create_button(legacy_frame, "🎬 Traditional RIFE GUI", 
                           "Original RIFE implementation with GUI",
                           lambda: self._launch_script("rife_gui_app.py"))
        
        self._create_button(legacy_frame, "📖 View Documentation", 
                           "Open project documentation and guides",
                           self._open_docs)
        
        # Footer with system info
        footer_frame = tk.Frame(self.root, bg='#2b2b2b')
        footer_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self._create_system_info(footer_frame)
    
    def _create_button(self, parent, title, description, command):
        """Create a styled button with description"""
        button_frame = tk.Frame(parent, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Main button
        btn = tk.Button(button_frame, text=title, command=command,
                       font=('Arial', 10, 'bold'), fg='#ffffff', bg='#0078d4',
                       activebackground='#106ebe', activeforeground='#ffffff',
                       width=25, height=1, relief=tk.FLAT)
        btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Description
        desc_label = tk.Label(button_frame, text=description,
                             font=('Arial', 9), fg='#cccccc', bg='#2b2b2b',
                             wraplength=300, justify=tk.LEFT)
        desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _create_system_info(self, parent):
        """Create system information display"""
        info_frame = tk.LabelFrame(parent, text="💻 System Information", 
                                  fg='#ffffff', bg='#2b2b2b', font=('Arial', 10, 'bold'))
        info_frame.pack(fill=tk.X)
        
        # Get system information
        try:
            import cupy as cp
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
            gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3)
            gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            gpu_color = '#00ff00'  # Green for GPU available
        except:
            gpu_info = "GPU: CUDA not available or not installed"
            gpu_color = '#ff6b6b'  # Red for GPU issues
        
        # Python version
        python_info = f"Python: {sys.version.split()[0]}"
        
        # Display info
        gpu_label = tk.Label(info_frame, text=gpu_info, font=('Arial', 9),
                            fg=gpu_color, bg='#2b2b2b')
        gpu_label.pack(anchor=tk.W, padx=10, pady=2)
        
        python_label = tk.Label(info_frame, text=python_info, font=('Arial', 9),
                               fg='#cccccc', bg='#2b2b2b')
        python_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Project directory
        project_label = tk.Label(info_frame, text=f"Project: {self.project_dir}",
                                 font=('Arial', 9), fg='#cccccc', bg='#2b2b2b')
        project_label.pack(anchor=tk.W, padx=10, pady=2)
    
    def _launch_script(self, script_path):
        """Launch a Python script"""
        try:
            full_path = self.project_dir / script_path
            
            if not full_path.exists():
                messagebox.showerror("File Not Found", 
                                   f"Script not found: {script_path}\\n\\n"
                                   f"Looking for: {full_path}")
                return
            
            # Launch the script in a new process
            subprocess.Popen([sys.executable, str(full_path)], 
                           cwd=str(self.project_dir))
            
            messagebox.showinfo("Launched", f"Started: {script_path}")
            
        except Exception as e:
            messagebox.showerror("Launch Error", 
                               f"Failed to launch {script_path}:\\n{str(e)}")
    
    def _open_docs(self):
        """Open documentation"""
        try:
            # Check for README.md
            readme_path = self.project_dir / "README.md"
            if readme_path.exists():
                if sys.platform == "win32":
                    os.startfile(str(readme_path))
                else:
                    subprocess.run(["xdg-open", str(readme_path)])
            else:
                # Open docs directory
                docs_path = self.docs_dir
                if docs_path.exists():
                    if sys.platform == "win32":
                        os.startfile(str(docs_path))
                    else:
                        subprocess.run(["xdg-open", str(docs_path)])
                else:
                    messagebox.showinfo("Documentation", 
                                      "Documentation files not found. "
                                      "Check the project README.md for information.")
            
        except Exception as e:
            messagebox.showerror("Documentation Error", 
                               f"Failed to open documentation:\\n{str(e)}")
    
    def run(self):
        """Run the launcher"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("🚀 Starting GPU-Only RIFE Launcher...")
    
    try:
        launcher = RIFELauncher()
        launcher.run()
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
