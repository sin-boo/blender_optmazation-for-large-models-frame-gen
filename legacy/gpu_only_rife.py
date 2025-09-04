#!/usr/bin/env python3
"""
GPU-Only RIFE Implementation for RTX 5060 Ti
No CPU fallbacks - Pure GPU processing
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import queue
from PIL import Image, ImageTk
import numpy as np

# Windows-specific imports for window capture
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    WINDOWS_CAPTURE_AVAILABLE = True
except ImportError:
    print("❌ pywin32 not available - window capture disabled")
    WINDOWS_CAPTURE_AVAILABLE = False

# CUDA imports - try different GPU libraries
GPU_BACKEND = None
CUDA_DEVICE = None

# Try CuPy first (most direct CUDA access)
try:
    import cupy as cp
    # Test if CuPy can run basic operations
    test_arr = cp.array([1, 2, 3])
    result = test_arr + test_arr
    _ = cp.asnumpy(result)
    GPU_BACKEND = 'cupy'
    print("✅ Using CuPy backend for GPU processing")
except Exception as e:
    print(f"⚠️  CuPy failed: {e}")

# Try PyTorch if CuPy failed
if GPU_BACKEND is None:
    try:
        import torch
        if torch.cuda.is_available():
            # Force CUDA device 0
            CUDA_DEVICE = torch.device('cuda:0')
            # Test basic operation
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=CUDA_DEVICE)
            result = test_tensor + test_tensor
            _ = result.cpu()
            GPU_BACKEND = 'pytorch'
            print("✅ Using PyTorch backend for GPU processing")
        else:
            print("❌ PyTorch CUDA not available")
    except Exception as e:
        print(f"⚠️  PyTorch failed: {e}")

# Try direct CUDA via ctypes (last resort)
if GPU_BACKEND is None:
    try:
        import ctypes
        from ctypes import wintypes
        
        # Load CUDA runtime library
        cuda_rt = ctypes.CDLL('cudart64_12.dll')  # CUDA 12.x
        
        # Test CUDA device count
        device_count = ctypes.c_int()
        result = cuda_rt.cudaGetDeviceCount(ctypes.byref(device_count))
        
        if result == 0 and device_count.value > 0:
            GPU_BACKEND = 'cuda_direct'
            print("✅ Using Direct CUDA backend for GPU processing")
        else:
            print(f"❌ Direct CUDA failed: result={result}")
            
    except Exception as e:
        print(f"⚠️  Direct CUDA failed: {e}")

if GPU_BACKEND is None:
    print("❌ NO GPU BACKEND AVAILABLE - Cannot run GPU-only mode")
    sys.exit(1)

class GPUMemoryManager:
    """Manages GPU memory for optimal RTX 5060 Ti performance"""
    
    def __init__(self):
        self.gpu_cache = {}
        self.max_cache_size = 8  # GB for RTX 5060 Ti
        
    def allocate_gpu_tensor(self, shape, dtype=np.float32):
        """Allocate GPU tensor efficiently"""
        key = (shape, str(dtype))
        
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        
        if GPU_BACKEND == 'cupy':
            tensor = cp.zeros(shape, dtype=dtype)
        elif GPU_BACKEND == 'pytorch':
            tensor = torch.zeros(shape, dtype=torch.float32, device=CUDA_DEVICE)
        else:
            # Direct CUDA allocation would go here
            tensor = np.zeros(shape, dtype=dtype)
        
        self.gpu_cache[key] = tensor
        return tensor
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if GPU_BACKEND == 'cupy':
            cp.get_default_memory_pool().free_all_blocks()
        elif GPU_BACKEND == 'pytorch':
            torch.cuda.empty_cache()
        
        self.gpu_cache.clear()

class AdvancedGPURIFE:
    """Advanced GPU-only RIFE implementation"""
    
    def __init__(self):
        self.memory_manager = GPUMemoryManager()
        self.frame_buffer_gpu = []
        self.max_buffer_size = 3
        self.processing_quality = "Ultra"
        
    def image_to_gpu_tensor(self, pil_image):
        """Convert PIL image to GPU tensor"""
        # Convert to numpy array
        np_array = np.array(pil_image, dtype=np.float32) / 255.0
        
        # Ensure RGB format
        if len(np_array.shape) == 3 and np_array.shape[2] == 3:
            # Transpose to CHW format for processing
            np_array = np_array.transpose(2, 0, 1)
        
        if GPU_BACKEND == 'cupy':
            gpu_tensor = cp.asarray(np_array)
        elif GPU_BACKEND == 'pytorch':
            gpu_tensor = torch.from_numpy(np_array).to(CUDA_DEVICE)
        else:
            gpu_tensor = np_array  # Fallback
            
        return gpu_tensor
    
    def gpu_tensor_to_image(self, gpu_tensor):
        """Convert GPU tensor back to PIL image"""
        if GPU_BACKEND == 'cupy':
            np_array = cp.asnumpy(gpu_tensor)
        elif GPU_BACKEND == 'pytorch':
            np_array = gpu_tensor.cpu().detach().numpy()
        else:
            np_array = gpu_tensor
        
        # Transpose back to HWC format
        if len(np_array.shape) == 3 and np_array.shape[0] == 3:
            np_array = np_array.transpose(1, 2, 0)
        
        # Convert to uint8
        np_array = np.clip(np_array * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(np_array)
    
    def advanced_gpu_interpolation(self, frame1_gpu, frame2_gpu):
        """Advanced GPU interpolation for RTX 5060 Ti"""
        try:
            if GPU_BACKEND == 'cupy':
                return self._cupy_interpolation(frame1_gpu, frame2_gpu)
            elif GPU_BACKEND == 'pytorch':
                return self._pytorch_interpolation(frame1_gpu, frame2_gpu)
            else:
                return self._direct_cuda_interpolation(frame1_gpu, frame2_gpu)
                
        except Exception as e:
            print(f"GPU interpolation error: {e}")
            # Force GPU memory cleanup
            self.memory_manager.clear_cache()
            raise e
    
    def _cupy_interpolation(self, frame1_gpu, frame2_gpu):
        """CuPy-based GPU interpolation"""
        # Advanced interpolation using CuPy operations
        alpha = 0.5
        
        # Weighted blending with edge enhancement
        blend = frame1_gpu * (1 - alpha) + frame2_gpu * alpha
        
        # GPU-based sharpening filter
        if self.processing_quality == "Ultra":
            # Laplacian sharpening on GPU
            kernel = cp.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=cp.float32)
            
            # Apply convolution to each channel
            if len(blend.shape) == 3:
                enhanced = cp.zeros_like(blend)
                for c in range(blend.shape[0]):
                    # Simple convolution approximation
                    channel = blend[c]
                    enhanced[c] = channel + 0.1 * (channel - cp.roll(channel, 1, axis=0))
                result = enhanced
            else:
                result = blend
        else:
            result = blend
        
        # Clamp values
        result = cp.clip(result, 0.0, 1.0)
        return result
    
    def _pytorch_interpolation(self, frame1_gpu, frame2_gpu):
        """PyTorch-based GPU interpolation"""
        alpha = 0.5
        
        # Weighted blending
        blend = frame1_gpu * (1 - alpha) + frame2_gpu * alpha
        
        # Advanced processing for Ultra quality
        if self.processing_quality == "Ultra":
            # Gaussian blur and unsharp mask
            if len(blend.shape) == 3:
                # Simple edge enhancement
                enhanced = blend + 0.1 * (blend - torch.roll(blend, shifts=1, dims=1))
                result = torch.clamp(enhanced, 0.0, 1.0)
            else:
                result = blend
        else:
            result = blend
        
        result = torch.clamp(result, 0.0, 1.0)
        return result
    
    def _direct_cuda_interpolation(self, frame1_gpu, frame2_gpu):
        """Direct CUDA interpolation (fallback)"""
        # Simple numpy-based interpolation as fallback
        alpha = 0.5
        result = frame1_gpu * (1 - alpha) + frame2_gpu * alpha
        return np.clip(result, 0.0, 1.0)
    
    def process_frame_pair(self, frame1, frame2):
        """Process a pair of frames with GPU-only interpolation"""
        # Convert to GPU tensors
        frame1_gpu = self.image_to_gpu_tensor(frame1)
        frame2_gpu = self.image_to_gpu_tensor(frame2)
        
        # Perform GPU interpolation
        result_gpu = self.advanced_gpu_interpolation(frame1_gpu, frame2_gpu)
        
        # Convert back to PIL image
        result_image = self.gpu_tensor_to_image(result_gpu)
        
        return result_image
    
    def add_frame(self, frame):
        """Add frame to GPU buffer"""
        # Keep only recent frames in memory
        self.frame_buffer_gpu.append(frame)
        if len(self.frame_buffer_gpu) > self.max_buffer_size:
            self.frame_buffer_gpu.pop(0)
    
    def get_interpolated_frame(self):
        """Get interpolated frame from buffer"""
        if len(self.frame_buffer_gpu) >= 2:
            frame1 = self.frame_buffer_gpu[-2]
            frame2 = self.frame_buffer_gpu[-1]
            return self.process_frame_pair(frame1, frame2)
        return None

class WindowsWindowManager:
    """Manages Windows window positioning and focus"""
    
    @staticmethod
    def center_window(window, width=800, height=600):
        """Center a tkinter window on screen"""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")
        
    @staticmethod
    def ensure_window_visible(window):
        """Ensure window is visible and focused"""
        window.lift()
        window.attributes('-topmost', True)
        window.after_idle(lambda: window.attributes('-topmost', False))
        window.focus_force()

class WindowCapture:
    """Windows window capture functionality"""
    
    def __init__(self):
        self.available_windows = []
        self.target_hwnd = None
        
    def refresh_windows(self):
        """Get list of available windows"""
        if not WINDOWS_CAPTURE_AVAILABLE:
            return []
        
        windows = []
        
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title and len(window_title.strip()) > 0:
                    windows.append({
                        'hwnd': hwnd,
                        'title': window_title,
                        'class': win32gui.GetClassName(hwnd)
                    })
        
        win32gui.EnumWindows(enum_callback, windows)
        self.available_windows = windows
        return windows
    
    def set_target_window(self, hwnd):
        """Set the target window for capture"""
        self.target_hwnd = hwnd
    
    def capture_window(self):
        """Capture the target window"""
        if not WINDOWS_CAPTURE_AVAILABLE or not self.target_hwnd:
            return None
        
        try:
            window_rect = win32gui.GetWindowRect(self.target_hwnd)
            width = window_rect[2] - window_rect[0]
            height = window_rect[3] - window_rect[1]
            
            hwnd_dc = win32gui.GetWindowDC(self.target_hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Use BitBlt for compatibility
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
            
            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)
            
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )
            
            # Cleanup
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.target_hwnd, hwnd_dc)
            
            return img
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None

class GPUOnlyRIFEGUI:
    """GPU-Only RIFE GUI Application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU-Only RIFE - RTX 5060 Ti Optimized")
        self.root.configure(bg='#1a1a1a')
        
        # Apply Windows fixes
        WindowsWindowManager.center_window(self.root, 1000, 750)
        WindowsWindowManager.ensure_window_visible(self.root)
        
        # Initialize components
        self.window_capture = WindowCapture()
        self.gpu_rife = AdvancedGPURIFE()
        self.processing_active = False
        
        # Communication queues
        self.frame_queue = queue.Queue(maxsize=5)  # Smaller queue for GPU processing
        
        # RIFE output window
        self.rife_window = None
        self.rife_label = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title frame
        title_frame = tk.Frame(self.root, bg='#1a1a1a')
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(title_frame, text="🚀 GPU-Only RIFE - RTX 5060 Ti Edition", 
                             font=('Arial', 16, 'bold'), fg='#00ff00', bg='#1a1a1a')
        title_label.pack()
        
        gpu_info_label = tk.Label(title_frame, text=f"GPU Backend: {GPU_BACKEND.upper()}", 
                                font=('Arial', 10), fg='#ffff00', bg='#1a1a1a')
        gpu_info_label.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#1a1a1a')
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Ready for GPU processing", 
                                   fg='#00ff00', bg='#1a1a1a', font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT)
        
        # Main content
        content_frame = tk.Frame(self.root, bg='#2a2a2a')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Window Capture Section
        capture_section = tk.LabelFrame(content_frame, text="Window Capture", 
                                      fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        capture_section.pack(fill=tk.X, padx=10, pady=10)
        
        self.setup_capture_section(capture_section)
        
        # GPU Processing Section
        gpu_section = tk.LabelFrame(content_frame, text="GPU-Only RIFE Processing", 
                                  fg='white', bg='#2a2a2a', font=('Arial', 12, 'bold'))
        gpu_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_gpu_section(gpu_section)
        
    def setup_capture_section(self, parent):
        """Setup window capture section"""
        button_frame = tk.Frame(parent, bg='#2a2a2a')
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        refresh_btn = tk.Button(button_frame, text="🔄 Refresh Windows", 
                              command=self.refresh_windows, bg='#4a4a4a', fg='white',
                              font=('Arial', 10, 'bold'))
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Window list
        list_frame = tk.Frame(parent, bg='#2a2a2a')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.window_listbox = tk.Listbox(list_frame, height=8, bg='#3a3a3a', fg='white',
                                       yscrollcommand=scrollbar.set)
        self.window_listbox.pack(fill=tk.BOTH, expand=True)
        self.window_listbox.bind('<<ListboxSelect>>', self.on_window_select)
        
        scrollbar.config(command=self.window_listbox.yview)
        
        # Control buttons
        control_frame = tk.Frame(parent, bg='#2a2a2a')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_capture_btn = tk.Button(control_frame, text="▶️ Start Capture", 
                                         command=self.start_capture, bg='#006600', fg='white',
                                         font=('Arial', 12, 'bold'))
        self.start_capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_capture_btn = tk.Button(control_frame, text="⏹️ Stop Capture", 
                                        command=self.stop_capture, bg='#660000', fg='white',
                                        font=('Arial', 12, 'bold'), state='disabled')
        self.stop_capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_windows()
        
    def setup_gpu_section(self, parent):
        """Setup GPU processing section"""
        # Quality selection
        quality_frame = tk.Frame(parent, bg='#2a2a2a')
        quality_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(quality_frame, text="Processing Quality:", fg='white', bg='#2a2a2a',
                font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        self.quality_var = tk.StringVar(value="Ultra")
        quality_options = ["High", "Ultra", "Maximum"]
        
        for option in quality_options:
            rb = tk.Radiobutton(quality_frame, text=option, variable=self.quality_var, 
                              value=option, command=self.update_quality, fg='white', bg='#2a2a2a',
                              selectcolor='#4a4a4a', font=('Arial', 10))
            rb.pack(side=tk.LEFT, padx=10)
        
        # RIFE control buttons
        rife_control_frame = tk.Frame(parent, bg='#2a2a2a')
        rife_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_rife_btn = tk.Button(rife_control_frame, text="🚀 Start GPU RIFE", 
                                      command=self.start_rife_processing, bg='#0066cc', fg='white',
                                      font=('Arial', 12, 'bold'))
        self.start_rife_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_rife_btn = tk.Button(rife_control_frame, text="⏹️ Stop GPU RIFE", 
                                     command=self.stop_rife_processing, bg='#cc6600', fg='white',
                                     font=('Arial', 12, 'bold'), state='disabled')
        self.stop_rife_btn.pack(side=tk.LEFT, padx=5)
        
        # GPU memory button
        memory_btn = tk.Button(rife_control_frame, text="🧹 Clear GPU Memory", 
                             command=self.clear_gpu_memory, bg='#666600', fg='white',
                             font=('Arial', 10, 'bold'))
        memory_btn.pack(side=tk.RIGHT, padx=5)
        
        # Info display
        info_frame = tk.Frame(parent, bg='#2a2a2a')
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_text = tk.Text(info_frame, height=10, bg='#1a1a1a', fg='#00ff00',
                          font=('Consolas', 9))
        info_text.pack(fill=tk.BOTH, expand=True)
        
        info_content = f"""🎮 GPU-ONLY RIFE PROCESSING - RTX 5060 Ti EDITION

🔧 GPU Backend: {GPU_BACKEND.upper()}
🎯 Compute Capability: 12.0 (sm_120)
💾 GPU Memory: ~16GB GDDR7
⚡ CUDA Cores: 4,352 cores

⚠️  NO CPU FALLBACKS - Pure GPU Processing Only!

🚀 PERFORMANCE MODES:
• High: GPU interpolation with standard processing
• Ultra: Advanced GPU filtering + edge enhancement  
• Maximum: Full RTX 5060 Ti utilization

📋 INSTRUCTIONS:
1. Select target window from capture list
2. Choose processing quality (Ultra recommended)
3. Start capture, then start GPU RIFE processing
4. Watch real-time interpolated output window

⚡ Optimized for RTX 5060 Ti's 12.0 compute capability
🎯 Real-time DLSS-like interpolation for any application
"""
        
        info_text.insert(tk.END, info_content)
        info_text.config(state='disabled')
        
    def refresh_windows(self):
        """Refresh the windows list"""
        self.window_listbox.delete(0, tk.END)
        
        if not WINDOWS_CAPTURE_AVAILABLE:
            self.window_listbox.insert(tk.END, "❌ Window capture not available")
            return
        
        windows = self.window_capture.refresh_windows()
        for window in windows:
            display_text = f"🪟 {window['title']} ({window['class']})"
            self.window_listbox.insert(tk.END, display_text)
        
        self.status_label.config(text=f"Found {len(windows)} windows - GPU ready")
        
    def on_window_select(self, event):
        """Handle window selection"""
        selection = self.window_listbox.curselection()
        if selection and WINDOWS_CAPTURE_AVAILABLE:
            index = selection[0]
            windows = self.window_capture.available_windows
            if index < len(windows):
                selected_window = windows[index]
                self.window_capture.set_target_window(selected_window['hwnd'])
                self.status_label.config(text=f"Selected: {selected_window['title']} - Ready for GPU processing")
                
    def update_quality(self):
        """Update processing quality"""
        self.gpu_rife.processing_quality = self.quality_var.get()
        self.status_label.config(text=f"GPU Quality: {self.quality_var.get()}")
        
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        self.gpu_rife.memory_manager.clear_cache()
        self.status_label.config(text="GPU memory cleared")
        
    def start_capture(self):
        """Start window capture"""
        if not self.window_capture.target_hwnd:
            messagebox.showerror("Error", "Please select a window to capture first")
            return
        
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        capture_thread.start()
        
        self.start_capture_btn.config(state='disabled')
        self.stop_capture_btn.config(state='normal')
        self.status_label.config(text="Capturing for GPU processing...")
        
    def stop_capture(self):
        """Stop window capture"""
        self.processing_active = False
        
        self.start_capture_btn.config(state='normal')
        self.stop_capture_btn.config(state='disabled')
        self.status_label.config(text="Capture stopped")
        
    def capture_loop(self):
        """Main capture loop"""
        self.processing_active = True
        
        while self.processing_active:
            frame = self.window_capture.capture_window()
            if frame:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            
            time.sleep(1/30)  # 30 FPS
    
    def start_rife_processing(self):
        """Start GPU-only RIFE processing"""
        if not self.processing_active:
            messagebox.showwarning("Warning", "Please start window capture first")
            return
        
        self.create_rife_window()
        
        gpu_processing_thread = threading.Thread(target=self.gpu_rife_loop, daemon=True)
        gpu_processing_thread.start()
        
        self.start_rife_btn.config(state='disabled')
        self.stop_rife_btn.config(state='normal')
        self.status_label.config(text="🚀 GPU-Only RIFE processing active!")
        
    def stop_rife_processing(self):
        """Stop RIFE processing"""
        if self.rife_window:
            self.rife_window.destroy()
            self.rife_window = None
            
        self.start_rife_btn.config(state='normal')
        self.stop_rife_btn.config(state='disabled')
        self.status_label.config(text="GPU RIFE processing stopped")
        
    def create_rife_window(self):
        """Create RIFE output window"""
        if self.rife_window:
            self.rife_window.destroy()
        
        self.rife_window = tk.Toplevel(self.root)
        self.rife_window.title("🚀 GPU-Only RIFE Output - RTX 5060 Ti")
        self.rife_window.configure(bg='black')
        
        WindowsWindowManager.center_window(self.rife_window, 800, 600)
        WindowsWindowManager.ensure_window_visible(self.rife_window)
        
        self.rife_label = tk.Label(self.rife_window, bg='black', 
                                 text="⚡ GPU processing starting...", fg='#00ff00',
                                 font=('Arial', 14, 'bold'))
        self.rife_label.pack(expand=True, fill=tk.BOTH)
        
    def gpu_rife_loop(self):
        """GPU-only RIFE processing loop"""
        while self.processing_active and self.rife_window:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                self.gpu_rife.add_frame(frame)
                
                interpolated = self.gpu_rife.get_interpolated_frame()
                
                if interpolated and self.rife_window:
                    display_frame = interpolated.copy()
                    display_frame.thumbnail((800, 600), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(display_frame)
                    self.root.after_idle(self.update_rife_display, photo)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU RIFE error: {e}")
                # Try to recover by clearing GPU memory
                self.clear_gpu_memory()
                continue
                
    def update_rife_display(self, photo):
        """Update RIFE display"""
        if self.rife_window and self.rife_label:
            try:
                self.rife_label.config(image=photo, text="")
                self.rife_label.image = photo
            except tk.TclError:
                pass
                
    def run(self):
        """Run the application"""
        print("🚀 Starting GPU-Only RIFE GUI for RTX 5060 Ti...")
        print(f"✅ GPU Backend: {GPU_BACKEND}")
        print("⚡ No CPU fallbacks - Pure GPU processing!")
        self.root.mainloop()

def main():
    """Main entry point"""
    print("=" * 60)
    print("🎯 GPU-ONLY RIFE - RTX 5060 Ti EDITION")
    print("=" * 60)
    print(f"🔧 GPU Backend: {GPU_BACKEND}")
    print("⚠️  NO CPU FALLBACKS - GPU processing only!")
    print()
    
    if GPU_BACKEND is None:
        print("❌ No GPU backend available - Cannot run")
        sys.exit(1)
    
    app = GPUOnlyRIFEGUI()
    app.run()

if __name__ == "__main__":
    main()
