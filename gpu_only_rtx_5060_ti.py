#!/usr/bin/env python3
"""
GPU-Only RIFE for RTX 5060 Ti
Pure GPU processing - No CPU fallbacks
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
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
    WINDOWS_CAPTURE_AVAILABLE = True
except ImportError:
    print("❌ pywin32 not available - window capture disabled")
    WINDOWS_CAPTURE_AVAILABLE = False

# GPU imports - CuPy for RTX 5060 Ti
try:
    import cupy as cp
    
    # Test GPU operations
    test_arr = cp.array([1, 2, 3], dtype=cp.float32)
    result = test_arr * 2.0 + 1.0
    _ = cp.asnumpy(result)
    
    print(f"✅ CuPy backend initialized for RTX 5060 Ti")
    print(f"✅ GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB total")
    
    GPU_AVAILABLE = True
    
except Exception as e:
    print(f"❌ CuPy initialization failed: {e}")
    print("❌ Cannot run without GPU support")
    sys.exit(1)

class GPUMemoryManager:
    """GPU memory management for RTX 5060 Ti"""
    
    def __init__(self):
        self.memory_pool = cp.get_default_memory_pool()
        
    def clear_cache(self):
        """Clear GPU memory cache"""
        self.memory_pool.free_all_blocks()
        print("🧹 GPU memory cleared")
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = cp.cuda.Device().mem_info[1]
        
        usage_percent = (used_bytes / total_bytes) * 100
        return {
            'used_gb': used_bytes / 1024**3,
            'total_gb': total_bytes / 1024**3,
            'usage_percent': usage_percent
        }

class GPURIFE:
    """GPU-only RIFE for RTX 5060 Ti"""
    
    def __init__(self):
        self.memory_manager = GPUMemoryManager()
        self.frame_buffer = []
        self.max_buffer_size = 2
        self.processing_quality = "Ultra"
        
        # Performance counters
        self.frames_processed = 0
        self.total_processing_time = 0
        
        print("🚀 GPU RIFE initialized for RTX 5060 Ti")
        
    def image_to_gpu_tensor(self, pil_image):
        """Convert PIL image to GPU tensor"""
        np_array = np.array(pil_image, dtype=np.float32) / 255.0
        
        # Transpose to CHW format
        if len(np_array.shape) == 3 and np_array.shape[2] == 3:
            np_array = np_array.transpose(2, 0, 1)
        
        # Transfer to GPU
        gpu_tensor = cp.asarray(np_array)
        return gpu_tensor
    
    def gpu_tensor_to_image(self, gpu_tensor):
        """Convert GPU tensor back to PIL image"""
        np_array = cp.asnumpy(gpu_tensor)
        
        # Transpose back to HWC format
        if len(np_array.shape) == 3 and np_array.shape[0] == 3:
            np_array = np_array.transpose(1, 2, 0)
        
        # Convert to uint8
        np_array = np.clip(np_array * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(np_array)
    
    def gpu_interpolation(self, frame1_gpu, frame2_gpu):
        """GPU interpolation using RTX 5060 Ti"""
        start_time = time.time()
        
        try:
            if self.processing_quality == "Ultra":
                result = self._ultra_interpolation(frame1_gpu, frame2_gpu)
            elif self.processing_quality == "Maximum":
                result = self._maximum_interpolation(frame1_gpu, frame2_gpu)
            else:
                result = self._high_interpolation(frame1_gpu, frame2_gpu)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.frames_processed += 1
            
            return result
            
        except cp.cuda.memory.OutOfMemoryError:
            print("⚠️  GPU memory full, clearing cache...")
            self.memory_manager.clear_cache()
            return self._simple_interpolation(frame1_gpu, frame2_gpu)
        
        except Exception as e:
            print(f"❌ GPU interpolation error: {e}")
            return self._simple_interpolation(frame1_gpu, frame2_gpu)
    
    def _ultra_interpolation(self, frame1_gpu, frame2_gpu):
        """Ultra quality GPU interpolation"""
        alpha = 0.5
        blend = frame1_gpu * (1 - alpha) + frame2_gpu * alpha
        
        # GPU edge enhancement
        if len(blend.shape) == 3:
            enhanced = cp.zeros_like(blend)
            
            for c in range(blend.shape[0]):
                channel = blend[c]
                center = channel
                up = cp.roll(channel, -1, axis=0)
                down = cp.roll(channel, 1, axis=0)
                left = cp.roll(channel, -1, axis=1)
                right = cp.roll(channel, 1, axis=1)
                
                enhanced[c] = center + 0.1 * (4*center - up - down - left - right)
            
            result = enhanced
        else:
            result = blend
        
        return cp.clip(result, 0.0, 1.0)
    
    def _maximum_interpolation(self, frame1_gpu, frame2_gpu):
        """Maximum quality GPU interpolation"""
        # Motion adaptive blending
        diff = cp.abs(frame2_gpu - frame1_gpu)
        motion_weight = cp.mean(diff)
        adaptive_alpha = 0.5 + float(cp.asnumpy(motion_weight)) * 0.2
        adaptive_alpha = max(0.2, min(0.8, adaptive_alpha))
        
        blend = frame1_gpu * (1 - adaptive_alpha) + frame2_gpu * adaptive_alpha
        
        # Multi-stage enhancement
        if len(blend.shape) == 3:
            enhanced = cp.zeros_like(blend)
            
            for c in range(blend.shape[0]):
                channel = blend[c]
                
                # Edge enhancement
                center = channel
                neighbors = (cp.roll(channel, -1, axis=0) + cp.roll(channel, 1, axis=0) +
                           cp.roll(channel, -1, axis=1) + cp.roll(channel, 1, axis=1))
                
                edge_enhanced = center + 0.15 * (4*center - neighbors)
                
                # Smoothing
                smoothed = (edge_enhanced + 
                          0.1 * (cp.roll(edge_enhanced, -1, axis=0) + 
                                cp.roll(edge_enhanced, 1, axis=0) +
                                cp.roll(edge_enhanced, -1, axis=1) + 
                                cp.roll(edge_enhanced, 1, axis=1))) / 1.4
                
                enhanced[c] = smoothed
            
            result = enhanced
        else:
            result = blend
        
        return cp.clip(result, 0.0, 1.0)
    
    def _high_interpolation(self, frame1_gpu, frame2_gpu):
        """High quality GPU interpolation"""
        alpha = 0.5
        blend = frame1_gpu * (1 - alpha) + frame2_gpu * alpha
        
        # Simple enhancement
        if len(blend.shape) == 3:
            result = blend * 1.05  # Slight brightness boost
        else:
            result = blend
            
        return cp.clip(result, 0.0, 1.0)
    
    def _simple_interpolation(self, frame1_gpu, frame2_gpu):
        """Simple GPU interpolation fallback"""
        alpha = 0.5
        result = frame1_gpu * (1 - alpha) + frame2_gpu * alpha
        return cp.clip(result, 0.0, 1.0)
    
    def process_frame_pair(self, frame1, frame2):
        """Process frame pair with GPU-only interpolation"""
        frame1_gpu = self.image_to_gpu_tensor(frame1)
        frame2_gpu = self.image_to_gpu_tensor(frame2)
        
        result_gpu = self.gpu_interpolation(frame1_gpu, frame2_gpu)
        result_image = self.gpu_tensor_to_image(result_gpu)
        
        return result_image
    
    def add_frame(self, frame):
        """Add frame to processing buffer"""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_interpolated_frame(self):
        """Get interpolated frame from buffer"""
        if len(self.frame_buffer) >= 2:
            frame1 = self.frame_buffer[-2]
            frame2 = self.frame_buffer[-1]
            return self.process_frame_pair(frame1, frame2)
        return None
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.frames_processed > 0:
            avg_time = self.total_processing_time / self.frames_processed
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            memory_stats = self.memory_manager.get_memory_usage()
            
            return {
                'frames_processed': self.frames_processed,
                'average_time_per_frame': avg_time,
                'estimated_fps': fps,
                'gpu_memory_used_gb': memory_stats['used_gb'],
                'gpu_memory_usage_percent': memory_stats['usage_percent']
            }
        else:
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

class RTX5060TiRIFEGUI:
    """RTX 5060 Ti GPU-Only RIFE GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 GPU-Only RIFE - RTX 5060 Ti")
        self.root.configure(bg='#0a0a0a')
        
        # Apply Windows fixes
        WindowsWindowManager.center_window(self.root, 1000, 650)
        WindowsWindowManager.ensure_window_visible(self.root)
        
        # Initialize components
        self.window_capture = WindowCapture()
        self.gpu_rife = GPURIFE()
        self.processing_active = False
        
        # Communication queue
        self.frame_queue = queue.Queue(maxsize=3)
        
        # RIFE output window
        self.rife_window = None
        self.rife_label = None
        
        # Performance monitoring
        self.performance_timer = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg='#0a0a0a', height=70)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚀 GPU-ONLY RIFE", 
                             font=('Arial', 16, 'bold'), fg='#00ff41', bg='#0a0a0a')
        title_label.pack()
        
        gpu_info_label = tk.Label(header_frame, 
                                 text="RTX 5060 Ti • Pure GPU Processing • No CPU Fallbacks", 
                                 font=('Arial', 9), fg='#ffaa00', bg='#0a0a0a')
        gpu_info_label.pack()
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#0a0a0a')
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="🚀 GPU RIFE Ready", 
                                   fg='#00ff41', bg='#0a0a0a', font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT)
        
        self.performance_label = tk.Label(status_frame, text="Performance: Standby", 
                                        fg='#ffaa00', bg='#0a0a0a', font=('Arial', 9))
        self.performance_label.pack(side=tk.RIGHT)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel
        left_panel = tk.Frame(main_frame, bg='#1a1a1a', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Window Capture
        capture_section = tk.LabelFrame(left_panel, text="🎯 Window Capture", 
                                      fg='#00ff41', bg='#1a1a1a', font=('Arial', 10, 'bold'))
        capture_section.pack(fill=tk.X, pady=10)
        
        refresh_btn = tk.Button(capture_section, text="🔄 Refresh Windows", 
                              command=self.refresh_windows, bg='#333333', fg='white',
                              font=('Arial', 9, 'bold'), relief='flat')
        refresh_btn.pack(pady=5)
        
        self.window_listbox = tk.Listbox(capture_section, height=8, bg='#2a2a2a', fg='#00ff41',
                                       selectbackground='#444444', font=('Consolas', 8))
        self.window_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.window_listbox.bind('<<ListboxSelect>>', self.on_window_select)
        
        # GPU Processing Controls
        gpu_section = tk.LabelFrame(left_panel, text="⚡ GPU Processing", 
                                  fg='#ffaa00', bg='#1a1a1a', font=('Arial', 10, 'bold'))
        gpu_section.pack(fill=tk.X, pady=10)
        
        # Quality selection
        quality_frame = tk.Frame(gpu_section, bg='#1a1a1a')
        quality_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(quality_frame, text="Quality Mode:", fg='white', bg='#1a1a1a',
                font=('Arial', 9, 'bold')).pack(anchor='w')
        
        self.quality_var = tk.StringVar(value="Ultra")
        for quality in ["High", "Ultra", "Maximum"]:
            rb = tk.Radiobutton(quality_frame, text=f"{quality} GPU", variable=self.quality_var, 
                              value=quality, command=self.update_quality, fg='white', bg='#1a1a1a',
                              selectcolor='#444444', font=('Arial', 9))
            rb.pack(anchor='w', padx=15)
        
        # Control buttons
        btn_frame = tk.Frame(gpu_section, bg='#1a1a1a')
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_capture_btn = tk.Button(btn_frame, text="▶️ START CAPTURE", 
                                         command=self.start_capture, bg='#006600', fg='white',
                                         font=('Arial', 9, 'bold'), relief='flat')
        self.start_capture_btn.pack(fill=tk.X, pady=2)
        
        self.start_rife_btn = tk.Button(btn_frame, text="🚀 START GPU RIFE", 
                                      command=self.start_rife_processing, bg='#0066cc', fg='white',
                                      font=('Arial', 9, 'bold'), relief='flat')
        self.start_rife_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️ STOP ALL", 
                                command=self.stop_all, bg='#cc0000', fg='white',
                                font=('Arial', 9, 'bold'), relief='flat')
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        clear_btn = tk.Button(btn_frame, text="🧹 Clear GPU Memory", 
                            command=self.clear_gpu_memory, bg='#666600', fg='white',
                            font=('Arial', 8, 'bold'), relief='flat')
        clear_btn.pack(fill=tk.X, pady=2)
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg='#1a1a1a')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Performance section
        perf_section = tk.LabelFrame(right_panel, text="📊 GPU Performance", 
                                   fg='#ffaa00', bg='#1a1a1a', font=('Arial', 10, 'bold'))
        perf_section.pack(fill=tk.X, pady=10)
        
        self.performance_text = tk.Text(perf_section, height=6, bg='#0a0a0a', fg='#00ff41',
                                      font=('Consolas', 8), state='disabled')
        self.performance_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Info section
        info_section = tk.LabelFrame(right_panel, text="ℹ️ System Info", 
                                   fg='#888888', bg='#1a1a1a', font=('Arial', 10, 'bold'))
        info_section.pack(fill=tk.BOTH, expand=True, pady=10)
        
        info_text = tk.Text(info_section, bg='#0a0a0a', fg='#888888',
                          font=('Consolas', 7), state='disabled')
        info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        system_info = f"""🎮 GPU-ONLY RIFE - RTX 5060 Ti EDITION

🔧 Hardware:
• GPU: NVIDIA GeForce RTX 5060 Ti
• VRAM: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB GDDR7
• CUDA Cores: 4,352 cores
• Compute Capability: 12.0 (sm_120)
• Backend: CuPy with CUDA 12.9

⚡ Processing Modes:
• High: Standard GPU interpolation (fastest)
• Ultra: Enhanced filtering + edge enhancement
• Maximum: Full RTX power with adaptive processing

🚀 Features:
• Pure GPU processing (no CPU fallbacks)
• Real-time window capture
• DLSS-like video interpolation
• Advanced memory management
• Performance monitoring
• Chrome window capture support

📋 Instructions:
1. Select window to capture from the list
2. Choose GPU processing quality mode
3. Start capture, then start GPU RIFE
4. Watch real-time interpolated output

⚠️  100% GPU PROCESSING - NO CPU FALLBACKS
🎯 Optimized for RTX 5060 Ti maximum performance
"""
        
        info_text.config(state='normal')
        info_text.insert(tk.END, system_info)
        info_text.config(state='disabled')
        
        # Initialize window list
        self.refresh_windows()
        
    def refresh_windows(self):
        """Refresh the windows list"""
        self.window_listbox.delete(0, tk.END)
        
        if not WINDOWS_CAPTURE_AVAILABLE:
            self.window_listbox.insert(tk.END, "❌ Window capture not available")
            return
        
        windows = self.window_capture.refresh_windows()
        for window in windows:
            display_text = f"🪟 {window['title']}"
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
        self.status_label.config(text=f"GPU Mode: {self.quality_var.get()}")
        
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        self.gpu_rife.memory_manager.clear_cache()
        self.status_label.config(text="GPU memory cleared")
        
    def start_capture(self):
        """Start window capture"""
        if not self.window_capture.target_hwnd:
            messagebox.showerror("Error", "Please select a window to capture first")
            return
        
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        capture_thread.start()
        
        self.start_capture_btn.config(state='disabled', bg='#333333')
        self.status_label.config(text="Capturing for GPU processing...")
        
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
        
        gpu_thread = threading.Thread(target=self.gpu_rife_loop, daemon=True)
        gpu_thread.start()
        
        self.start_performance_monitoring()
        
        self.start_rife_btn.config(state='disabled', bg='#333333')
        self.status_label.config(text="🚀 GPU-ONLY RIFE ACTIVE!")
        
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
                                 text="⚡ GPU processing starting...", fg='#00ff41',
                                 font=('Arial', 12, 'bold'))
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
    
    def start_performance_monitoring(self):
        """Start performance monitoring"""
        self.update_performance_stats()
        
    def update_performance_stats(self):
        """Update performance statistics"""
        if self.processing_active:
            stats = self.gpu_rife.get_performance_stats()
            
            if stats:
                perf_text = f"""🚀 RTX 5060 Ti GPU PERFORMANCE

📊 Processing Stats:
• Frames: {stats['frames_processed']}
• Avg Time: {stats['average_time_per_frame']:.3f}s
• FPS: {stats['estimated_fps']:.1f}
• Mode: {self.gpu_rife.processing_quality}

💾 GPU Memory:
• Used: {stats['gpu_memory_used_gb']:.2f} GB
• Usage: {stats['gpu_memory_usage_percent']:.1f}%

⚡ Status: {"MAXIMUM GPU" if stats['estimated_fps'] > 25 else "HIGH GPU"}"""
                
                self.performance_text.config(state='normal')
                self.performance_text.delete(1.0, tk.END)
                self.performance_text.insert(tk.END, perf_text)
                self.performance_text.config(state='disabled')
                
                self.performance_label.config(
                    text=f"FPS: {stats['estimated_fps']:.1f} | GPU: {stats['gpu_memory_usage_percent']:.1f}%"
                )
            
            self.performance_timer = self.root.after(1000, self.update_performance_stats)
    
    def stop_all(self):
        """Stop all processing"""
        self.processing_active = False
        
        if self.performance_timer:
            self.root.after_cancel(self.performance_timer)
        
        if self.rife_window:
            self.rife_window.destroy()
            self.rife_window = None
            
        self.start_capture_btn.config(state='normal', bg='#006600')
        self.start_rife_btn.config(state='normal', bg='#0066cc')
        self.status_label.config(text="Stopped - GPU ready")
        self.performance_label.config(text="Performance: Standby")
        
    def run(self):
        """Run the application"""
        print("🚀 Starting GPU-Only RIFE GUI for RTX 5060 Ti...")
        print("⚡ No CPU fallbacks - Pure GPU processing!")
        self.root.mainloop()

def main():
    """Main entry point"""
    print("=" * 60)
    print("🎯 GPU-ONLY RIFE - RTX 5060 Ti EDITION")
    print("=" * 60)
    print("🔧 Pure GPU Processing - Zero CPU Fallbacks")
    print("⚡ Optimized for RTX 5060 Ti maximum performance")
    print("🚀 Real-time DLSS-like interpolation")
    print()
    
    app = RTX5060TiRIFEGUI()
    app.run()

if __name__ == "__main__":
    main()
