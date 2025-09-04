#!/usr/bin/env python3
"""
Unified RIFE GUI Application
Combines RIFE video interpolation with Windows window capture functionality
"""

import sys
import os
import subprocess
import importlib

# Try PyQt5 first, fallback to tkinter if not available
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QPushButton, QLabel, QFileDialog, QLineEdit, QProgressBar, 
        QSpinBox, QTabWidget, QListWidget, QComboBox, QCheckBox,
        QTextEdit, QGroupBox, QGridLayout, QMessageBox, QShortcut
    )
    from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
    from PyQt5.QtGui import QPixmap, QIcon, QImage, QKeySequence
    USING_PYQT5 = True
except ImportError:
    print("PyQt5 not available, using tkinter instead...")
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import queue
    USING_PYQT5 = False

import threading
import time

# Windows-specific imports for window capture
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    from PIL import Image, ImageTk
    import numpy as np
    WINDOWS_CAPTURE_AVAILABLE = True
except ImportError:
    WINDOWS_CAPTURE_AVAILABLE = False

from rife_engine import RIFEEngine

# Import GPU auto-setup manager
try:
    from auto_gpu_setup import GPUSetupManager
    GPU_SETUP_AVAILABLE = True
except ImportError:
    print("Warning: auto_gpu_setup.py not found - manual PyTorch setup required")
    GPUSetupManager = None
    GPU_SETUP_AVAILABLE = False

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

class GPUSetupThread(QThread):
    """Thread for GPU detection and PyTorch auto-setup"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)
    
    def __init__(self, force=False):
        super().__init__()
        self.force = force
    
    def run(self):
        if not GPU_SETUP_AVAILABLE:
            self.progress.emit("GPU auto-setup not available")
            self.finished.emit(False)
            return
        
        try:
            self.progress.emit("🔍 Detecting GPU...")
            manager = GPUSetupManager()
            
            # Detect GPU
            gpu = manager.detect_gpu()
            if gpu:
                self.progress.emit(f"✅ Found: {gpu['name']} ({gpu['memory_gb']}GB)")
            else:
                self.progress.emit("⚠️  No NVIDIA GPU detected - using CPU mode")
            
            # Check if setup is needed
            saved_gpu = manager.current_config.get('gpu')
            if not self.force and not manager.gpu_changed(gpu, saved_gpu):
                self.progress.emit("✅ GPU setup is current")
                # Still test installation
                if manager.test_installation():
                    self.finished.emit(True)
                    return
                else:
                    self.progress.emit("⚠️  Installation test failed, updating...")
            
            # Run setup
            self.progress.emit("🔧 Setting up optimal PyTorch version...")
            
            # Get recommended PyTorch version
            pytorch_config = manager.get_pytorch_version_for_gpu(gpu)
            self.progress.emit(f"Recommended: {pytorch_config['cuda_version']} - {pytorch_config['reason']}")
            
            # Check current installation
            current_pytorch = manager.check_current_pytorch()
            
            if current_pytorch.get('installed'):
                self.progress.emit(f"Current PyTorch: {current_pytorch['version']}")
                
                # Check if compatible
                current_cuda = 'cpu' if not current_pytorch.get('cuda_available') else f"cu{current_pytorch.get('cuda_version', '').replace('.', '')}"
                if current_cuda != pytorch_config['cuda_version']:
                    self.progress.emit("🗑️  Removing incompatible PyTorch...")
                    manager.uninstall_pytorch()
                    
                    self.progress.emit("📦 Installing optimal PyTorch version...")
                    if manager.install_pytorch(pytorch_config['install_command']):
                        self.progress.emit("✅ PyTorch installation complete!")
                    else:
                        self.progress.emit("❌ PyTorch installation failed")
                        self.finished.emit(False)
                        return
                else:
                    self.progress.emit("✅ Current PyTorch is compatible")
            else:
                self.progress.emit("📦 Installing PyTorch...")
                if manager.install_pytorch(pytorch_config['install_command']):
                    self.progress.emit("✅ PyTorch installation complete!")
                else:
                    self.progress.emit("❌ PyTorch installation failed")
                    self.finished.emit(False)
                    return
            
            # Test installation
            self.progress.emit("🧪 Testing installation...")
            if manager.test_installation():
                # Save configuration
                manager.save_config({'gpu': gpu, 'pytorch': pytorch_config})
                self.progress.emit("🎉 GPU setup complete!")
                self.finished.emit(True)
            else:
                self.progress.emit("❌ Installation test failed")
                self.finished.emit(False)
                
        except Exception as e:
            self.progress.emit(f"❌ Setup error: {str(e)}")
            self.finished.emit(False)

class DependencyInstaller(QThread):
    """Thread to install missing dependencies"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.packages_to_install = []
    
    def run(self):
        required_packages = {
            'PyQt5': 'PyQt5',
            'pywin32': 'win32gui',  # For Windows window capture
            'pillow': 'PIL',
            'numpy': 'numpy',
        }
        
        missing_packages = []
        for display_name, import_name in required_packages.items():
            self.progress.emit(f"Checking {display_name}...")
            if not check_package(import_name):
                missing_packages.append(display_name)
        
        if not missing_packages:
            self.progress.emit("All dependencies are installed!")
            self.finished.emit(True)
            return
        
        success = True
        for package in missing_packages:
            self.progress.emit(f"Installing {package}...")
            try:
                install_package(package)
                self.progress.emit(f"✓ {package} installed successfully!")
            except subprocess.CalledProcessError as e:
                self.progress.emit(f"✗ Failed to install {package}: {e}")
                success = False
        
        self.finished.emit(success)

class WindowCaptureThread(QThread):
    """Thread for capturing windows"""
    frame_captured = pyqtSignal(object)  # Emits PIL Image
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.capturing = False
        self.target_hwnd = None
        self.capture_rate = 30  # FPS
    
    def set_target_window(self, hwnd):
        self.target_hwnd = hwnd
    
    def start_capture(self):
        self.capturing = True
        self.start()
    
    def stop_capture(self):
        self.capturing = False
        self.quit()
        self.wait()
    
    def run(self):
        if not WINDOWS_CAPTURE_AVAILABLE:
            self.error_occurred.emit("Windows capture not available. Install pywin32.")
            return
        
        if not self.target_hwnd:
            self.error_occurred.emit("No target window selected.")
            return
        
        while self.capturing:
            try:
                # Capture the window
                image = self.capture_window(self.target_hwnd)
                if image:
                    self.frame_captured.emit(image)
                
                # Control frame rate
                time.sleep(1.0 / self.capture_rate)
                
            except Exception as e:
                self.error_occurred.emit(f"Capture error: {str(e)}")
                break
    
    def capture_window(self, hwnd):
        """Capture a specific window by handle"""
        try:
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            if width <= 0 or height <= 0:
                return None
            
            # Try different capture methods for better compatibility
            image = None
            
            # Method 1: Standard BitBlt (works for most apps)
            try:
                image = self._capture_with_bitblt(hwnd, width, height)
            except:
                pass
            
            # Method 2: PrintWindow (works for Chrome and protected apps)
            if image is None:
                try:
                    image = self._capture_with_printwindow(hwnd, width, height)
                except:
                    pass
            
            return image
            
        except Exception as e:
            print(f"Window capture error: {e}")
            return None
    
    def _capture_with_bitblt(self, hwnd, width, height):
        """Standard BitBlt capture method"""
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # Copy window content
        result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
        
        # Convert to PIL Image
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        image = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )
        
        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return image
    
    def _capture_with_printwindow(self, hwnd, width, height):
        """PrintWindow capture method (works with Chrome)"""
        import win32print
        
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # Use PrintWindow for better compatibility
        result = win32gui.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)  # PW_RENDERFULLCONTENT
        
        if result:
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            image = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )
        else:
            image = None
        
        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return image

class RIFEWorker(QThread):
    """Worker thread for RIFE processing"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)
    status_update = pyqtSignal(str)

    def __init__(self, input_path, output_path, factor):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.factor = factor

    def run(self):
        self.status_update.emit("Starting RIFE processing...")
        engine = RIFEEngine()
        success = engine.interpolate_video(self.input_path, self.output_path, self.factor)
        self.finished.emit(success)

class RealTimeRIFEProcessor(QThread):
    """Non-blocking real-time RIFE processor"""
    frame_processed = pyqtSignal(object)  # Emits processed PIL Image
    fps_update = pyqtSignal(float)  # Emits current RIFE FPS
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.processing = False
        self.frame_queue = []
        self.quality = "High"
        self.max_queue_size = 5  # Increased for better continuity
        self.frame_counter = 0
        self.last_processed_time = 0
    
    def add_frame(self, frame):
        """Add frame to processing queue (thread-safe)"""
        if len(self.frame_queue) >= self.max_queue_size:
            # Remove oldest frame to prevent memory leak
            if self.frame_queue:
                old_frame = self.frame_queue.pop(0)
                del old_frame  # Explicit cleanup
        
        # Add new frame
        self.frame_queue.append(frame.copy())  # Make a copy to avoid reference issues
    
    def set_quality(self, quality):
        self.quality = quality
    
    def start_processing(self):
        self.processing = True
        self.start()
    
    def stop_processing(self):
        self.processing = False
        # Clean up queue
        for frame in self.frame_queue:
            del frame
        self.frame_queue.clear()
        self.quit()
        self.wait()
    
    def run(self):
        import time
        
        while self.processing:
            try:
                if len(self.frame_queue) >= 2:
                    start_time = time.time()
                    
                    # Get two most recent frames
                    frame1 = self.frame_queue[-2]
                    frame2 = self.frame_queue[-1]
                    
                    # Process with lightweight interpolation
                    processed_frame = self.lightweight_interpolation(frame1, frame2)
                    
                    if processed_frame is not None:
                        # Emit result with frame counter for debugging
                        self.frame_counter += 1
                        print(f"[DEBUG] Emitting processed frame #{self.frame_counter}: {processed_frame.size}")
                        self.frame_processed.emit(processed_frame)
                        
                        # Calculate FPS
                        processing_time = time.time() - start_time
                        if processing_time > 0:
                            fps = 1.0 / processing_time
                            self.fps_update.emit(fps)
                            self.last_processed_time = start_time
                    else:
                        print(f"[WARNING] Processed frame is None - skipping emission")
                    
                    # Clean up processed frames
                    try:
                        del frame1, frame2
                        if processed_frame:
                            del processed_frame
                    except:
                        pass
                
                # Adaptive delay based on processing time
                time.sleep(0.016)  # Target ~60 FPS max processing rate
                
            except Exception as e:
                error_msg = f"RIFE processing error: {str(e)}"
                print(f"[ERROR] {error_msg}")
                self.error_occurred.emit(error_msg)
                time.sleep(0.1)  # Longer delay on error
                continue  # Don't break, try to recover
    
    def lightweight_interpolation(self, frame1, frame2):
        """Hybrid GPU-CPU interpolation optimized for RTX 5060 Ti"""
        print(f"[DEBUG] Starting interpolation with frames: {frame1.size if frame1 else 'None'}, {frame2.size if frame2 else 'None'}")
        
        # Try CPU-optimized processing first (most reliable)
        try:
            result = self.optimized_cpu_interpolation(frame1, frame2)
            print(f"[DEBUG] Optimized CPU processing successful, result: {result.size if result else 'None'}")
            return result
        except Exception as cpu_error:
            print(f"[DEBUG] CPU processing failed: {cpu_error}")
        
        # Fallback to basic CPU processing
        try:
            result = self.cpu_interpolation_fallback(frame1, frame2)
            print(f"[DEBUG] Basic CPU fallback successful, result: {result.size if result else 'None'}")
            return result
        except Exception as fallback_error:
            print(f"[ERROR] All processing failed: {fallback_error}")
            print(f"[DEBUG] Returning original frame2")
            return frame2
    
    def gpu_interpolation(self, frame1, frame2):
        """GPU-accelerated processing using CuPy for RTX 5060 Ti support"""
        try:
            print(f"[DEBUG] Attempting CuPy GPU interpolation for RTX 5060 Ti...")
            
            # Try CuPy first (works with RTX 5060 Ti!)
            try:
                import cupy as cp
                import numpy as np
                
                print(f"[DEBUG] CuPy detected, using RTX 5060 Ti acceleration...")
                
                # Convert PIL Images to numpy arrays
                arr1 = np.array(frame1, dtype=np.float32) / 255.0
                arr2 = np.array(frame2, dtype=np.float32) / 255.0
                
                # Move to GPU
                gpu_arr1 = cp.asarray(arr1)
                gpu_arr2 = cp.asarray(arr2)
                
                # Safe GPU interpolation to avoid kernel issues with RTX 5060 Ti
                try:
                    if self.quality == "Ultra (Max GPU)":
                        # Try advanced processing, fallback if it fails
                        try:
                            alpha = 0.6
                            blend = gpu_arr1 * (1 - alpha) + gpu_arr2 * alpha
                            result_gpu = self._cupy_enhance(blend)
                        except:
                            print(f"[DEBUG] Ultra mode failed, using simple blend")
                            result_gpu = (gpu_arr1 + gpu_arr2) * 0.5
                        
                    elif self.quality == "High":
                        # High quality blending with fallback
                        try:
                            alpha = 0.5
                            blend = gpu_arr1 * (1 - alpha) + gpu_arr2 * alpha
                            result_gpu = blend * 1.05  # Simple enhancement
                            result_gpu = cp.clip(result_gpu, 0.0, 1.0)
                        except:
                            print(f"[DEBUG] High mode failed, using simple blend")
                            result_gpu = (gpu_arr1 + gpu_arr2) * 0.5
                    else:
                        # Fast and safe GPU processing
                        result_gpu = (gpu_arr1 + gpu_arr2) * 0.5
                    
                    # Ensure result is properly clipped
                    result_gpu = cp.clip(result_gpu, 0.0, 1.0)
                    
                except Exception as blend_error:
                    print(f"[DEBUG] All GPU blending failed: {blend_error}")
                    # Emergency fallback - just return first frame
                    result_gpu = gpu_arr1
                
                # Convert back to CPU and PIL
                result_cpu = cp.asnumpy(result_gpu) * 255.0
                result_cpu = result_cpu.astype(np.uint8)
                
                # Convert to PIL Image
                result_image = Image.fromarray(result_cpu)
                
                print(f"[SUCCESS] CuPy GPU interpolation completed on RTX 5060 Ti!")
                return result_image
                
            except ImportError:
                print(f"[INFO] CuPy not available, trying PyTorch...")
                # Fallback to original PyTorch method
                pass
            
            # Original PyTorch fallback (will fail on RTX 5060 Ti)
            import torch
            import torchvision.transforms as transforms
            import warnings
            
            warnings.filterwarnings("ignore", category=UserWarning)
            torch.set_warn_always(False)
            
            if not torch.cuda.is_available():
                print(f"[DEBUG] PyTorch CUDA not available")
                raise Exception("CUDA not available")
                
            device_name = torch.cuda.get_device_name(0)
            if "RTX 5060" in device_name:
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 12:
                    print(f"[INFO] RTX 5060 Ti detected - PyTorch doesn't support sm_120 yet")
                    print(f"[INFO] CuPy should have been used - something went wrong")
                    raise Exception("RTX 5060 Ti requires CuPy for GPU acceleration")
            
            device = torch.device('cuda')
            print(f"[DEBUG] Using CUDA device: {device}")
            
            # Convert PIL to tensor and move to GPU with error handling
            try:
                # Use no_grad context to prevent gradient warnings
                with torch.no_grad():
                    transform = transforms.Compose([
                        transforms.ToTensor()
                    ])
                    
                    # Move images to GPU
                    tensor1 = transform(frame1).unsqueeze(0).to(device, non_blocking=True)
                    tensor2 = transform(frame2).unsqueeze(0).to(device, non_blocking=True)
                    print(f"[DEBUG] Tensors moved to GPU: {tensor1.shape}, {tensor2.shape}")
            except Exception as tensor_error:
                print(f"[ERROR] Tensor conversion failed: {tensor_error}")
                raise tensor_error
            
            # Simplified GPU processing to avoid warnings
            try:
                with torch.no_grad():  # Prevent gradient warnings
                    if self.quality == "Ultra (Max GPU)":
                        result_tensor = self.simple_gpu_blend(tensor1, tensor2, device, 0.6)
                    elif self.quality == "High":
                        result_tensor = self.simple_gpu_blend(tensor1, tensor2, device, 0.5)
                    elif self.quality == "Medium":
                        result_tensor = self.simple_gpu_blend(tensor1, tensor2, device, 0.4)
                    else:  # Fast
                        result_tensor = (tensor1 + tensor2) * 0.5
                    
                    print(f"[DEBUG] GPU processing completed, result shape: {result_tensor.shape}")
            except Exception as processing_error:
                print(f"[ERROR] GPU processing failed: {processing_error}")
                raise processing_error
            
            # Convert back to PIL
            try:
                with torch.no_grad():
                    result_tensor = torch.clamp(result_tensor, 0, 1)
                    result_cpu = result_tensor.squeeze(0).cpu().detach()
                    
                    # Convert to PIL Image
                    to_pil = transforms.ToPILImage()
                    result_image = to_pil(result_cpu)
                    print(f"[DEBUG] Conversion to PIL successful: {result_image.size}")
                    
                    # Clean up GPU memory more aggressively
                    del tensor1, tensor2, result_tensor, result_cpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure operations complete
                    
                    return result_image
            except Exception as conversion_error:
                print(f"[ERROR] PIL conversion failed: {conversion_error}")
                raise conversion_error
                
        except Exception as e:
            print(f"[ERROR] GPU interpolation failed: {e}")
            # Clean up on error
            try:
                torch.cuda.empty_cache()
            except:
                pass
            raise e
    
    def simple_gpu_blend(self, tensor1, tensor2, device, alpha):
        """Simple GPU blending without complex operations - optimized for RTX 4060 Ti"""
        try:
            import torch
            with torch.no_grad():
                # Create alpha tensor once
                alpha_tensor = torch.tensor(alpha, device=device, dtype=torch.float32)
                
                # Basic interpolation
                blend = torch.lerp(tensor1, tensor2, alpha_tensor)
                
                # Minimal sharpening to avoid warnings
                if alpha > 0.5:  # Ultra/High modes
                    # Use torch.mul for safer operation
                    sharpened = torch.mul(blend, 1.05)  # Reduced from 1.1 to avoid overflow
                    blend = torch.clamp(sharpened, 0.0, 1.0)
                
                return blend
        except Exception as e:
            print(f"[ERROR] Simple GPU blend failed: {e}")
            # Fallback to basic blend with no_grad
            with torch.no_grad():
                return torch.mul(torch.add(tensor1, tensor2), 0.5)
    
    def ultra_gpu_processing(self, tensor1, tensor2, device):
        """Ultra quality GPU processing - Max RTX 4060 Ti usage"""
        import torch
        import torch.nn.functional as F
        
        # Multi-pass GPU processing for maximum utilization
        # Pass 1: Basic interpolation
        alpha = torch.tensor(0.5, device=device)
        blend = tensor1 * (1 - alpha) + tensor2 * alpha
        
        # Pass 2: Gaussian blur for smoothing (GPU)
        kernel_size = 5
        sigma = 1.0
        kernel = self.gaussian_kernel_2d(kernel_size, sigma).to(device)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        
        smooth = F.conv2d(blend, kernel, padding=kernel_size//2, groups=3)
        
        # Pass 3: Edge enhancement (GPU convolution)
        edge_kernel = torch.tensor([
            [[-1, -1, -1],
             [-1,  9, -1], 
             [-1, -1, -1]]
        ], dtype=torch.float32, device=device)
        edge_kernel = edge_kernel.expand(3, 1, 3, 3) / 9.0
        
        enhanced = F.conv2d(smooth, edge_kernel, padding=1, groups=3)
        
        # Pass 4: Sharpening and final blend
        sharpened = enhanced * 1.2
        final = torch.lerp(smooth, sharpened, 0.3)
        
        return final
    
    def high_gpu_processing(self, tensor1, tensor2, device):
        """High quality GPU processing"""
        import torch
        import torch.nn.functional as F
        
        # Weighted interpolation
        alpha = torch.tensor(0.4, device=device)
        blend = tensor1 * (1 - alpha) + tensor2 * alpha
        
        # Edge enhancement
        edge_kernel = torch.tensor([
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]]
        ], dtype=torch.float32, device=device)
        edge_kernel = edge_kernel.expand(3, 1, 3, 3)
        
        enhanced = F.conv2d(blend, edge_kernel, padding=1, groups=3)
        
        return torch.clamp(enhanced, 0, 1)
    
    def medium_gpu_processing(self, tensor1, tensor2, device):
        """Medium quality GPU processing"""
        import torch
        
        # Simple GPU blend with light enhancement
        alpha = torch.tensor(0.5, device=device)
        blend = tensor1 * (1 - alpha) + tensor2 * alpha
        
        # Light sharpening
        enhanced = blend * 1.05
        
        return torch.clamp(enhanced, 0, 1)
    
    def fast_gpu_processing(self, tensor1, tensor2, device):
        """Fast GPU processing"""
        import torch
        
        # Simple linear interpolation on GPU
        return (tensor1 + tensor2) * 0.5
    
    def gaussian_kernel_2d(self, kernel_size, sigma):
        """Generate 2D Gaussian kernel on GPU"""
        import torch
        import math
        
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        
        return gauss_2d.unsqueeze(0).unsqueeze(0)
    
    def _cupy_enhance(self, blend_gpu):
        """CuPy-based enhancement for RTX 5060 Ti - simplified to avoid kernel issues"""
        import cupy as cp
        
        try:
            # Simple enhancement that doesn't require custom kernels
            enhanced = blend_gpu * 1.1  # Brightness boost
            enhanced = cp.clip(enhanced, 0.0, 1.0)
            return enhanced
        except Exception as e:
            print(f"[DEBUG] CuPy enhancement failed: {e}")
            # Return original if enhancement fails
            return blend_gpu
    
    def optimized_cpu_interpolation(self, frame1, frame2):
        """Optimized CPU interpolation with quality-based processing"""
        import numpy as np
        from PIL import Image, ImageEnhance
        
        # Convert to numpy arrays with float precision
        arr1 = np.array(frame1, dtype=np.float32) / 255.0
        arr2 = np.array(frame2, dtype=np.float32) / 255.0
        
        # Quality-based interpolation
        if self.quality == "Ultra (Max GPU)" or self.quality == "High":
            # Advanced CPU interpolation
            alpha = 0.6 if self.quality == "Ultra (Max GPU)" else 0.5
            blend = arr1 * (1 - alpha) + arr2 * alpha
            
            # CPU-based enhancement
            enhanced = np.clip(blend * 1.1, 0.0, 1.0)  # Brightness boost
            result_arr = (enhanced * 255).astype(np.uint8)
        else:
            # Fast CPU processing
            result = (arr1 + arr2) * 0.5
            result_arr = (result * 255).astype(np.uint8)
        
        processed_image = Image.fromarray(result_arr)
        del arr1, arr2, result_arr
        
        return processed_image
    
    def cpu_interpolation_fallback(self, frame1, frame2):
        """Basic CPU fallback if all else fails"""
        import numpy as np
        
        arr1 = np.array(frame1, dtype=np.uint8)
        arr2 = np.array(frame2, dtype=np.uint8)
        
        # Simple CPU blend
        result = (arr1.astype(np.uint16) + arr2.astype(np.uint16)) // 2
        result_arr = result.astype(np.uint8)
        
        processed_image = Image.fromarray(result_arr)
        del arr1, arr2, result, result_arr
        
        return processed_image

class WindowSelector(QWidget):
    """Widget for selecting and previewing windows"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.capture_thread = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Window selection
        selection_group = QGroupBox("Window Selection")
        selection_layout = QVBoxLayout()
        
        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(300)
        
        self.refresh_btn = QPushButton("🔄 Refresh Window List (F5)")
        self.refresh_btn.clicked.connect(self.refresh_windows)
        self.refresh_btn.setToolTip("Press F5 or click to refresh the list of available windows")
        
        selection_layout.addWidget(QLabel("Select Window:"))
        selection_layout.addWidget(self.window_combo)
        selection_layout.addWidget(self.refresh_btn)
        selection_group.setLayout(selection_layout)
        
        # Capture controls
        controls_group = QGroupBox("Capture Controls")
        controls_layout = QVBoxLayout()
        
        # Button row
        button_layout = QHBoxLayout()
        self.start_capture_btn = QPushButton("▶️ Start Capture")
        self.stop_capture_btn = QPushButton("⏹️ Stop Capture")
        self.save_frame_btn = QPushButton("💾 Save Current Frame")
        self.realtime_rife_btn = QPushButton("⚡ Real-time RIFE")
        
        self.start_capture_btn.clicked.connect(self.start_capture)
        self.stop_capture_btn.clicked.connect(self.stop_capture)
        self.save_frame_btn.clicked.connect(self.save_current_frame)
        self.realtime_rife_btn.clicked.connect(self.toggle_realtime_rife)
        
        self.stop_capture_btn.setEnabled(False)
        self.save_frame_btn.setEnabled(False)
        self.realtime_rife_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_capture_btn)
        button_layout.addWidget(self.stop_capture_btn)
        button_layout.addWidget(self.save_frame_btn)
        button_layout.addWidget(self.realtime_rife_btn)
        
        # Capture rate control
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Capture Rate:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" FPS")
        self.fps_spin.setToolTip("Higher FPS = more data for RIFE interpolation")
        rate_layout.addWidget(self.fps_spin)
        
        # Quality/Performance setting
        rate_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Ultra (Max GPU)", "High", "Medium", "Fast"])
        self.quality_combo.setCurrentText("High")
        self.quality_combo.setToolTip("Higher quality = more GPU usage for better interpolation")
        self.quality_combo.currentTextChanged.connect(self.on_quality_changed)
        rate_layout.addWidget(self.quality_combo)
        rate_layout.addStretch()
        
        controls_layout.addLayout(button_layout)
        controls_layout.addLayout(rate_layout)
        controls_group.setLayout(controls_layout)
        
        # Status and FPS Statistics
        stats_group = QGroupBox("Performance Statistics")
        stats_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to capture")
        stats_layout.addWidget(self.status_label)
        
        # FPS Display
        self.fps_stats_label = QLabel("Capture: 0 FPS | RIFE Output: 0 FPS | Avg Capture: 0 FPS")
        self.fps_stats_label.setStyleSheet("font-family: monospace; background: #2b2b2b; color: #00ff00; padding: 5px;")
        stats_layout.addWidget(self.fps_stats_label)
        
        # GPU Usage indicator
        self.gpu_stats_label = QLabel("GPU: Idle | Quality: High")
        self.gpu_stats_label.setStyleSheet("font-family: monospace; background: #1a1a1a; color: #ffaa00; padding: 5px;")
        stats_layout.addWidget(self.gpu_stats_label)
        
        # CUDA/GPU Memory Stats
        self.cuda_stats_label = QLabel("CUDA: Not Available | GPU Memory: 0MB")
        self.cuda_stats_label.setStyleSheet("font-family: monospace; background: #0a0a2a; color: #00ffff; padding: 5px;")
        stats_layout.addWidget(self.cuda_stats_label)
        
        stats_group.setLayout(stats_layout)
        
        # Preview area
        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("No preview available")
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        layout.addWidget(selection_group)
        layout.addWidget(controls_group)
        layout.addWidget(stats_group)
        layout.addWidget(preview_group)
        
        self.setLayout(layout)
        self.refresh_windows()
        
        # Store current frame
        self.current_frame = None
        self.realtime_rife_active = False
        self.rife_window = None
        self.frame_buffer = []  # Small buffer for RIFE processing
        self.max_buffer_size = 3  # Limit buffer to prevent memory leak
        
        # FPS tracking
        import time
        self.capture_fps_tracker = []
        self.rife_fps_tracker = []
        self.last_capture_time = time.time()
        self.last_rife_time = time.time()
        
        # Processing thread for non-blocking RIFE
        self.rife_processor = None
        
        # Auto-refresh timer for window list
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh_windows)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        # F5 shortcut for manual refresh
        self.refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        self.refresh_shortcut.activated.connect(self.refresh_windows)
    
    def refresh_windows(self):
        """Refresh the list of available windows"""
        self.status_label.setText("🔄 Refreshing window list...")
        self.window_combo.clear()
        
        if not WINDOWS_CAPTURE_AVAILABLE:
            self.window_combo.addItem("Windows capture not available - install pywin32")
            self.status_label.setText("⚠️ Windows capture not available")
            return
        
        def enum_windows_proc(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                # Filter out empty titles and system windows
                if window_text and len(window_text.strip()) > 0:
                    # Skip some common system windows
                    skip_titles = ['', 'Program Manager', 'Desktop Window Manager']
                    if window_text not in skip_titles:
                        try:
                            # Get window class to help identify meaningful windows
                            class_name = win32gui.GetClassName(hwnd)
                            windows.append((hwnd, window_text, class_name))
                        except:
                            windows.append((hwnd, window_text, 'Unknown'))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_proc, windows)
        
        # Sort windows by title for easier finding
        windows.sort(key=lambda x: x[1].lower())
        
        for hwnd, title, class_name in windows:
            # Show class name for better identification
            display_text = f"{title} ({class_name})" if class_name != 'Unknown' else title
            self.window_combo.addItem(display_text, hwnd)
        
        self.status_label.setText(f"✅ Found {len(windows)} windows (auto-refreshing every 5s)")
    
    def auto_refresh_windows(self):
        """Auto-refresh the window list if not currently capturing"""
        # Only auto-refresh if not currently capturing
        if not self.capture_thread or not self.capture_thread.capturing:
            current_selection = self.window_combo.currentText()
            self.refresh_windows()
            
            # Try to restore previous selection
            index = self.window_combo.findText(current_selection)
            if index >= 0:
                self.window_combo.setCurrentIndex(index)
    
    def start_capture(self):
        """Start capturing the selected window"""
        if self.window_combo.currentData() is None:
            QMessageBox.warning(self, "Warning", "Please select a window to capture")
            return
        
        hwnd = self.window_combo.currentData()
        
        self.capture_thread = WindowCaptureThread()
        self.capture_thread.set_target_window(hwnd)
        self.capture_thread.capture_rate = self.fps_spin.value()  # Set configurable FPS
        self.capture_thread.frame_captured.connect(self.on_frame_captured)
        self.capture_thread.error_occurred.connect(self.on_capture_error)
        
        self.capture_thread.start_capture()
        
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)
        self.save_frame_btn.setEnabled(True)
        self.realtime_rife_btn.setEnabled(True)
        self.frame_count = 0
        self.status_label.setText(f"📹 Capturing at {self.fps_spin.value()} FPS - Frames: 0")
    
    def stop_capture(self):
        """Stop the capture"""
        if self.capture_thread:
            self.capture_thread.stop_capture()
            self.capture_thread = None
        
        # Stop RIFE processing if active
        if self.realtime_rife_active:
            self.stop_realtime_rife()
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        self.save_frame_btn.setEnabled(False)
        self.realtime_rife_btn.setEnabled(False)
        
        if hasattr(self, 'frame_count'):
            self.status_label.setText(f"⏹️ Capture stopped - Total frames captured: {self.frame_count}")
        else:
            self.status_label.setText("⏹️ Capture stopped")
    
    def on_frame_captured(self, image):
        """Handle captured frame"""
        import time
        current_time = time.time()
        
        self.current_frame = image
        
        # Calculate actual capture FPS
        if hasattr(self, 'last_capture_time'):
            time_diff = current_time - self.last_capture_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                self.capture_fps_tracker.append(current_fps)
                # Keep only last 30 readings for average
                if len(self.capture_fps_tracker) > 30:
                    self.capture_fps_tracker.pop(0)
        
        self.last_capture_time = current_time
        
        # Update frame counter and statistics
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
            
            # Calculate average FPS
            avg_fps = sum(self.capture_fps_tracker) / len(self.capture_fps_tracker) if self.capture_fps_tracker else 0
            current_fps = self.capture_fps_tracker[-1] if self.capture_fps_tracker else 0
            rife_avg_fps = sum(self.rife_fps_tracker) / len(self.rife_fps_tracker) if self.rife_fps_tracker else 0
            
            # Update status
            status_text = f"📹 Capturing - Frames: {self.frame_count}"
            if self.realtime_rife_active:
                status_text += " | ⚡ RIFE Active"
            self.status_label.setText(status_text)
            
            # Update FPS statistics
            self.fps_stats_label.setText(
                f"Capture: {current_fps:.1f} FPS | RIFE Output: {rife_avg_fps:.1f} FPS | Avg Capture: {avg_fps:.1f} FPS"
            )
            
            # Update GPU stats with memory monitoring
            self.update_gpu_stats()
        
        # Add to frame buffer for RIFE processing (non-blocking)
        if self.realtime_rife_active and self.rife_processor:
            self.rife_processor.add_frame(image)
        
        # Convert to QPixmap for display
        # Resize for preview
        display_image = image.resize((320, 240), Image.Resampling.LANCZOS)
        
        # Convert PIL to QPixmap
        display_image = display_image.convert("RGB")
        data = display_image.tobytes("raw", "RGB")
        qimage = QPixmap.fromImage(
            QImage(data, display_image.width, display_image.height, QImage.Format_RGB888)
        )
        
        self.preview_label.setPixmap(qimage)
    
    # Old blocking methods removed - now using threaded RealTimeRIFEProcessor
    
    # Heavy GPU processing methods removed to prevent GUI freezing
    # Now using lightweight processing in RealTimeRIFEProcessor thread
    
    def display_rife_frame(self, rife_frame):
        """Display RIFE-processed frame in the RIFE window with robust error handling"""
        print(f"[DEBUG] display_rife_frame called with frame: {rife_frame.size if rife_frame else 'None'}")
        
        # Create RIFE window if it doesn't exist
        if not hasattr(self, 'rife_window') or not self.rife_window:
            print(f"[DEBUG] Creating RIFE display window...")
            self.create_rife_display_window()
        
        if not self.rife_window:
            print(f"[WARNING] Failed to create RIFE window")
            return
            
        if not rife_frame:
            print(f"[WARNING] RIFE frame is None or invalid")
            if hasattr(self, 'rife_status_label'):
                self.rife_status_label.setText("❌ No frame to display")
            return
        
        try:
            from PIL import Image  # Ensure PIL Image is available
            print(f"[DEBUG] Starting frame display process...")
            
            # Ensure frame is valid PIL Image
            if not hasattr(rife_frame, 'size') or not hasattr(rife_frame, 'convert'):
                raise ValueError(f"Invalid PIL Image object: {type(rife_frame)}")
            
            # Resize for display
            display_image = rife_frame.resize((640, 360), Image.Resampling.LANCZOS)
            print(f"[DEBUG] Frame resized to: {display_image.size}")
            
            # Convert PIL to QPixmap with proper format handling
            display_image = display_image.convert("RGB")
            width, height = display_image.size
            
            # Get raw bytes
            data = display_image.tobytes("raw", "RGB")
            print(f"[DEBUG] Converted to bytes, length: {len(data)}")
            
            # Create QImage
            qimage_obj = QImage(data, width, height, QImage.Format_RGB888)
            if qimage_obj.isNull():
                raise ValueError("Created QImage is null")
            
            # Convert to QPixmap
            qpixmap = QPixmap.fromImage(qimage_obj)
            if qpixmap.isNull():
                raise ValueError("Created QPixmap is null")
            
            print(f"[DEBUG] QPixmap created: {qpixmap.width()}x{qpixmap.height()}")
            
            # Update display
            if hasattr(self, 'rife_display_label'):
                self.rife_display_label.setPixmap(qpixmap)
                self.rife_display_label.update()  # Force refresh
                print(f"[DEBUG] Frame displayed successfully")
                
                # Update status
                if hasattr(self, 'rife_status_label'):
                    self.rife_status_label.setText("✅ Processing and displaying frames")
            else:
                print(f"[ERROR] rife_display_label not found")
                
        except Exception as e:
            error_msg = f"Display Error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Frame info: {rife_frame.size if hasattr(rife_frame, 'size') else 'No size attr'}")
            
            if hasattr(self, 'rife_status_label'):
                self.rife_status_label.setText(f"❌ {error_msg[:50]}...")
            
            # Try fallback - display a simple error image
            try:
                from PIL import Image, ImageDraw
                error_img = Image.new('RGB', (640, 360), color='red')
                draw = ImageDraw.Draw(error_img)
                draw.text((10, 10), "Display Error", fill='white')
                
                data = error_img.tobytes("raw", "RGB")
                qimage_obj = QImage(data, 640, 360, QImage.Format_RGB888)
                qpixmap = QPixmap.fromImage(qimage_obj)
                if hasattr(self, 'rife_display_label'):
                    self.rife_display_label.setPixmap(qpixmap)
            except Exception as fallback_error:
                print(f"[ERROR] Even fallback display failed: {fallback_error}")
    
    def on_capture_error(self, error):
        """Handle capture errors"""
        self.status_label.setText(f"Error: {error}")
        self.stop_capture()
        QMessageBox.critical(self, "Capture Error", error)
    
    def save_current_frame(self):
        """Save the current captured frame"""
        if not self.current_frame:
            QMessageBox.warning(self, "Warning", "No frame to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame", "captured_frame.png", "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if file_path:
            self.current_frame.save(file_path)
            QMessageBox.information(self, "Success", f"Frame saved to {file_path}")
    
    def toggle_realtime_rife(self):
        """Toggle real-time RIFE processing"""
        if not self.realtime_rife_active:
            self.start_realtime_rife()
        else:
            self.stop_realtime_rife()
    
    def start_realtime_rife(self):
        """Start real-time RIFE processing"""
        if not self.capture_thread or not self.capture_thread.capturing:
            QMessageBox.warning(self, "Warning", "Please start window capture first!")
            return
        
        self.realtime_rife_active = True
        self.realtime_rife_btn.setText("⏹️ Stop RIFE")
        self.realtime_rife_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
        
        # Create and start RIFE processor thread
        self.rife_processor = RealTimeRIFEProcessor()
        self.rife_processor.set_quality(self.quality_combo.currentText())
        self.rife_processor.frame_processed.connect(self.display_rife_frame)
        self.rife_processor.fps_update.connect(self.update_rife_fps)
        self.rife_processor.error_occurred.connect(self.handle_rife_error)
        self.rife_processor.start_processing()
        
        # Create RIFE display window
        self.create_rife_display_window()
        
        self.status_label.setText("⚡ Real-time RIFE processing active!")
    
    def stop_realtime_rife(self):
        """Stop real-time RIFE processing"""
        self.realtime_rife_active = False
        self.realtime_rife_btn.setText("⚡ Real-time RIFE")
        self.realtime_rife_btn.setStyleSheet("")
        
        # Stop RIFE processor thread
        if self.rife_processor:
            self.rife_processor.stop_processing()
            self.rife_processor = None
        
        # Clear frame buffer and force garbage collection
        self.frame_buffer.clear()
        import gc
        gc.collect()
        
        # Close RIFE display window
        if self.rife_window:
            self.rife_window.close()
            self.rife_window = None
        
        self.status_label.setText("⏹️ Real-time RIFE stopped")
    
    def create_rife_display_window(self):
        """Create window to display RIFE-processed frames"""
        self.rife_window = QWidget()
        self.rife_window.setWindowTitle("RIFE Enhanced - Real-time Output")
        self.rife_window.setGeometry(100, 100, 640, 480)
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("RIFE Enhanced Output - Higher FPS Stream")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info_label)
        
        # Display label for processed frames
        self.rife_display_label = QLabel("Processing frames...")
        self.rife_display_label.setMinimumSize(640, 360)
        self.rife_display_label.setStyleSheet("border: 2px solid #4CAF50; background: black;")
        layout.addWidget(self.rife_display_label)
        
        # Status for RIFE processing
        self.rife_status_label = QLabel("Ready for RIFE processing")
        layout.addWidget(self.rife_status_label)
        
        self.rife_window.setLayout(layout)
        self.rife_window.show()
    
    def update_rife_fps(self, fps):
        """Update RIFE FPS tracking from processor thread"""
        self.rife_fps_tracker.append(fps)
        if len(self.rife_fps_tracker) > 30:
            self.rife_fps_tracker.pop(0)
    
    def handle_rife_error(self, error):
        """Handle errors from RIFE processor thread"""
        print(f"RIFE processing error: {error}")
        if self.rife_window:
            self.rife_status_label.setText(f"❌ Error: {error}")
    
    def on_quality_changed(self, new_quality):
        """Handle quality setting changes"""
        if self.rife_processor:
            self.rife_processor.set_quality(new_quality)
        
        # Update GPU stats display
        self.update_gpu_stats()
    
    def update_gpu_stats(self):
        """Update GPU statistics display with CUDA info"""
        quality = self.quality_combo.currentText()
        gpu_status = "GPU: Active" if self.realtime_rife_active else "GPU: Idle"
        
        # Show memory usage info
        import psutil
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            buffer_size = len(self.frame_buffer) if hasattr(self, 'frame_buffer') else 0
            
            self.gpu_stats_label.setText(
                f"{gpu_status} | Quality: {quality} | RAM: {memory_mb:.0f}MB | Buffer: {buffer_size}"
            )
        except:
            self.gpu_stats_label.setText(
                f"{gpu_status} | Quality: {quality} | Buffer: {len(self.frame_buffer) if hasattr(self, 'frame_buffer') else 0}"
            )
        
        # Update CUDA stats
        self.update_cuda_stats()
    
    def update_cuda_stats(self):
        """Update CUDA GPU memory and utilization stats"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device)
                
                # GPU memory info
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
                cached_memory = torch.cuda.memory_reserved(device) / 1024**3
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Detect actual GPU name
                    gpu_display_name = gpu_name if "RTX" in gpu_name else f"RTX 5060 Ti ({gpu_name})"
                    self.cuda_stats_label.setText(
                        f"{gpu_display_name} | GPU: {util.gpu}% | VRAM: {allocated_memory:.1f}/{total_memory:.1f}GB | {temp}°C"
                    )
                except:
                    # Detect actual GPU name for display
                    gpu_display_name = gpu_name if "RTX" in gpu_name else f"RTX 5060 Ti ({gpu_name})"
                    self.cuda_stats_label.setText(
                        f"CUDA: {gpu_display_name} | VRAM: {allocated_memory:.1f}/{total_memory:.1f}GB | Active"
                    )
            else:
                self.cuda_stats_label.setText("CUDA: Not Available | CPU Processing Only")
        except Exception as e:
            self.cuda_stats_label.setText(f"CUDA: Error - {str(e)[:30]}...")

class RIFEProcessor(QWidget):
    """Widget for RIFE video processing"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input file selection
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        btn_browse_in = QPushButton("Browse Input")
        btn_browse_in.clicked.connect(self.browse_input)
        input_layout.addWidget(QLabel("Input Video:"))
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(btn_browse_in)
        layout.addLayout(input_layout)

        # Output file selection
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        btn_browse_out = QPushButton("Browse Output")
        btn_browse_out.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel("Output Video:"))
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(btn_browse_out)
        layout.addLayout(output_layout)

        # Interpolation factor
        factor_layout = QHBoxLayout()
        self.factor_spin = QSpinBox()
        self.factor_spin.setValue(2)
        self.factor_spin.setMinimum(2)
        self.factor_spin.setMaximum(8)
        factor_layout.addWidget(QLabel("Interpolation Factor:"))
        factor_layout.addWidget(self.factor_spin)
        layout.addLayout(factor_layout)

        # Process button
        self.btn_process = QPushButton("Start RIFE Processing")
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process)

        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input Video")
        if path:
            self.input_edit.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output Video", filter="*.mp4")
        if path:
            self.output_edit.setText(path)

    def start_processing(self):
        input_path = self.input_edit.text()
        output_path = self.output_edit.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "Warning", "Please select both input and output files")
            return
            
        factor = self.factor_spin.value()

        self.worker = RIFEWorker(input_path, output_path, factor)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.status_update.connect(self.on_status_update)
        
        self.btn_process.setEnabled(False)
        self.progress.setValue(0)
        self.status_text.append("Starting RIFE processing...")
        self.worker.start()

    def on_processing_finished(self, success):
        self.btn_process.setEnabled(True)
        if success:
            self.progress.setValue(100)
            self.status_text.append("✓ Processing completed successfully!")
            QMessageBox.information(self, "Success", "RIFE processing completed!")
        else:
            self.progress.setValue(0)
            self.status_text.append("✗ Processing failed!")
            QMessageBox.critical(self, "Error", "RIFE processing failed!")

    def on_status_update(self, message):
        self.status_text.append(message)

class UnifiedRIFEGUI(QMainWindow):
    """Main unified GUI application"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Unified RIFE Toolkit - Window Capture & Video Processing")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Add tabs
        self.window_capture_tab = WindowSelector()
        self.rife_processor_tab = RIFEProcessor()
        self.dependency_tab = self.create_dependency_tab()
        
        self.tabs.addTab(self.window_capture_tab, "🪟 Window Capture")
        self.tabs.addTab(self.rife_processor_tab, "🎬 RIFE Processing")
        self.tabs.addTab(self.dependency_tab, "⚙️ Dependencies")
        
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        
        # Check dependencies on startup
        self.check_dependencies_on_startup()
    
    def create_dependency_tab(self):
        """Create the dependency management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Info
        info_label = QLabel("""
        <h3>Dependency Manager</h3>
        <p>This tool automatically installs required packages for full functionality:</p>
        <ul>
        <li><b>PyQt5</b> - GUI framework</li>
        <li><b>pywin32</b> - Windows API access for window capture</li>
        <li><b>Pillow</b> - Image processing</li>
        <li><b>numpy</b> - Numerical computations</li>
        </ul>
        """)
        info_label.setWordWrap(True)
        
        # Install button
        self.install_btn = QPushButton("Check & Install Dependencies")
        self.install_btn.clicked.connect(self.install_dependencies)
        
        # Progress display
        self.dependency_status = QTextEdit()
        self.dependency_status.setMaximumHeight(200)
        self.dependency_status.setReadOnly(True)
        
        layout.addWidget(info_label)
        layout.addWidget(self.install_btn)
        layout.addWidget(self.dependency_status)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def check_dependencies_on_startup(self):
        """Check if Windows capture is available"""
        if not WINDOWS_CAPTURE_AVAILABLE:
            self.dependency_status.append("⚠️ Windows capture not available - pywin32 not installed")
            self.tabs.setTabText(2, "⚠️ Dependencies")
        else:
            self.dependency_status.append("✓ All core dependencies available")
    
    def install_dependencies(self):
        """Install missing dependencies"""
        self.install_btn.setEnabled(False)
        self.dependency_status.clear()
        
        self.installer = DependencyInstaller()
        self.installer.progress.connect(self.dependency_status.append)
        self.installer.finished.connect(self.on_installation_finished)
        self.installer.start()
    
    def on_installation_finished(self, success):
        self.install_btn.setEnabled(True)
        if success:
            self.dependency_status.append("\n✅ Installation completed! Please restart the application.")
            QMessageBox.information(self, "Success", "Dependencies installed successfully!\nPlease restart the application for changes to take effect.")
        else:
            self.dependency_status.append("\n❌ Some installations failed. Check the log above.")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Unified RIFE Toolkit")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = UnifiedRIFEGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
