#!/usr/bin/env python3
"""
Debug test script for RIFE GUI app
Run this to verify GPU processing and frame display work correctly
"""

import sys
import os

def test_dependencies():
    """Test that all required dependencies are available"""
    print("🧪 Testing Dependencies...")
    
    dependencies = {
        "PyQt5": "PyQt5.QtWidgets",
        "PIL": "PIL.Image", 
        "PyTorch": "torch",
        "Windows APIs": "win32gui",
        "NumPy": "numpy"
    }
    
    missing = []
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}: OK")
        except ImportError as e:
            print(f"  ❌ {name}: MISSING - {e}")
            missing.append(name)
    
    return missing

def test_cuda_setup():
    """Test CUDA availability and configuration"""
    print("\n🚀 Testing CUDA/GPU Setup...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"  ✅ CUDA available: {device_count} device(s)")
            print(f"  🎮 Current device: {current_device} - {device_name}")
            
            # Test basic GPU operations
            test_tensor = torch.randn(100, 100).cuda()
            result = test_tensor * 2.0
            result_cpu = result.cpu()
            
            print(f"  ✅ GPU tensor operations: OK")
            
            # Test memory
            total_mem = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            print(f"  💾 GPU Memory: {total_mem:.1f} GB total")
            
            return True
        else:
            print(f"  ❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"  ❌ CUDA test failed: {e}")
        return False

def test_simple_frame_processing():
    """Test basic frame creation and processing"""
    print("\n🖼️  Testing Frame Processing...")
    
    try:
        from PIL import Image, ImageDraw
        import torch
        import torchvision.transforms as transforms
        
        # Create test frames
        frame1 = Image.new('RGB', (320, 240), color='red')
        frame2 = Image.new('RGB', (320, 240), color='blue')
        
        # Add some details
        draw1 = ImageDraw.Draw(frame1)
        draw2 = ImageDraw.Draw(frame2)
        draw1.rectangle([50, 50, 150, 150], fill='white')
        draw2.rectangle([100, 100, 200, 200], fill='white')
        
        print(f"  ✅ Created test frames: {frame1.size}, {frame2.size}")
        
        # Test GPU processing if available
        if torch.cuda.is_available():
            print(f"  🚀 Testing GPU interpolation...")
            
            transform = transforms.ToTensor()
            device = torch.device('cuda')
            
            with torch.no_grad():
                tensor1 = transform(frame1).unsqueeze(0).to(device)
                tensor2 = transform(frame2).unsqueeze(0).to(device)
                
                # Simple blend
                result_tensor = (tensor1 + tensor2) * 0.5
                result_tensor = torch.clamp(result_tensor, 0, 1)
                
                # Convert back
                result_cpu = result_tensor.squeeze(0).cpu()
                to_pil = transforms.ToPILImage()
                result_image = to_pil(result_cpu)
                
                print(f"  ✅ GPU interpolation successful: {result_image.size}")
                
                # Save test result
                result_image.save("test_gpu_interpolation.png")
                print(f"  💾 Saved test result to: test_gpu_interpolation.png")
                
                return True
        else:
            print(f"  ⚠️  GPU not available, testing CPU fallback...")
            
            # CPU fallback
            import numpy as np
            arr1 = np.array(frame1)
            arr2 = np.array(frame2)
            result_arr = ((arr1.astype(np.uint16) + arr2.astype(np.uint16)) // 2).astype(np.uint8)
            result_image = Image.fromarray(result_arr)
            
            result_image.save("test_cpu_interpolation.png")
            print(f"  ✅ CPU interpolation successful: {result_image.size}")
            print(f"  💾 Saved test result to: test_cpu_interpolation.png")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Frame processing test failed: {e}")
        return False

def test_qt_display():
    """Test Qt display functionality"""
    print("\n🖥️  Testing Qt Display...")
    
    try:
        from PyQt5.QtWidgets import QApplication, QLabel, QWidget
        from PyQt5.QtGui import QPixmap, QImage
        from PIL import Image
        
        # Create test image
        test_img = Image.new('RGB', (640, 360), color='green')
        
        # Convert to Qt format
        data = test_img.tobytes("raw", "RGB")
        qimage = QImage(data, test_img.width, test_img.height, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        
        if qpixmap.isNull():
            print(f"  ❌ QPixmap creation failed")
            return False
        else:
            print(f"  ✅ QPixmap creation successful: {qpixmap.width()}x{qpixmap.height()}")
            return True
            
    except Exception as e:
        print(f"  ❌ Qt display test failed: {e}")
        return False

def main():
    print("🔧 RIFE GUI Debug Test Script")
    print("=" * 50)
    
    # Change to the app directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    missing_deps = test_dependencies()
    cuda_ok = test_cuda_setup()
    frame_ok = test_simple_frame_processing()
    qt_ok = test_qt_display()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Dependencies: {'✅ OK' if not missing_deps else '❌ Missing: ' + ', '.join(missing_deps)}")
    print(f"  CUDA/GPU: {'✅ OK' if cuda_ok else '❌ Failed'}")
    print(f"  Frame Processing: {'✅ OK' if frame_ok else '❌ Failed'}")
    print(f"  Qt Display: {'✅ OK' if qt_ok else '❌ Failed'}")
    
    if missing_deps:
        print("\n🚨 Missing Dependencies - Install with:")
        for dep in missing_deps:
            if dep == "PyTorch":
                print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            elif dep == "Windows APIs":
                print("    pip install pywin32")
            else:
                print(f"    pip install {dep.lower()}")
    
    overall_status = not missing_deps and cuda_ok and frame_ok and qt_ok
    print(f"\n🎯 Overall Status: {'✅ READY TO RUN' if overall_status else '❌ ISSUES DETECTED'}")
    
    if overall_status:
        print("\n💡 Your system should work with the RIFE GUI!")
        print("   Try running: python unified_rife_gui.py")
    else:
        print("\n⚠️  Fix the issues above before running the main app")

if __name__ == "__main__":
    main()
