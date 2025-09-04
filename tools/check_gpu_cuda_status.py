#!/usr/bin/env python3
"""
Check GPU and CUDA status for RTX 5060 Ti
"""

import sys
import subprocess
import os

def check_nvidia_gpu():
    """Check NVIDIA GPU details"""
    print("🔍 CHECKING NVIDIA GPU...")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    name, driver, memory, compute_cap = parts[:4]
                    print(f"GPU {i}: {name}")
                    print(f"  Driver: {driver}")
                    print(f"  Memory: {memory} MB")
                    print(f"  Compute Capability: {compute_cap}")
                    
                    # Check if it's RTX 5060 Ti
                    if "5060 Ti" in name:
                        print(f"  ✅ RTX 5060 Ti detected!")
                        print(f"  📊 Compute Cap: {compute_cap} (sm_{compute_cap.replace('.', '')})")
                        return compute_cap
                    print()
        else:
            print("❌ nvidia-smi not found or failed")
            return None
            
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return None

def check_cuda_version():
    """Check CUDA version"""
    print("\n🔍 CHECKING CUDA VERSION...")
    print("=" * 50)
    
    try:
        # Check nvcc version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    print(f"NVCC: {line.strip()}")
                    # Extract version
                    if 'V' in line:
                        version = line.split('V')[1].split(',')[0].strip()
                        print(f"  ✅ CUDA Toolkit: {version}")
                        return version
        else:
            print("❌ nvcc not found")
            
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
    
    # Check CUDA runtime
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip()
                    print(f"  ✅ CUDA Runtime: {cuda_version}")
                    return cuda_version
    except:
        pass
    
    return None

def check_pytorch():
    """Check PyTorch version and CUDA support"""
    print("\n🔍 CHECKING PYTORCH...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version (PyTorch): {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                
                # Check if this GPU is supported
                compute_cap = f"{props.major}.{props.minor}"
                if props.major >= 8:  # RTX 30/40/50 series
                    print(f"    ✅ Supported by PyTorch")
                elif props.major == 12:  # RTX 5060 Ti
                    print(f"    ⚠️  Very new GPU - may need PyTorch nightly")
                else:
                    print(f"    ❌ May not be fully supported")
                    
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False

def check_cupy():
    """Check CuPy status"""
    print("\n🔍 CHECKING CUPY...")
    print("=" * 50)
    
    try:
        import cupy as cp
        print(f"✅ CuPy version: {cp.__version__}")
        
        # Test basic GPU operation
        try:
            a = cp.array([1, 2, 3])
            b = cp.array([4, 5, 6])
            c = a + b
            print(f"✅ CuPy GPU test: {cp.asnumpy(c)}")
            return True
        except Exception as e:
            print(f"❌ CuPy GPU operation failed: {e}")
            return False
            
    except ImportError:
        print("❌ CuPy not installed")
        return False

def recommend_setup(compute_cap):
    """Recommend setup for RTX 5060 Ti"""
    print("\n🚀 RECOMMENDATIONS FOR RTX 5060 Ti:")
    print("=" * 50)
    
    if compute_cap and "12.0" in compute_cap:
        print("Your RTX 5060 Ti has compute capability 12.0 - very new!")
        print()
        print("REQUIRED SETUP:")
        print("1. CUDA Toolkit 12.6 or later")
        print("2. PyTorch nightly build with CUDA 12.4+ support")
        print("3. CuPy built for CUDA 12.x")
        print()
        print("INSTALLATION COMMANDS:")
        print("# Uninstall current PyTorch")
        print("pip uninstall torch torchvision torchaudio")
        print()
        print("# Install PyTorch nightly with CUDA 12.4")
        print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
        print()
        print("# Install CuPy for CUDA 12.x")
        print("pip install cupy-cuda12x")
        
    else:
        print("For optimal GPU performance:")
        print("1. Update to latest NVIDIA drivers")
        print("2. Install CUDA Toolkit 12.x")
        print("3. Use PyTorch with matching CUDA version")

def main():
    print("🎯 RTX 5060 Ti - GPU CUDA STATUS CHECK")
    print("=" * 50)
    print()
    
    # Check GPU
    compute_cap = check_nvidia_gpu()
    
    # Check CUDA
    cuda_version = check_cuda_version()
    
    # Check PyTorch
    pytorch_working = check_pytorch()
    
    # Check CuPy
    cupy_working = check_cupy()
    
    # Recommendations
    recommend_setup(compute_cap)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"✅ GPU Detected: {'Yes' if compute_cap else 'No'}")
    print(f"✅ CUDA Available: {'Yes' if cuda_version else 'No'}")
    print(f"✅ PyTorch GPU: {'Yes' if pytorch_working else 'No'}")
    print(f"✅ CuPy GPU: {'Yes' if cupy_working else 'No'}")
    
    if compute_cap == "12.0" and not pytorch_working:
        print("\n⚠️  RTX 5060 Ti DETECTED BUT GPU NOT WORKING")
        print("   Need to install PyTorch nightly for compute capability 12.0 support!")
        return False
    
    if pytorch_working and cupy_working:
        print("\n🎉 GPU SETUP READY FOR GPU-ONLY RIFE!")
        return True
    else:
        print("\n❌ GPU setup incomplete - run recommended installation commands")
        return False

if __name__ == "__main__":
    main()
