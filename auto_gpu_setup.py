#!/usr/bin/env python3
"""
Intelligent GPU Detection and PyTorch Auto-Installer
Automatically detects GPU and installs compatible PyTorch version
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path

class GPUSetupManager:
    def __init__(self):
        self.config_file = Path("gpu_config.json")
        self.current_config = self.load_config()
        
    def load_config(self):
        """Load existing GPU configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_config(self, config):
        """Save GPU configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def detect_gpu(self):
        """Detect current GPU using nvidia-smi"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                gpu_info = lines[0].split(', ')
                
                if len(gpu_info) >= 3:
                    name = gpu_info[0].strip()
                    compute_cap = gpu_info[1].strip()
                    memory_mb = int(gpu_info[2].strip())
                    
                    return {
                        'name': name,
                        'compute_capability': compute_cap,
                        'memory_gb': round(memory_mb / 1024, 1),
                        'detected_method': 'nvidia-smi'
                    }
        except:
            pass
        
        # Fallback: try to get info from PyTorch if available
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                name = torch.cuda.get_device_name(device)
                props = torch.cuda.get_device_properties(device)
                memory_gb = props.total_memory / (1024**3)
                major, minor = props.major, props.minor
                compute_cap = f"{major}.{minor}"
                
                return {
                    'name': name,
                    'compute_capability': compute_cap,
                    'memory_gb': round(memory_gb, 1),
                    'detected_method': 'pytorch'
                }
        except:
            pass
        
        return None
    
    def get_pytorch_version_for_gpu(self, gpu_info):
        """Determine the correct PyTorch version for the detected GPU"""
        if not gpu_info:
            return {
                'cuda_version': 'cpu',
                'install_command': 'pip install torch torchvision torchaudio',
                'reason': 'No NVIDIA GPU detected'
            }
        
        name = gpu_info['name'].upper()
        compute_cap = gpu_info['compute_capability']
        
        try:
            # Parse compute capability
            major = int(float(compute_cap))
            minor = int((float(compute_cap) * 10) % 10)
        except:
            major, minor = 7, 5  # Default fallback
        
        # GPU-specific logic
        gpu_mappings = {
            # RTX 50 series (latest) - use nightly builds for sm_120+ support
            'RTX 5090': {'cuda': 'nightly', 'min_major': 12},
            'RTX 5080': {'cuda': 'nightly', 'min_major': 12},
            'RTX 5070': {'cuda': 'nightly', 'min_major': 12},
            'RTX 5060': {'cuda': 'nightly', 'min_major': 12},
            
            # RTX 40 series
            'RTX 4090': {'cuda': 'cu121', 'min_major': 8},
            'RTX 4080': {'cuda': 'cu121', 'min_major': 8},
            'RTX 4070': {'cuda': 'cu121', 'min_major': 8},
            'RTX 4060': {'cuda': 'cu121', 'min_major': 8},
            
            # RTX 30 series
            'RTX 3090': {'cuda': 'cu118', 'min_major': 8},
            'RTX 3080': {'cuda': 'cu118', 'min_major': 8},
            'RTX 3070': {'cuda': 'cu118', 'min_major': 8},
            'RTX 3060': {'cuda': 'cu118', 'min_major': 8},
            
            # RTX 20 series
            'RTX 2080': {'cuda': 'cu118', 'min_major': 7},
            'RTX 2070': {'cuda': 'cu118', 'min_major': 7},
            'RTX 2060': {'cuda': 'cu118', 'min_major': 7},
            
            # GTX 16 series
            'GTX 1660': {'cuda': 'cu118', 'min_major': 7},
            'GTX 1650': {'cuda': 'cu118', 'min_major': 7},
        }
        
        # Find matching GPU
        selected_cuda = None
        for gpu_pattern, config in gpu_mappings.items():
            if gpu_pattern in name:
                # Check if compute capability is sufficient
                if major >= config.get('min_major', 7):
                    selected_cuda = config['cuda']
                    break
        
        # Fallback based on compute capability
        if not selected_cuda:
            if major >= 12:  # sm_120+ (RTX 50 series) - needs nightly
                selected_cuda = 'nightly'
            elif major >= 8:  # sm_80+ (RTX 30/40 series)
                selected_cuda = 'cu121'
            elif major >= 7:  # sm_70+ (RTX 20 series, GTX 16 series)
                selected_cuda = 'cu118'
            elif major >= 6:  # sm_60+ (GTX 10 series)
                selected_cuda = 'cu118'
            else:
                selected_cuda = 'cpu'  # Very old GPU, use CPU
        
        if selected_cuda == 'cpu':
            return {
                'cuda_version': 'cpu',
                'install_command': 'pip install torch torchvision torchaudio',
                'reason': f'GPU {name} (sm_{major}{minor}) too old for GPU acceleration'
            }
        
        # Build install command
        if selected_cuda == 'nightly':
            # Use PyTorch nightly for bleeding-edge GPU support
            return {
                'cuda_version': 'nightly',
                'install_command': 'pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124',
                'reason': f'Nightly build required for {name} (compute {compute_cap}) - bleeding edge GPU'
            }
        
        cuda_map = {
            'cu124': 'https://download.pytorch.org/whl/cu124',
            'cu121': 'https://download.pytorch.org/whl/cu121',
            'cu118': 'https://download.pytorch.org/whl/cu118'
        }
        
        index_url = cuda_map.get(selected_cuda, cuda_map['cu121'])
        
        return {
            'cuda_version': selected_cuda,
            'install_command': f'pip install torch torchvision torchaudio --index-url {index_url}',
            'reason': f'Optimal for {name} (compute {compute_cap})'
        }
    
    def check_current_pytorch(self):
        """Check current PyTorch installation"""
        try:
            import torch
            return {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
                'installed': True
            }
        except ImportError:
            return {'installed': False}
    
    def gpu_changed(self, current_gpu, saved_gpu):
        """Check if GPU has changed since last run"""
        if not saved_gpu:
            return True  # First run
        
        if not current_gpu:
            return saved_gpu.get('name') is not None  # GPU was removed
        
        # Check if it's a different GPU
        return (current_gpu.get('name') != saved_gpu.get('name') or
                current_gpu.get('compute_capability') != saved_gpu.get('compute_capability'))
    
    def uninstall_pytorch(self):
        """Uninstall current PyTorch installation"""
        print("🗑️  Uninstalling current PyTorch installation...")
        
        packages_to_remove = ['torch', 'torchvision', 'torchaudio']
        
        for package in packages_to_remove:
            try:
                print(f"   Removing {package}...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', package, '-y'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ {package} removed")
                else:
                    print(f"   ⚠️  {package} not found or failed to remove")
            except Exception as e:
                print(f"   ❌ Error removing {package}: {e}")
    
    def install_pytorch(self, install_command):
        """Install PyTorch with the specified command"""
        print("📦 Installing PyTorch...")
        print(f"Command: {install_command}")
        
        try:
            # Split command properly
            if '--index-url' in install_command:
                parts = install_command.split()
                cmd = [sys.executable, '-m'] + parts
            else:
                cmd = [sys.executable, '-m'] + install_command.split()
            
            # Run installation
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True, universal_newlines=True)
            
            print("Installing... (this may take a few minutes)")
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print("✅ PyTorch installation successful!")
                return True
            else:
                print(f"❌ Installation failed:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def test_installation(self):
        """Test the PyTorch installation and check for GPU compatibility issues"""
        print("🧪 Testing PyTorch installation...")
        
        try:
            import torch
            print(f"   PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                print(f"   ✅ CUDA available: {device_count} device(s)")
                print(f"   🎮 Current device: {device_name}")
                
                # Check for RTX 5060 Ti compatibility issues
                if "RTX 5060" in device_name:
                    major, minor = torch.cuda.get_device_capability(current_device)
                    if major >= 12:  # sm_120 or higher
                        cuda_version = torch.version.cuda
                        if cuda_version and float(cuda_version) < 12.0:
                            print(f"   ❌ GPU Compatibility Issue Detected!")
                            print(f"   RTX 5060 Ti (sm_{major}{minor}) requires CUDA 12.x")
                            print(f"   Current PyTorch uses CUDA {cuda_version}")
                            print(f"   This will cause warnings and performance issues!")
                            return False
                
                # Test basic operation
                test_tensor = torch.randn(100, 100).cuda()
                result = test_tensor * 2.0
                result_cpu = result.cpu()
                
                print(f"   ✅ GPU operations: OK")
                
                # Check memory
                props = torch.cuda.get_device_properties(current_device)
                total_memory = props.total_memory / (1024**3)
                print(f"   💾 GPU Memory: {total_memory:.1f} GB")
                
                return True
            else:
                print("   ⚠️  CUDA not available (CPU-only mode)")
                return True
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return False
    
    def run_setup(self, force=False):
        """Main setup routine"""
        print("🔧 GPU Auto-Setup Manager")
        print("=" * 50)
        
        # Detect current GPU
        print("🔍 Detecting GPU...")
        current_gpu = self.detect_gpu()
        
        if current_gpu:
            print(f"   ✅ Found: {current_gpu['name']}")
            print(f"   🎯 Compute Capability: {current_gpu['compute_capability']}")
            print(f"   💾 VRAM: {current_gpu['memory_gb']} GB")
        else:
            print("   ⚠️  No NVIDIA GPU detected")
        
        # Check if GPU changed
        saved_gpu = self.current_config.get('gpu')
        gpu_has_changed = force or self.gpu_changed(current_gpu, saved_gpu)
        
        if not gpu_has_changed:
            print("✅ GPU unchanged, current setup should be compatible")
            # Still test the installation
            if self.test_installation():
                print("🎉 Setup validation complete!")
                return True
            else:
                print("⚠️  Installation test failed, forcing reinstall...")
                gpu_has_changed = True
        
        if gpu_has_changed:
            print("🔄 GPU change detected or forced reinstall")
            
            # Get optimal PyTorch version
            pytorch_config = self.get_pytorch_version_for_gpu(current_gpu)
            print(f"   Recommended: {pytorch_config['cuda_version']}")
            print(f"   Reason: {pytorch_config['reason']}")
            
            # Check current PyTorch
            current_pytorch = self.check_current_pytorch()
            
            if current_pytorch.get('installed'):
                print(f"   Current PyTorch: {current_pytorch['version']}")
                
                # Uninstall if different CUDA version needed
                current_cuda = 'cpu' if not current_pytorch.get('cuda_available') else f"cu{current_pytorch.get('cuda_version', '').replace('.', '')}"
                if current_cuda != pytorch_config['cuda_version']:
                    self.uninstall_pytorch()
                else:
                    print("   ✅ Current installation is compatible")
                    success = self.test_installation()
                    if success:
                        self.save_config({'gpu': current_gpu, 'pytorch': pytorch_config})
                        return True
            
            # Install new PyTorch
            if self.install_pytorch(pytorch_config['install_command']):
                if self.test_installation():
                    # Save configuration
                    self.save_config({'gpu': current_gpu, 'pytorch': pytorch_config})
                    print("🎉 Setup complete!")
                    return True
                else:
                    print("❌ Installation test failed")
                    return False
            else:
                print("❌ Installation failed")
                return False
        
        return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Auto-Setup Manager')
    parser.add_argument('--force', action='store_true', help='Force reinstall even if GPU unchanged')
    parser.add_argument('--test-only', action='store_true', help='Only test current installation')
    
    args = parser.parse_args()
    
    manager = GPUSetupManager()
    
    if args.test_only:
        print("🧪 Testing current installation only...")
        success = manager.test_installation()
        sys.exit(0 if success else 1)
    
    success = manager.run_setup(force=args.force)
    
    if success:
        print("\n💡 You can now run the RIFE GUI:")
        print("   python unified_rife_gui.py")
    else:
        print("\n⚠️  Setup failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
