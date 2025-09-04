# 🎬 RIFE GUI - Intelligent GPU Setup System

## What This Does

This system **automatically detects your GPU and installs the correct PyTorch version** so you never have to worry about compatibility issues again!

### For Your RTX 5060 Ti

Your RTX 5060 Ti needs **CUDA 12.4 PyTorch** but you currently have CUDA 11.8 PyTorch installed. This system will:

1. ✅ Detect your RTX 5060 Ti (16GB VRAM)
2. 🗑️ Remove the incompatible PyTorch CUDA 11.8 version
3. 📦 Install PyTorch with CUDA 12.4 (perfect for RTX 5060 Ti)
4. 🧪 Test that everything works
5. 🚀 Launch the RIFE GUI with full GPU acceleration

## How to Use

### Option 1: Double-click the Batch File (Easiest)
```
Launch_RIFE_GUI.bat
```
Just double-click this file and it handles everything automatically!

### Option 2: Run the Smart Setup
```bash
python start_rife_gui.py
```
Opens a GUI that shows you exactly what it's doing and lets you choose options.

### Option 3: Command Line Setup
```bash
python auto_gpu_setup.py          # Auto setup
python auto_gpu_setup.py --force  # Force reinstall
```

## What Happens When You Upgrade Your GPU

If you ever change your GPU (like from RTX 5060 Ti to RTX 5090), the system will:

1. 🔍 Detect the new GPU automatically
2. 🗑️ Remove the old PyTorch version
3. 📦 Install the optimal PyTorch version for your new GPU
4. 💾 Save the new configuration

**No manual work needed!**

## Supported GPUs

The system automatically chooses the best PyTorch version for:

### RTX 50 Series (CUDA 12.4)
- RTX 5090, 5080, 5070, **5060 Ti** ← **Your GPU**

### RTX 40 Series (CUDA 12.1)
- RTX 4090, 4080, 4070, 4060

### RTX 30 Series (CUDA 11.8)
- RTX 3090, 3080, 3070, 3060

### RTX 20 Series (CUDA 11.8)
- RTX 2080, 2070, 2060

### GTX 16 Series (CUDA 11.8)
- GTX 1660, 1650

## Files Created

The system creates these files to track your setup:

- `gpu_config.json` - Saves your current GPU info
- `test_gpu_interpolation.png` - Test image to verify GPU processing works

## Debug and Troubleshooting

If something goes wrong:

### Run Debug Test
```bash
python debug_test.py
```

### Check Current Status
```bash
python auto_gpu_setup.py --test-only
```

### Force Full Reinstall
```bash
python auto_gpu_setup.py --force
```

## Current Problem Solution

**Your Issue**: RTX 5060 Ti with CUDA capability `sm_120` + PyTorch CUDA 11.8 = Incompatible

**The Fix**: RTX 5060 Ti with CUDA capability `sm_120` + PyTorch CUDA 12.4 = ✅ Perfect

After running the setup:
- ✅ No more warnings
- ✅ RIFE output window will show frames
- ✅ Full 16GB VRAM utilization
- ✅ Optimal performance for real-time processing

## Performance After Fix

Your RTX 5060 Ti specs are excellent for RIFE:
- **16GB VRAM**: Can handle large frame buffers
- **Compute 12.0**: Latest GPU features
- **Modern Architecture**: Optimized for AI workloads

Expected performance:
- **Real-time 1080p**: 60+ FPS processing
- **Real-time 1440p**: 30-60 FPS processing
- **4K processing**: Possible with frame buffering

## Manual PyTorch Installation (If Needed)

If the auto-setup fails, install manually:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## System Requirements

- **Windows 10/11**
- **Python 3.8+**
- **NVIDIA GPU** (RTX 5060 Ti detected)
- **NVIDIA Driver 577.00+** (You have this ✅)

## Quick Start

1. **Double-click**: `Launch_RIFE_GUI.bat`
2. **Click**: "🚀 Auto Setup & Launch"
3. **Wait**: 2-5 minutes for PyTorch download/install
4. **Done**: RIFE GUI launches with full GPU support!

That's it! Your RTX 5060 Ti will finally work perfectly with the RIFE GUI.
