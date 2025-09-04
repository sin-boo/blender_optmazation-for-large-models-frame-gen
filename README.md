# GPU-Only RIFE - Real-Time Video Frame Interpolation

A high-performance, GPU-accelerated implementation of RIFE (Real-time Intermediate Flow Estimation) optimized for NVIDIA RTX GPUs, specifically designed for real-time video frame interpolation with DLSS-like performance.

## 🚀 Features

- **Pure GPU Processing**: 100% GPU-accelerated using CuPy and CUDA
- **Real-time Performance**: Optimized for NVIDIA RTX 5060 Ti and similar GPUs
- **Window Capture**: Capture and interpolate frames from any application window
- **Multiple Quality Modes**: High, Ultra, and Maximum quality settings
- **Memory Optimized**: Efficient GPU memory management with pre-allocated buffers
- **Professional GUI**: Easy-to-use Tkinter interface with real-time performance monitoring
- **No CPU Fallbacks**: Designed for pure GPU operation

## 🎯 Target Hardware

- **Primary**: NVIDIA RTX 5060 Ti (16GB VRAM, Compute Capability 12.0)
- **Compatible**: RTX 30/40/50 series GPUs with sufficient VRAM
- **CUDA**: Requires CUDA 12.x support

## 📋 Requirements

### Software Dependencies
```
Python 3.8+
PyTorch (with CUDA support)
CuPy (CUDA 12.x)
PIL (Pillow)
NumPy
Tkinter (usually included with Python)
pywin32 (Windows only)
```

### Hardware Requirements
- NVIDIA GPU with CUDA Compute Capability 6.0+
- Minimum 8GB GPU VRAM (16GB recommended)
- CUDA 12.x compatible GPU drivers

## 🛠️ Installation

1. **Install CUDA Toolkit** (if not already installed):
   ```bash
   # Download from NVIDIA: https://developer.nvidia.com/cuda-downloads
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install cupy-cuda12x
   pip install pillow numpy pywin32
   ```

3. **Clone this Repository**:
   ```bash
   git clone <repository-url>
   cd rife-gui-app
   ```

## 🚀 Quick Start

### 🎯 **Easiest Way - Use the Launcher**
```bash
python LAUNCH_RIFE.py
```
The launcher provides a simple GUI to choose the right implementation for your needs.

### 📊 **Run Performance Benchmark**
```bash
python gpu_benchmark.py
```
Test your GPU performance and get optimization recommendations.

### ⚡ **Direct Launch - Performance Implementations**

1. **`core/gpu_rife_optimized.py`** - 🚀 High-performance GPU-only with advanced optimizations
2. **`core/gpu_only_rtx_5060_ti.py`** - 💎 Specifically optimized for RTX 5060 Ti
3. **`core/final_gpu_only_rife.py`** - ⚡ Complete GPU-only real-time processing
4. **`core/rtx_5060_ti_optimized_rife.py`** - 🔧 Alternative optimized version

## 📁 Project Structure

The project is organized for easy navigation:

```
rife-gui-app/
├── 🚀 LAUNCH_RIFE.py          # Main launcher (START HERE)
├── 📊 gpu_benchmark.py        # Performance benchmark system
├── 📄 README.md              # This documentation
│
├── 🔥 core/                   # High-performance implementations
│   ├── gpu_rife_optimized.py  # Advanced optimized version
│   ├── gpu_only_rtx_5060_ti.py # RTX 5060 Ti specific
│   ├── final_gpu_only_rife.py # Complete GPU-only RIFE
│   └── rife_engine.py         # Traditional RIFE engine
│
├── 🛠️ tools/                  # Utilities and testing
│   ├── check_gpu_cuda_status.py
│   ├── auto_gpu_setup.py
│   └── test_*.py             # Testing scripts
│
├── 📚 docs/                  # Documentation
├── 📁 legacy/               # Development versions
└── 🖼️ examples/             # Sample outputs
```

### 🎯 **For New Users**
Just run `python LAUNCH_RIFE.py` - it guides you to the right tool!

### GUI Controls

- **Window Selection**: Choose target application window for capture
- **Quality Mode**: Select processing quality (High/Ultra/Maximum)
- **Start Capture**: Begin window capture
- **Start GPU Processing**: Enable real-time RIFE interpolation
- **Performance Monitor**: Real-time FPS and GPU memory usage
- **Clear GPU Memory**: Free allocated GPU resources

## 🔧 Configuration

### Quality Modes

- **High**: Balanced performance and quality
- **Ultra**: Enhanced quality with moderate performance impact
- **Maximum**: Highest quality processing (requires more VRAM)

### GPU Memory Management

The system automatically manages GPU memory with:
- Pre-allocated frame buffers for common resolutions
- Dynamic memory scaling based on input size
- Automatic cleanup and garbage collection

## 📊 Performance

### RTX 5060 Ti Benchmarks
- **1080p Real-time**: 60+ FPS
- **1440p Real-time**: 45+ FPS  
- **Memory Usage**: 8-12GB VRAM typical
- **Latency**: <16ms processing time

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce quality mode
   - Clear GPU memory between sessions
   - Close other GPU-intensive applications

2. **No Output Window**:
   - Check console for CUDA errors
   - Verify GPU drivers are up to date
   - Ensure sufficient VRAM available

3. **Poor Performance**:
   - Verify GPU is being used (not CPU fallback)
   - Check for thermal throttling
   - Update NVIDIA drivers

### Debug Mode
Enable detailed logging by modifying the debug flags in the source code.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source. Please respect the licenses of the underlying technologies (PyTorch, CuPy, etc.).

## 🙏 Acknowledgments

- RIFE algorithm by the original research team
- PyTorch and CuPy communities
- NVIDIA CUDA development tools

## 📞 Support

For issues, questions, or improvements, please open an issue on GitHub.

---

**Note**: This implementation is optimized for real-time performance on RTX GPUs and may require adjustments for other hardware configurations.
