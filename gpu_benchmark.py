#!/usr/bin/env python3
"""
GPU-Only RIFE Benchmark System
Automated performance testing for RTX 5060 Ti and other GPUs
"""

import time
import gc
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

try:
    import cupy as cp
    import numpy as np
    from PIL import Image
    import psutil
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"⚠️  Missing dependencies: {e}")
    print("Please install: pip install cupy-cuda12x numpy pillow psutil")

class GPUBenchmark:
    """Comprehensive GPU benchmark suite for RIFE implementations"""
    
    def __init__(self):
        self.results = {}
        self.gpu_info = None
        self.start_time = None
        
        if DEPENDENCIES_AVAILABLE:
            self.initialize_gpu()
    
    def initialize_gpu(self):
        """Initialize GPU and gather system information"""
        try:
            # Initialize CuPy
            cp.cuda.Device(0).use()
            self.gpu_info = {
                'name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'),
                'compute_capability': f"{cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}",
                'total_memory': cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3),  # GB
                'free_memory': cp.cuda.MemPool().get_limit() / (1024**3) if hasattr(cp.cuda, 'MemPool') else 'Unknown'
            }
            
            # CPU info
            self.cpu_info = {
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown',
                'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
                'ram_available': psutil.virtual_memory().available / (1024**3)  # GB
            }
            
            print(f"🚀 GPU: {self.gpu_info['name']} ({self.gpu_info['total_memory']:.1f}GB)")
            print(f"💻 CPU: {self.cpu_info['cpu_count']} cores, {self.cpu_info['ram_total']:.1f}GB RAM")
            
        except Exception as e:
            print(f"❌ GPU initialization failed: {e}")
            self.gpu_info = None
    
    def create_test_frames(self, resolution: Tuple[int, int], count: int = 10) -> List[cp.ndarray]:
        """Create synthetic test frames for benchmarking"""
        width, height = resolution
        frames = []
        
        print(f"📸 Creating {count} test frames at {width}x{height}...")
        
        for i in range(count):
            # Create synthetic frame with gradient and motion
            x = cp.linspace(0, 1, width)
            y = cp.linspace(0, 1, height)
            X, Y = cp.meshgrid(x, y)
            
            # Add time-based animation
            t = i / count * 2 * cp.pi
            frame_data = cp.sin(X * 10 + t) * cp.cos(Y * 10 + t) * 0.5 + 0.5
            
            # Convert to RGB
            frame = cp.stack([frame_data, frame_data * 0.8, frame_data * 0.6], axis=2)
            frame = (frame * 255).astype(cp.uint8)
            
            frames.append(frame)
        
        return frames
    
    def benchmark_memory_usage(self, resolutions: List[Tuple[int, int]]) -> Dict:
        """Benchmark GPU memory usage across different resolutions"""
        print("\n🧠 Testing GPU Memory Usage...")
        memory_results = {}
        
        for resolution in resolutions:
            try:
                width, height = resolution
                
                # Clear GPU memory
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                
                # Measure initial memory
                mem_start = cp.get_default_memory_pool().used_bytes()
                
                # Create test frames
                frames = self.create_test_frames(resolution, count=5)
                
                # Allocate buffers (simulating RIFE workflow)
                buffer1 = cp.zeros((height, width, 3), dtype=cp.float32)
                buffer2 = cp.zeros((height, width, 3), dtype=cp.float32)
                output_buffer = cp.zeros((height, width, 3), dtype=cp.float32)
                
                # Measure peak memory
                mem_peak = cp.get_default_memory_pool().used_bytes()
                
                memory_used = (mem_peak - mem_start) / (1024**2)  # MB
                
                memory_results[f"{width}x{height}"] = {
                    'memory_mb': memory_used,
                    'memory_per_frame_mb': memory_used / len(frames),
                    'success': True
                }
                
                print(f"  {width}x{height}: {memory_used:.1f}MB ({memory_used/len(frames):.1f}MB per frame)")
                
                # Cleanup
                del frames, buffer1, buffer2, output_buffer
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                memory_results[f"{width}x{height}"] = {
                    'error': str(e),
                    'success': False
                }
                print(f"  {width}x{height}: ❌ Error - {e}")
        
        return memory_results
    
    def benchmark_tensor_operations(self, resolutions: List[Tuple[int, int]]) -> Dict:
        """Benchmark core tensor operations used in RIFE"""
        print("\n⚡ Testing Tensor Operations Performance...")
        tensor_results = {}
        
        operations = {
            'tensor_creation': self._bench_tensor_creation,
            'convolution': self._bench_convolution,
            'interpolation': self._bench_interpolation,
            'memory_transfer': self._bench_memory_transfer
        }
        
        for resolution in resolutions:
            width, height = resolution
            
            try:
                res_results = {}
                
                # Test each operation
                for op_name, op_func in operations.items():
                    times = []
                    
                    # Run each operation multiple times for accurate timing
                    for _ in range(5):
                        cp.cuda.Stream.null.synchronize()  # Ensure GPU is ready
                        
                        start_time = time.time()
                        op_func(width, height)
                        cp.cuda.Stream.null.synchronize()  # Wait for completion
                        
                        times.append(time.time() - start_time)
                    
                    # Calculate statistics
                    avg_time = sum(times) / len(times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    
                    res_results[op_name] = {
                        'avg_time_ms': avg_time * 1000,
                        'fps': fps,
                        'min_time_ms': min(times) * 1000,
                        'max_time_ms': max(times) * 1000
                    }
                
                tensor_results[f"{width}x{height}"] = res_results
                print(f"  {width}x{height}: ✅ Complete")
                
            except Exception as e:
                tensor_results[f"{width}x{height}"] = {'error': str(e)}
                print(f"  {width}x{height}: ❌ Error - {e}")
        
        return tensor_results
    
    def _bench_tensor_creation(self, width: int, height: int):
        """Benchmark tensor creation and basic operations"""
        # Create tensors
        a = cp.random.random((height, width, 3), dtype=cp.float32)
        b = cp.random.random((height, width, 3), dtype=cp.float32)
        
        # Basic operations
        c = a + b
        d = a * b
        e = cp.maximum(a, b)
        
        # Cleanup
        del a, b, c, d, e
    
    def _bench_convolution(self, width: int, height: int):
        """Benchmark convolution operations (core of neural networks)"""
        from scipy import ndimage
        
        # Create input tensor
        input_tensor = cp.random.random((height, width, 3), dtype=cp.float32)
        
        # Simple convolution kernel (edge detection)
        kernel = cp.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=cp.float32)
        
        # Apply convolution to each channel
        for channel in range(3):
            output = ndimage.convolve(cp.asnumpy(input_tensor[:, :, channel]), 
                                    cp.asnumpy(kernel), mode='constant')
        
        del input_tensor
    
    def _bench_interpolation(self, width: int, height: int):
        """Benchmark interpolation operations"""
        # Create two frames
        frame1 = cp.random.random((height, width, 3), dtype=cp.float32)
        frame2 = cp.random.random((height, width, 3), dtype=cp.float32)
        
        # Simple linear interpolation
        alpha = 0.5
        interpolated = frame1 * alpha + frame2 * (1 - alpha)
        
        # More complex blending
        weight_map = cp.random.random((height, width, 1), dtype=cp.float32)
        weighted_blend = frame1 * weight_map + frame2 * (1 - weight_map)
        
        del frame1, frame2, interpolated, weighted_blend
    
    def _bench_memory_transfer(self, width: int, height: int):
        """Benchmark CPU<->GPU memory transfer"""
        # CPU to GPU
        cpu_array = np.random.random((height, width, 3)).astype(np.float32)
        gpu_array = cp.asarray(cpu_array)
        
        # GPU to CPU
        result_cpu = cp.asnumpy(gpu_array)
        
        del cpu_array, gpu_array, result_cpu
    
    def benchmark_real_world_scenario(self, resolution: Tuple[int, int], duration: int = 10) -> Dict:
        """Simulate real-world RIFE processing scenario"""
        print(f"\n🎬 Real-world scenario test at {resolution[0]}x{resolution[1]} for {duration} seconds...")
        
        width, height = resolution
        
        try:
            # Initialize buffers
            frame_buffer = self.create_test_frames(resolution, count=2)
            processing_buffer = cp.zeros((height, width, 3), dtype=cp.float32)
            
            frames_processed = 0
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Simulate RIFE processing pipeline
                
                # 1. Frame preprocessing
                frame1 = frame_buffer[0].astype(cp.float32) / 255.0
                frame2 = frame_buffer[1].astype(cp.float32) / 255.0
                
                # 2. Motion estimation (simplified)
                motion_vector = frame2 - frame1
                
                # 3. Feature extraction (simplified convolution)
                kernel = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32) / 4.0
                for channel in range(3):
                    conv_result = cp.convolve(frame1[:, :, channel].flatten(), 
                                            kernel.flatten(), mode='same')
                    processing_buffer[:, :, channel] = conv_result.reshape(height, width)
                
                # 4. Frame interpolation
                alpha = 0.5
                interpolated = frame1 * alpha + frame2 * (1 - alpha)
                
                # 5. Post-processing
                final_frame = cp.clip(interpolated * 255, 0, 255).astype(cp.uint8)
                
                frames_processed += 1
                
                # Rotate frames for continuous processing
                frame_buffer[0], frame_buffer[1] = frame_buffer[1], frame_buffer[0]
                
                # Clear intermediate results
                del frame1, frame2, motion_vector, interpolated, final_frame
            
            actual_duration = time.time() - start_time
            fps = frames_processed / actual_duration
            
            result = {
                'resolution': f"{width}x{height}",
                'duration_seconds': actual_duration,
                'frames_processed': frames_processed,
                'fps': fps,
                'success': True
            }
            
            print(f"  📊 Processed {frames_processed} frames in {actual_duration:.1f}s")
            print(f"  🚀 Average FPS: {fps:.2f}")
            
            return result
            
        except Exception as e:
            return {
                'resolution': f"{width}x{height}",
                'error': str(e),
                'success': False
            }
    
    def run_full_benchmark(self, output_file: Optional[str] = None) -> Dict:
        """Run complete benchmark suite"""
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Cannot run benchmark - missing dependencies")
            return {}
        
        print("🔥 Starting GPU-Only RIFE Benchmark Suite...")
        print("=" * 60)
        
        self.start_time = datetime.now()
        
        # Test resolutions
        test_resolutions = [
            (640, 480),    # SD
            (1280, 720),   # HD
            (1920, 1080),  # Full HD
            (2560, 1440),  # 1440p
            (3840, 2160),  # 4K (if memory permits)
        ]
        
        # Filter resolutions based on available memory
        if self.gpu_info and self.gpu_info['total_memory'] < 12:
            test_resolutions = test_resolutions[:3]  # Skip 1440p and 4K for GPUs < 12GB
            print("ℹ️  Limiting test resolutions due to GPU memory constraints")
        
        results = {
            'timestamp': self.start_time.isoformat(),
            'system_info': {
                'gpu': self.gpu_info,
                'cpu': self.cpu_info if hasattr(self, 'cpu_info') else {}
            },
            'test_resolutions': [f"{w}x{h}" for w, h in test_resolutions]
        }
        
        # Run benchmark components
        try:
            results['memory_usage'] = self.benchmark_memory_usage(test_resolutions)
            results['tensor_operations'] = self.benchmark_tensor_operations(test_resolutions)
            
            # Real-world scenario (test on primary resolution)
            primary_resolution = (1920, 1080) if (1920, 1080) in test_resolutions else test_resolutions[-1]
            results['real_world'] = self.benchmark_real_world_scenario(primary_resolution, duration=5)
            
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        # Calculate summary
        results['summary'] = self._generate_summary(results)
        
        # Save results
        if output_file:
            self._save_results(results, output_file)
        
        print("\n" + "=" * 60)
        print("📋 BENCHMARK COMPLETE!")
        self._print_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate benchmark summary"""
        summary = {
            'overall_score': 0,
            'recommendations': [],
            'performance_rating': 'Unknown'
        }
        
        try:
            # Calculate overall performance score
            score = 0
            factors = 0
            
            # Memory efficiency score
            if 'memory_usage' in results:
                memory_scores = []
                for res, data in results['memory_usage'].items():
                    if data.get('success') and data.get('memory_mb', 0) < 2000:  # Under 2GB
                        memory_scores.append(100)
                    elif data.get('success'):
                        memory_scores.append(max(0, 100 - (data.get('memory_mb', 0) - 2000) / 10))
                
                if memory_scores:
                    score += sum(memory_scores) / len(memory_scores)
                    factors += 1
            
            # FPS score
            if 'real_world' in results and results['real_world'].get('success'):
                fps = results['real_world'].get('fps', 0)
                if fps > 60:
                    fps_score = 100
                elif fps > 30:
                    fps_score = 80 + (fps - 30) / 30 * 20
                elif fps > 15:
                    fps_score = 60 + (fps - 15) / 15 * 20
                else:
                    fps_score = max(0, fps / 15 * 60)
                
                score += fps_score
                factors += 1
            
            if factors > 0:
                summary['overall_score'] = score / factors
            
            # Performance rating
            if summary['overall_score'] >= 90:
                summary['performance_rating'] = 'Excellent'
            elif summary['overall_score'] >= 75:
                summary['performance_rating'] = 'Good'
            elif summary['overall_score'] >= 60:
                summary['performance_rating'] = 'Fair'
            else:
                summary['performance_rating'] = 'Needs Optimization'
            
            # Generate recommendations
            if 'real_world' in results and results['real_world'].get('fps', 0) < 30:
                summary['recommendations'].append("Consider reducing processing quality or resolution")
            
            if 'memory_usage' in results:
                max_memory = max([data.get('memory_mb', 0) for data in results['memory_usage'].values() 
                                if data.get('success')])
                if max_memory > 8000:  # > 8GB
                    summary['recommendations'].append("High memory usage detected - consider memory optimization")
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _print_summary(self, results: Dict):
        """Print benchmark summary to console"""
        summary = results.get('summary', {})
        
        print(f"🏆 Overall Score: {summary.get('overall_score', 0):.1f}/100")
        print(f"⭐ Rating: {summary.get('performance_rating', 'Unknown')}")
        
        if 'real_world' in results and results['real_world'].get('success'):
            fps = results['real_world'].get('fps', 0)
            print(f"🎬 Real-world FPS: {fps:.2f}")
        
        if summary.get('recommendations'):
            print("💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
    
    def _save_results(self, results: Dict, filename: str):
        """Save benchmark results to JSON file"""
        try:
            # Ensure the filename has .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="GPU-Only RIFE Benchmark System")
    parser.add_argument("--output", "-o", type=str, 
                       default=f"rife_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                       help="Output file for results (default: timestamped filename)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (fewer resolutions and shorter duration)")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = GPUBenchmark()
    
    if args.quick:
        print("⚡ Running quick benchmark...")
        # Override test parameters for quick run
        benchmark.quick_mode = True
    
    results = benchmark.run_full_benchmark(output_file=args.output)
    
    return results

if __name__ == "__main__":
    main()
