# file: rife_engine.py
# Traditional RIFE implementation using rife-ncnn-vulkan
# Note: For pure GPU implementations, see:
#   - gpu_only_rtx_5060_ti.py (optimized for RTX 5060 Ti)
#   - rtx_5060_ti_optimized_rife.py (alternative GPU-only version)
#   - final_gpu_only_rife.py (complete GPU implementation)
﻿import os
import subprocess
import shutil
import time

class RIFEEngine:
    def __init__(self, model="rife-v4.6"):
        self.base_dir = os.path.join(os.getcwd(), "rife-ncnn-vulkan-20221029-windows")
        self.rife_executable = os.path.join(self.base_dir, "rife-ncnn-vulkan.exe")
        self.ffmpeg_executable = r"C:\Users\0-0\Desktop\addon\face traking\blender_VT\New folder\gpu_Test\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
        self.model = model
        self.temp_dir = os.path.join(os.getcwd(), "temp_frames")
        self.temp_audio_dir = os.path.join(os.getcwd(), "temp_audio")
        self.interpolated_dir = os.path.join(os.getcwd(), "interpolated_frames")
        print(f"RIFE Engine initialized - Executable: {self.rife_executable}")
        
    def find_models(self):
        models_dir = os.path.join(self.base_dir)
        try:
            models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            return models
        except FileNotFoundError:
            return []

    def test_rife_executable(self):
        """Test if RIFE executable works"""
        try:
            result = subprocess.run([self.rife_executable], capture_output=True, text=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False

    def is_v4_model(self, model_name):
        """Check if model is RIFE v4 or newer (supports -n parameter)"""
        v4_models = ['rife-v4', 'rife-v4.6', 'rife-v4.0', 'rife-v4.1', 'rife-v4.2', 'rife-v4.3', 'rife-v4.4', 'rife-v4.5']
        return any(v4_model in model_name.lower() for v4_model in v4_models)

    def interpolate_multiple_times(self, input_dir, output_dir, model_path, times=1):
        """Run RIFE interpolation multiple times for higher factors"""
        current_input = input_dir
        
        for i in range(times):
            temp_output = os.path.join(os.getcwd(), f"temp_interp_{i}")
            os.makedirs(temp_output, exist_ok=True)
            
            # Run RIFE interpolation
            cmd = [self.rife_executable, "-i", current_input, "-o", temp_output, "-m", model_path]
            print(f"Running RIFE pass {i+1}/{times}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"RIFE stderr: {result.stderr}")
                print(f"RIFE stdout: {result.stdout}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
            
            # Clean up previous temp directory
            if i > 0:
                shutil.rmtree(current_input, ignore_errors=True)
            
            current_input = temp_output
        
        # Move final result to output directory
        if current_input != output_dir:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.move(current_input, output_dir)

    def interpolate(self, input_path, output_path, factor=2, progress_callback=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist")
        if not os.path.exists(self.rife_executable):
            raise FileNotFoundError(f"RIFE executable not found at {self.rife_executable}")
        if not os.path.exists(self.ffmpeg_executable):
            raise FileNotFoundError(f"FFmpeg executable not found at {self.ffmpeg_executable}")
        
        # Clean up old directories
        for temp_path in [self.temp_dir, self.interpolated_dir, self.temp_audio_dir]:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
            os.makedirs(temp_path, exist_ok=True)
        
        output_dir = output_path
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_output_filename = f"{input_filename}_interpolated_{factor}x_{timestamp}.mp4"
        final_output_path = os.path.join(output_dir, final_output_filename)
        
        try:
            # Extract audio to a separate directory
            audio_path = os.path.join(self.temp_audio_dir, "audio.aac")
            print("Extracting audio with FFmpeg...")
            subprocess.run([self.ffmpeg_executable, "-i", input_path, "-vn", "-acodec", "copy", audio_path], check=True)
            if progress_callback: progress_callback(20)
            
            # Extract frames
            frame_pattern = os.path.join(self.temp_dir, "frame_%06d.png")
            print("Extracting frames with FFmpeg...")
            subprocess.run([self.ffmpeg_executable, "-i", input_path, "-q:v", "2", frame_pattern], check=True)
            if progress_callback: progress_callback(40)
            
            # Check extracted frames
            frame_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
            print(f"Extracted {len(frame_files)} frames")
            
            # Run RIFE interpolation
            print(f"Running RIFE interpolation with model: {self.model}")
            model_path = os.path.join(self.base_dir, self.model)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model folder {model_path} not found")
            
            # Choose RIFE command based on model version and factor
            if self.is_v4_model(self.model) and factor == 2:
                # RIFE v4 models support -n parameter for 2x interpolation
                cmd = [
                    self.rife_executable, "-i", self.temp_dir, "-o", self.interpolated_dir,
                    "-m", model_path, "-n", "1"
                ]
                print(f"Using v4 command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"RIFE stderr: {result.stderr}")
                    print(f"RIFE stdout: {result.stdout}")
                    raise subprocess.CalledProcessError(result.returncode, cmd)
            else:
                # For older models or higher factors, run multiple times
                if factor == 2:
                    times = 1
                elif factor == 4:
                    times = 2
                elif factor == 8:
                    times = 3
                else:
                    times = 1  # Default to 1 for unsupported factors
                
                print(f"Using basic interpolation, running {times} time(s)")
                self.interpolate_multiple_times(self.temp_dir, self.interpolated_dir, model_path, times)
            
            if progress_callback: progress_callback(70)
            
            # --- NEW CODE START ---
            print("Renaming interpolated frames for FFmpeg...")
            files_to_rename = sorted([f for f in os.listdir(self.interpolated_dir) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))
            for i, filename in enumerate(files_to_rename):
                old_path = os.path.join(self.interpolated_dir, filename)
                new_path = os.path.join(self.interpolated_dir, f"frame_{i+1:06d}.png")
                os.rename(old_path, new_path)
            # --- NEW CODE END ---
            
            # Check interpolated frames
            interpolated_files = [f for f in os.listdir(self.interpolated_dir) if f.endswith('.png')]
            print(f"Generated {len(interpolated_files)} interpolated frames")
            
            if len(interpolated_files) == 0:
                raise Exception("No interpolated frames were generated")
            
            # Calculate target framerate
            original_fps = 30  # Assuming 30fps, could extract this from video
            target_fps = original_fps * factor
            
            # Reassemble video
            print("Reassembling video with FFmpeg...")
            
            interpolated_frame_pattern = os.path.join(self.interpolated_dir, "frame_%06d.png")
            
            cmd = [
                self.ffmpeg_executable, "-r", str(target_fps), "-i", interpolated_frame_pattern, 
                "-i", audio_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                "-c:a", "aac", "-shortest", final_output_path
            ]
            
            print(f"Running reassembly with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if progress_callback: progress_callback(100)
            
            print(f"Interpolation complete! Output saved to {final_output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error during processing: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
        finally:
            # Clean up temp directories
            for temp_path in [self.temp_dir, self.interpolated_dir, self.temp_audio_dir]:
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
            # Clean up any remaining temp interpolation directories
            for i in range(10):
                temp_interp = os.path.join(os.getcwd(), f"temp_interp_{i}")
                if os.path.exists(temp_interp):
                    shutil.rmtree(temp_interp, ignore_errors=True)
